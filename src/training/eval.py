"""Sequential manifest evaluation through the shared AgentRollout."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

from rewards.shopsim_reward import METRIC_KEYS
from training.rollout import AgentRollout, RolloutResult
from training.runtime import append_metrics


SAMPLING_SEED_STRATEGY = "per_episode_manifest_index_v1"


def episode_seed(base_seed: int, manifest_index: int) -> int:
    if (type(base_seed) is not int or type(manifest_index) is not int
            or base_seed < 0 or manifest_index < 0 or base_seed + manifest_index >= 2**32):
        raise ValueError("episode seed requires nonnegative integers with sum < 2**32")
    return base_seed + manifest_index


def seed_episode(seed: int) -> None:
    # Evaluator only: never reseed QwenPolicy.generate or GRPO group sampling.
    from transformers import set_seed
    set_seed(seed)  # Python, NumPy, torch CPU and all CUDA devices, once per episode.


def load_task_ids(path: str | Path, *, scenario: str | None = None,
                  split: str | None = None) -> list[str]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    items = value.get("tasks") if isinstance(value, Mapping) else value
    if not isinstance(items, list) or not items:
        raise ValueError("task manifest must contain a non-empty task list")
    task_ids = []
    for item in items:
        if scenario is not None and (not isinstance(item, Mapping) or item.get("scenario") != scenario):
            raise ValueError("task manifest scenario mismatch")
        if split is not None and (not isinstance(item, Mapping) or item.get("official_split") != split):
            raise ValueError("task manifest official_split mismatch")
        task_id = item.get("task_id") if isinstance(item, Mapping) else item
        if task_id is None or not str(task_id):
            raise ValueError("task manifest missing task_id")
        task_ids.append(str(task_id))
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task manifest contains duplicate task_id")
    return task_ids


def episode_summary(result: RolloutResult) -> dict[str, Any]:
    # Keep full visible conversation in memory for one episode only, never evaluator-only payloads.
    row = asdict(result)
    row["response_characters"] = sum(map(len, row.pop("visible_responses")))
    row.pop("messages")
    row.pop("token_trace")  # Training-only capture does not change evaluation output.
    return row


def prompt_profile(policy: Any, messages: list[dict[str, str]]) -> dict[str, int] | None:
    """Count the actual final rendered generation input; never retokenize sampled output.

    Historical assistant text is rendered by evaluation's existing native template.
    Context counts are not a claim that this history equals GRPO's append-only stream.
    """
    import re
    tokenizer = getattr(policy, "tokenizer", None)
    if tokenizer is None:
        return None
    kwargs = dict(policy.generation.chat_template_kwargs)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    actual = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs)
    if encoded["input_ids"] != actual:
        raise ValueError("profiling template IDs disagree with generation input")
    # Role headers are externally inserted. Content + im_end inside historical
    # assistant blocks are assistant context, including native thinking rendering.
    spans = [(m.start(1), m.end(1)) for m in re.finditer(
        r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", text, re.DOTALL)]
    assistant = sum(hi > lo and any(start <= lo < hi <= end for start, end in spans)
                    for lo, hi in encoded["offset_mapping"])
    return {"rendered_input_tokens": len(actual),
            "observation_header_tokens": len(actual) - assistant}


def completed_rows(path: Path, task_ids: list[str], scenario: str,
                   *, base_seed: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if ([str(row["task_id"]) for row in rows] != task_ids[:len(rows)]
            or any(row["scenario"] != scenario for row in rows)):
        raise ValueError("saved episodes are not a unique ordered prefix of this manifest")
    if base_seed is not None:
        for index, row in enumerate(rows):
            if (row.get("manifest_index") != index
                    or row.get("episode_seed") != episode_seed(base_seed, index)):
                raise ValueError("saved episode seed/manifest_index mismatch")
    return rows


def evaluation_summary(rows, *, scenario, reward_alpha, expected, invocation_s, errors):
    count = len(rows)
    statuses = Counter(row["status"] for row in rows)
    means = {key: sum(row["reward_metrics"][key] for row in rows) / count if count else None
             for key in METRIC_KEYS}
    wall = sum(row["wall_time_s"] for row in rows)
    generation_s = sum(row["generation_time_s"] for row in rows)
    generated = [row["generated_tokens"] for row in rows if row["generated_tokens"] is not None]
    diagnostics = {"invocation_wall_s": invocation_s, "completed_episode_wall_s": wall,
                   "infrastructure_errors": sum(e["event"] == "infrastructure_error" for e in errors),
                   "error_count": len(errors),
                   "action_count": sum(len(row["actions"]) for row in rows),
                   "invalid_action_count": sum(row["invalid_action_count"] for row in rows),
                   "malformed_action_count": sum(row["malformed_action_count"] for row in rows),
                   "max_steps_count": statuses.get("max_steps", 0),
                   "context_limit_count": statuses.get("context_limit", 0),
                   "finish_count": sum(row["reward_metrics"]["r_finish"] == 1 for row in rows),
                   "success_count": sum(row["reward_metrics"]["r_succ"] == 1 for row in rows),
                   "trajectories_per_hour": count / wall * 3600 if wall else None,
                   "generation_tokens_per_second": sum(generated) / generation_s
                       if len(generated) == count and generation_s else None}
    for key in ("steps", "wall_time_s", "generation_count", "generation_time_s", "environment_wait_s",
                "response_characters", "input_tokens", "generated_tokens", "full_trajectory_tokens",
                "observation_header_tokens", "generation_cap_count", "context_window_cap_count"):
        values = [row[key] for row in rows if row.get(key) is not None]
        diagnostics[f"mean_{key}"] = sum(values) / len(values) if values else None
        diagnostics[f"total_{key}"] = sum(values) if len(values) == count and count else None
    return {"scenario": scenario, "reward_alpha": reward_alpha, "episodes": count,
            "expected_episodes": expected, "complete": count == expected,
            "metrics": means, "status_counts": dict(statuses), "diagnostics": diagnostics}


def evaluate_policy(*, policy: Any, scenario: str, task_ids: Iterable[str], env_factory: Any,
                    reward_alpha: float = 1.0, max_action_steps: int = 30,
                    output_dir: str | Path | None = None, resume: bool = False,
                    base_seed: int = 1,
                    sampling_seed_strategy: str = SAMPLING_SEED_STRATEGY) -> dict[str, Any]:
    from env.teacher_env_client import TeacherEnvError
    tasks = list(map(str, task_ids))
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("evaluation requires nonempty unique task IDs")
    if sampling_seed_strategy != SAMPLING_SEED_STRATEGY:
        raise ValueError("unsupported evaluation sampling_seed_strategy")
    episode_seed(base_seed, len(tasks) - 1)
    path = Path(output_dir) if output_dir is not None else None
    if resume and path is None:
        raise ValueError("resume requires output_dir")
    if path:
        path.mkdir(parents=True, exist_ok=True)
        if (path / "episodes.jsonl").exists() and not resume:
            raise FileExistsError("evaluation episodes already exist in output_dir")
    rows = completed_rows(path / "episodes.jsonl", tasks, scenario,
                          base_seed=base_seed) if path and resume else []
    errors = ([json.loads(line) for line in (path / "errors.jsonl").read_text().splitlines()]
              if path and (path / "errors.jsonl").exists() else [])

    class ObservedPolicy:
        last_messages = None
        caps = 0
        context_caps = 0
        turn = 0

        @property
        def last_usage(self):
            return policy.last_usage

        @property
        def last_generation(self):
            return getattr(policy, "last_generation", {})

        def generate(self, messages):
            self.last_messages = [dict(m) for m in messages]
            response = policy.generate(messages)
            usage = getattr(policy, "last_usage", {})
            limit = getattr(getattr(policy, "generation", None), "max_new_tokens", None)
            generation = self.last_generation
            normal_cap = generation.get("normal_generation_cap", limit is not None and usage.get("generated_tokens") == limit)
            self.caps += int(normal_cap)
            self.context_caps += int(generation.get("context_window_cap", False))
            self.turn += 1
            if path:
                append_metrics(path / "responses.jsonl", {
                    "task_id": current_task, "turn": self.turn, "captured_at_ns": time.time_ns(),
                    "manifest_index": manifest_index, "episode_seed": current_seed,
                    "visible_response": response, "generated_tokens": usage.get("generated_tokens"),
                    "at_token_cap": normal_cap,
                    **generation,
                })
            print(json.dumps({"event": "turn", "task_id": current_task,
                              "generated_tokens": usage.get("generated_tokens"),
                              "at_token_cap": normal_cap,
                              **{key: generation.get(key) for key in (
                                  "input_tokens_before_generation", "configured_max_new_tokens",
                                  "effective_max_new_tokens", "remaining_context_tokens",
                                  "context_budget_reduced", "eos_reached", "context_window_cap",
                                  "normal_generation_cap", "context_limit", "model_called")}}), flush=True)
            return response

    observed = ObservedPolicy()
    runner = AgentRollout(policy=observed, env_factory=env_factory, scenario=scenario,
                          reward_alpha=reward_alpha, max_action_steps=max_action_steps)
    started = time.monotonic()
    current_task = None

    def save_summary():
        result = evaluation_summary(rows, scenario=scenario, reward_alpha=reward_alpha, expected=len(tasks),
                                    invocation_s=time.monotonic() - started, errors=errors)
        if path:
            temp = path / "summary.json.tmp"
            temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            temp.replace(path / "summary.json")
        return result

    try:
        for manifest_index, current_task in enumerate(tasks[len(rows):], start=len(rows)):
            current_seed = episode_seed(base_seed, manifest_index)
            seed_episode(current_seed)
            print(json.dumps({"event": "task_start", "task_id": current_task,
                              "index": manifest_index + 1, "total": len(tasks),
                              "manifest_index": manifest_index, "episode_seed": current_seed}), flush=True)
            observed.caps = 0
            observed.context_caps = 0
            observed.turn = 0
            observed.last_messages = None
            result = runner.run(current_task)
            row = episode_summary(result)
            row.update(manifest_index=manifest_index, episode_seed=current_seed)
            row["generation_cap_count"] = observed.caps
            row["context_window_cap_count"] = observed.context_caps
            profile = prompt_profile(policy, observed.last_messages) if observed.last_messages else None
            row["full_trajectory_tokens"] = (profile["rendered_input_tokens"] + policy.last_usage["generated_tokens"]
                                             if profile else None)
            row["observation_header_tokens"] = profile["observation_header_tokens"] if profile else None
            # Completed rows survive Ctrl+C; incomplete current task gets a fresh reset on resume.
            if path:
                append_metrics(path / "episodes.jsonl", row)
            rows.append(row)
            save_summary()
            print(json.dumps({"event": "task_complete", "index": len(rows), "total": len(tasks),
                              **row}, ensure_ascii=False), flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        event = "interrupted" if isinstance(exc, KeyboardInterrupt) else (
            "infrastructure_error" if isinstance(exc, TeacherEnvError) else "error")
        error = {"event": event, "task_id": current_task, "error_type": type(exc).__name__}
        errors.append(error)
        if path:
            append_metrics(path / "errors.jsonl", error)
        save_summary()
        print(json.dumps(error), flush=True)
        raise
    return save_summary()

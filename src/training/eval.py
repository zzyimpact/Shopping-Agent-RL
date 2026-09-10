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
    return row


def evaluate_policy(*, policy: Any, scenario: str, task_ids: Iterable[str], env_factory: Any,
                    reward_alpha: float = 1.0, max_action_steps: int = 30,
                    output_dir: str | Path | None = None) -> dict[str, Any]:
    runner = AgentRollout(policy=policy, env_factory=env_factory, scenario=scenario,
                          reward_alpha=reward_alpha, max_action_steps=max_action_steps)
    path = Path(output_dir) if output_dir is not None else None
    if path:
        path.mkdir(parents=True, exist_ok=True)
        # Avoid mixing episodes from distinct evaluations. Run infrastructure handles training resumes.
        if (path / "episodes.jsonl").exists():
            raise FileExistsError("evaluation episodes already exist in output_dir")
    started = time.monotonic()
    rows = []
    for task_id in task_ids:
        row = episode_summary(runner.run(str(task_id)))
        rows.append(row)
        if path:
            append_metrics(path / "episodes.jsonl", row)
    if not rows:
        raise ValueError("evaluation requires at least one task")
    statuses = Counter(row["status"] for row in rows)
    means = {key: sum(row["reward_metrics"][key] for row in rows) / len(rows) for key in METRIC_KEYS}
    diagnostics = {"wall_time_s": time.monotonic() - started,
                   "invalid_action_count": sum(row["invalid_action_count"] for row in rows),
                   "malformed_action_count": sum(row["malformed_action_count"] for row in rows),
                   "max_steps_count": statuses.get("max_steps", 0)}
    for key in ("steps", "wall_time_s", "generation_count", "generation_time_s", "response_characters",
                "input_tokens", "generated_tokens"):
        values = [row[key] for row in rows if row[key] is not None]
        diagnostics[f"mean_{key}"] = sum(values) / len(values) if values else None
    summary = {"scenario": scenario, "reward_alpha": reward_alpha, "episodes": len(rows),
               "metrics": means, "status_counts": dict(statuses), "diagnostics": diagnostics}
    if path:
        (path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary

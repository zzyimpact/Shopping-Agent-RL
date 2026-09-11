import json
import os

import pytest

from env.teacher_env_client import TeacherEnvError
from rewards.shopsim_reward import METRIC_KEYS
from training.eval import (SAMPLING_SEED_STRATEGY, completed_rows, episode_seed,
                           evaluate_policy, load_task_ids, seed_episode)
from tests.training.test_rollout import FakeEnv, FakePolicy, step_payload


def test_aggregation_and_episode_output(tmp_path):
    envs = iter([FakeEnv([step_payload("done", done=True, success=True)]),
                 FakeEnv([step_payload("same", valid=False)])])
    summary = evaluate_policy(policy=FakePolicy(["Thought: x\nAction: click[Buy]"] * 2), scenario="single",
                              task_ids=["t1", "t2"], env_factory=lambda: next(envs), output_dir=tmp_path)
    assert summary["episodes"] == 2
    assert summary["metrics"] == dict.fromkeys(METRIC_KEYS, 0.5)
    assert summary["diagnostics"]["invalid_action_count"] == 1
    assert summary["diagnostics"]["mean_generated_tokens"] == 5
    rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert [row["task_id"] for row in rows] == ["t1", "t2"]
    assert "PRIVATE" not in json.dumps(rows)
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    responses = [json.loads(line) for line in (tmp_path / "responses.jsonl").read_text().splitlines()]
    assert [row["task_id"] for row in responses] == ["t1", "t2"]
    assert all(row["visible_response"].startswith("Thought:") for row in responses)


def test_manifest_split_and_scenario_are_not_interchangeable(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": [{"task_id": "t", "scenario": "single", "official_split": "test"}]}))
    assert load_task_ids(path, scenario="single", split="test") == ["t"]
    with pytest.raises(ValueError, match="split"):
        load_task_ids(path, scenario="single", split="train")
    with pytest.raises(ValueError, match="scenario"):
        load_task_ids(path, scenario="single_persona")


def test_evaluation_resume_preserves_completed_rows(tmp_path):
    envs = iter([FakeEnv([step_payload("done", done=True, success=True)])])
    evaluate_policy(policy=FakePolicy(["Thought: x\nAction: click[Buy]"]), scenario="single",
                    task_ids=["t1"], env_factory=lambda: next(envs), output_dir=tmp_path)
    envs = iter([FakeEnv([step_payload("done", done=True, success=True)])])
    summary = evaluate_policy(policy=FakePolicy(["Thought: x\nAction: click[Buy]"]), scenario="single",
                              task_ids=["t1"], env_factory=lambda: next(envs), output_dir=tmp_path, resume=True)
    assert summary["episodes"] == 1
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), TeacherEnvError("offline", kind="infrastructure")])
def test_interruption_resume_and_timing_denominator(tmp_path, failure):
    class BrokenEnv(FakeEnv):
        def step(self, *args, **kwargs):
            raise failure
    envs = iter([FakeEnv([step_payload("done", done=True, success=True)]), BrokenEnv([])])
    policy = FakePolicy(["Thought: x\nAction: click[Buy]"] * 2)
    with pytest.raises(type(failure)):
        evaluate_policy(policy=policy, scenario="single", task_ids=["t1", "t2"],
                        env_factory=lambda: next(envs), output_dir=tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["episodes"] == 1 and not summary["complete"]
    saved = (tmp_path / "episodes.jsonl").read_text()
    env = FakeEnv([step_payload("done", done=True, success=True)])
    summary = evaluate_policy(policy=FakePolicy(["Thought: x\nAction: click[Buy]"]),
                              scenario="single", task_ids=["t1", "t2"], env_factory=lambda: env,
                              output_dir=tmp_path, resume=True)
    assert env.task_id == "t2" and summary["episodes"] == 2 and summary["complete"]
    assert (tmp_path / "episodes.jsonl").read_text().startswith(saved)
    diagnostics = summary["diagnostics"]
    assert diagnostics["trajectories_per_hour"] == pytest.approx(7200 / diagnostics["completed_episode_wall_s"])
    assert diagnostics["error_count"] == 1
    with pytest.raises(ValueError, match="prefix"):
        evaluate_policy(policy=policy, scenario="single", task_ids=["t2", "t1"],
                        env_factory=lambda: env, output_dir=tmp_path, resume=True)


def test_cli_resume_identity_and_dry_run(tmp_path):
    import runpy
    from pathlib import Path
    script = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/eval_policy.py"))
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"tasks": [{"task_id": "t", "scenario": "single", "official_split": "test"}]}))
    for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}")
    args = ["--scenario", "single", "--model-path", str(tmp_path), "--manifest", str(manifest),
            "--output-dir", str(tmp_path / "run")]
    assert script["main"](args + ["--dry-run"]) == 0
    assert not (tmp_path / "run").exists()
    config = script["parse_config"](args)
    config.pop("dry_run"); config.pop("resume")
    _, inputs = script["validate_inputs"](config)
    root = script["prepare_eval_run"](config, inputs, resume=False)
    saved = json.loads((root / "run_manifest.json").read_text())["identity"]["config"]
    assert saved["generation"]["chat_template_kwargs"]["enable_thinking"] is False
    assert not saved["generation"]["do_sample"] and saved["generation"]["max_new_tokens"] == 512
    resumed = script["parse_config"](args + ["--resume"])
    assert resumed.pop("resume")
    resumed.pop("dry_run")
    assert script["prepare_eval_run"](resumed, inputs, resume=True) == root
    with pytest.raises(ValueError, match="identity"):
        script["prepare_eval_run"]({**resumed, "seed": 2}, inputs, resume=True)
    with pytest.raises(ValueError, match="frozen P2"):
        script["validate_inputs"]({**config, "fixed_128": True})
    health = {"status": "ok", "environment_version": "task-scoped-v3-multisession"}
    for split in (None, "train"):
        with pytest.raises(ValueError, match="TEST-only"):
            script["validate_eval_health"]({**health, "task_split": split})
    script["validate_eval_health"]({**health, "task_split": "test"})
    (root / "invalidation.json").write_text('{"status":"INVALIDATED_BY_GENERATION_CONTRACT"}')
    with pytest.raises(ValueError, match="DO NOT RESUME"):
        script["main"](args + ["--resume"])


def test_episode_seed_validation_and_cpu_rng():
    import random
    import numpy as np
    import torch
    assert [episode_seed(1, i) for i in range(3)] == [1, 2, 3]
    for base, index in [(-1, 0), (1, -1), (True, 0), (1, 2**32 - 1)]:
        with pytest.raises(ValueError, match="episode seed"):
            episode_seed(base, index)
    def draw():
        return random.random(), float(np.random.random()), float(torch.rand(()))
    seed_episode(2)
    first, second = draw(), draw()
    assert first != second
    seed_episode(2)
    assert (draw(), draw()) == (first, second)


def test_sampled_uninterrupted_equals_interrupted_resume(tmp_path):
    import random
    import numpy as np
    import torch

    class SamplingPolicy:
        last_usage = {"input_tokens": 10, "generated_tokens": 5}

        def generate(self, messages):
            draws = (random.random(), float(np.random.random()), float(torch.rand(())))
            return f"Thought: {draws!r}\nAction: click[Buy]"

    class InterruptedEnv(FakeEnv):
        def step(self, *args, **kwargs):
            # Interrupt after the incomplete episode has already consumed RNG.
            raise KeyboardInterrupt()

    def fresh_env():
        return FakeEnv([step_payload("next"), step_payload("done", done=True, success=True)])

    def run(path, factory, resume=False):
        return evaluate_policy(policy=SamplingPolicy(), scenario="single", task_ids=["t1", "t2", "t3"],
                               env_factory=factory, output_dir=path, resume=resume, base_seed=1)

    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run(full, fresh_env)
    envs = iter([fresh_env(), InterruptedEnv([])])
    with pytest.raises(KeyboardInterrupt):
        run(resumed, lambda: next(envs))
    saved = (resumed / "episodes.jsonl").read_bytes()
    seed_episode(999)  # A new process/model initialization need not preserve the old stream.
    torch.rand(100)
    summary = run(resumed, fresh_env, resume=True)
    assert summary["episodes"] == 3
    assert (resumed / "episodes.jsonl").read_bytes().startswith(saved)

    def read(path, name):
        return [json.loads(line) for line in (path / name).read_text().splitlines()]

    for path in (full, resumed):
        rows = read(path, "episodes.jsonl")
        assert [(row["manifest_index"], row["episode_seed"]) for row in rows] == [(0, 1), (1, 2), (2, 3)]
    for task in ("t2", "t3"):
        uninterrupted = [r for r in read(full, "responses.jsonl") if r["task_id"] == task]
        restarted = [r for r in read(resumed, "responses.jsonl") if r["task_id"] == task][-2:]
        for key in ("visible_response", "manifest_index", "episode_seed", "turn"):
            assert [r[key] for r in uninterrupted] == [r[key] for r in restarted]
        assert restarted[0]["visible_response"] != restarted[1]["visible_response"]
    with pytest.raises(ValueError, match="seed/manifest_index"):
        completed_rows(resumed / "episodes.jsonl", ["t1", "t2", "t3"], "single", base_seed=2)
    with pytest.raises(ValueError, match="sampling_seed_strategy"):
        evaluate_policy(policy=SamplingPolicy(), scenario="single", task_ids=["t1"],
                        env_factory=fresh_env, sampling_seed_strategy="global")


def test_formal_config_and_sampled_resume_guards(tmp_path):
    from copy import deepcopy
    from pathlib import Path
    import runpy
    project = Path(__file__).resolve().parents[2]
    script = runpy.run_path(str(project / "scripts/eval_policy.py"))
    # Synthetic fixture only; the real frozen hash is checked by remote CLI dry-run.
    manifest = tmp_path / "tasks.json"
    tasks = [{"task_id": str(i), "scenario": "single", "official_split": "test"} for i in range(128)]
    manifest.write_text(json.dumps({"tasks": tasks}))
    import hashlib
    script["FIXED_IDS_HASH"]["single"] = hashlib.sha256("\n".join(str(i) for i in range(128)).encode()).hexdigest()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}")
    args = ["--config", str(project / "configs/training/eval_formal.yaml"), "--scenario", "single",
            "--model-path", str(tmp_path), "--manifest", str(manifest), "--fixed-128",
            "--output-dir", str(tmp_path / "formal")]
    assert script["main"](args + ["--dry-run"]) == 0
    assert not (tmp_path / "formal").exists()
    config = script["parse_config"](args)
    config.pop("resume"); config.pop("dry_run")
    assert config["generation"] == {
        "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        "max_new_tokens": 512, "max_context_tokens": 32768,
        "chat_template_kwargs": {"enable_thinking": False}}
    assert config["sampling_seed_strategy"] == SAMPLING_SEED_STRATEGY
    assert config["base_seed"] == config["seed"] == 1
    assert config["use_model_defaults"] is False
    _, inputs = script["validate_inputs"](config)
    root = script["prepare_eval_run"](config, inputs, resume=False)
    for filename in ("config.json", "run_manifest.json"):
        saved = json.loads((root / filename).read_text())
        assert (saved if filename == "config.json" else saved["identity"]["config"]) == config
    assert script["prepare_eval_run"](config, inputs, resume=True) == root
    for key, value in [("base_seed", 2), ("seed", 2), ("sampling_seed_strategy", "global"),
                       ("model_path", "/other"), ("adapter_path", "/other_adapter"),
                       ("generation", {**config["generation"], "do_sample": False})]:
        with pytest.raises(ValueError, match="identity"):
            script["prepare_eval_run"]({**config, key: value}, inputs, resume=True)
    for key in ("task_manifest_sha256", "task_ids_sha256", "model_config_sha256"):
        with pytest.raises(ValueError, match="identity"):
            script["prepare_eval_run"](config, {**inputs, key: "changed"}, resume=True)
    for generation_key, value in [("do_sample", False), ("max_new_tokens", 1024), ("top_k", 0)]:
        altered = deepcopy(config)
        altered["generation"][generation_key] = value
        with pytest.raises(ValueError, match="frozen non-thinking"):
            script["validate_inputs"](altered)
    for key, value in [("sampling_seed_strategy", "global"), ("use_model_defaults", True), ("base_seed", 2)]:
        with pytest.raises(ValueError):
            script["validate_inputs"]({**config, key: value})
    previous = json.loads((root / "run_manifest.json").read_text())
    previous["git_commit"] = "different-commit"
    (root / "run_manifest.json").write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="original project commit"):
        script["prepare_eval_run"](config, inputs, resume=True)
    (root / "audit_status.json").write_text('{"status":"PAUSED_FOR_BASELINE_SEMANTICS_AUDIT"}')
    with pytest.raises(ValueError, match="DO NOT RESUME"):
        script["main"](args + ["--resume"])


@pytest.mark.skipif(not os.environ.get("QWEN_TOKENIZER_PATH"), reason="local tokenizer opt-in; no weights")
def test_real_eval_profile_native_template():
    from types import SimpleNamespace
    from transformers import AutoTokenizer
    from training.eval import prompt_profile
    from training.policy import GenerationConfig
    tokenizer = AutoTokenizer.from_pretrained(os.environ["QWEN_TOKENIZER_PATH"], local_files_only=True)
    policy = SimpleNamespace(tokenizer=tokenizer, generation=GenerationConfig())
    messages = [{"role": "system", "content": "Shop"}, {"role": "user", "content": "page1"},
                {"role": "assistant", "content": "Thought: search\nAction: search[shoes]"},
                {"role": "user", "content": "page2"}]
    counts = prompt_profile(policy, messages)
    assert counts["rendered_input_tokens"] == len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
    assert 0 < counts["observation_header_tokens"] < counts["rendered_input_tokens"]

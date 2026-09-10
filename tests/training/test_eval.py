import json

import pytest

from rewards.shopsim_reward import METRIC_KEYS
from training.eval import evaluate_policy, load_task_ids
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


def test_manifest_split_and_scenario_are_not_interchangeable(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": [{"task_id": "t", "scenario": "single", "official_split": "test"}]}))
    assert load_task_ids(path, scenario="single", split="test") == ["t"]
    with pytest.raises(ValueError, match="split"):
        load_task_ids(path, scenario="single", split="train")
    with pytest.raises(ValueError, match="scenario"):
        load_task_ids(path, scenario="single_persona")

from pathlib import Path

import pytest

from env.teacher_env_client import EnvResult, TeacherEnvError
from rollout.profiler import _run_attempt
from rollout.progress import ProgressLogger
from rollout.storage import GracefulCollectionStop, TeacherLedger
from rollout.teacher_client import TeacherResponse
from tests.rollout.test_storage import manifest


class FakeEnv:
    def __init__(self, fail=False):
        self.resets = 0
        self.steps = 0
        self.fail = fail

    def reset(self, scenario, task_id):
        self.resets += 1
        return EnvResult({"session_id": f"s{self.resets}", "observation": "obs", "policy_context": {
            "system_prompt": "prompt", "source": "upstream.py",
            "prompt_hash": __import__("hashlib").sha256(b"prompt").hexdigest(),
        }, "instruction": "hidden"}, 0.01)

    def step(self, session, response):
        self.steps += 1
        if self.fail:
            raise TeacherEnvError("remote environment request failed")
        return EnvResult({"action": "click[p1]", "observation": "done", "done": True,
                          "reward": 1.0, "reward_detail": {"r_type": 1, "r_att": 1, "r_option": 1, "r_price": 1,
                          "query_match": False}, "purchase": {"asin": "p1"}, "goal": {}}, 0.02)


class FakeClient:
    def __init__(self, text="Thought: go\nAction: click[p1]"):
        self.text = text
    def generate(self, messages):
        return TeacherResponse(self.text, "model-a", 200, "req", 4, 2, 0.01, 0, "chat_completions")


def test_successful_termination_and_fresh_reset(tmp_path):
    with TeacherLedger(tmp_path, manifest(), profile=True) as ledger:
        guard = GracefulCollectionStop(ledger, "resume")
        result = _run_attempt(ledger=ledger, env=FakeEnv(), client=FakeClient(), task_id="task-1", scenario="single",
                              phase="first_success", attempt_index=1, expected_model="model-a",
                              logger=ProgressLogger(), guard=guard)
        assert result == "success"
        assert len(list(ledger.paths.trajectories.glob("*.json"))) == 1


def test_malformed_action_and_environment_failure_are_separate(tmp_path):
    with TeacherLedger(tmp_path / "m", manifest(), profile=True) as ledger:
        guard = GracefulCollectionStop(ledger, "resume")
        assert _run_attempt(ledger=ledger, env=FakeEnv(), client=FakeClient("plain text"), task_id="t1", scenario="single",
                            phase="first_success", attempt_index=1, expected_model="model-a", logger=ProgressLogger(), guard=guard) == "malformed_action"
    with TeacherLedger(tmp_path / "e", manifest(), profile=True) as ledger:
        guard = GracefulCollectionStop(ledger, "resume")
        assert _run_attempt(ledger=ledger, env=FakeEnv(fail=True), client=FakeClient(), task_id="t2", scenario="single",
                            phase="first_success", attempt_index=1, expected_model="model-a", logger=ProgressLogger(), guard=guard) == "environment_failure"


def test_model_mismatch_is_run_stop(tmp_path):
    with TeacherLedger(tmp_path, manifest(), profile=True) as ledger:
        guard = GracefulCollectionStop(ledger, "resume")
        with pytest.raises(RuntimeError, match="expected"):
            _run_attempt(ledger=ledger, env=FakeEnv(), client=FakeClient(), task_id="task-1", scenario="single",
                         phase="first_success", attempt_index=1, expected_model="different", logger=ProgressLogger(), guard=guard)


def test_profile_driver_runs_fixed_plan_with_fake_clients(tmp_path, monkeypatch):
    import json
    import rollout.profiler as profiler

    task_file = tmp_path / "tasks.json"
    task_file.write_text(json.dumps({"metadata": {"task_ids_sha256": "ids", "source_manifest_sha256": "source", "seed": 1},
                                     "tasks": [{"task_id": f"t{i}", "official_split": "train"} for i in range(24)]}))
    env_file = tmp_path / ".env.teacher"
    env_file.write_text("\n".join(["TEACHER_API_URL=http://local.invalid", "TEACHER_API_KEY=fixture",
                                   "TEACHER_API_MODEL=model-a", "TEACHER_API_STYLE=responses",
                                   "TEACHER_REASONING_EFFORT=high"]))

    class DriverEnv(FakeEnv):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def health(self):
            return EnvResult({"status": "ok", "source_fingerprint": "source-fp"}, 0.001)
        def release(self, session):
            return EnvResult({"released": True}, 0.001)
        def close(self):
            pass

    class DriverClient(FakeClient):
        def __init__(self, **kwargs):
            self.text = "Thought: go\nAction: click[p1]"
        def close(self):
            pass

    monkeypatch.setattr(profiler, "TeacherEnvClient", DriverEnv)
    monkeypatch.setattr(profiler, "TeacherClient", DriverClient)
    assert profiler.run_profile(scenario="single", env_endpoint="http://fake", task_file=task_file,
                                data_root=tmp_path / "data", env_file=env_file) == 0
    runs = list((tmp_path / "data" / "single").glob("*/"))
    assert len(runs) == 1
    assert len(list((runs[0] / "trajectories").glob("*.json"))) == 72

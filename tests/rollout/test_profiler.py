from pathlib import Path
import io

import pytest

from env.teacher_env_client import EnvResult, TeacherEnvError
from rollout.profiler import _find_resume_run, _run_attempt
from rollout.progress import ProgressLogger
from rollout.storage import GracefulCollectionStop, TeacherLedger
from rollout.teacher_client import TeacherResponse
from tests.rollout.test_storage import manifest


class FakeEnv:
    def __init__(self, fail=False):
        self.resets = 0
        self.steps = 0
        self.fail = fail
        self.releases = 0

    def reset(self, scenario, task_id):
        self.resets += 1
        return EnvResult({"session_id": f"s{self.resets}", "observation": "obs",
                          "policy_observation": "obs\n\n搜索功能是否可用: True\n\n可点击的按钮: []",
                          "policy_observation_version": "single-eval-policy-v1",
                          "profiler_protocol_version": "p3a-visible-action-v2", "policy_context": {
            "system_prompt": "prompt", "source": "upstream.py",
            "prompt_hash": __import__("hashlib").sha256(b"prompt").hexdigest(),
        }, "instruction": "hidden"}, 0.01)

    def step(self, session, response):
        self.steps += 1
        if self.fail:
            raise TeacherEnvError("remote environment request failed")
        return EnvResult({"action": "click[p1]", "observation": "done",
                          "policy_observation": "done\n\n搜索功能是否可用: False\n\n可点击的按钮: []",
                          "action_valid": True, "done": True,
                          "reward": 1.0, "reward_detail": {"r_type": 1, "r_att": 1, "r_option": 1, "r_price": 1,
                          "query_match": False}, "purchase": {"asin": "p1"}, "goal": {}}, 0.02)

    def release(self, session):
        self.releases += 1
        return EnvResult({"released": True}, 0.001)


class FakeClient:
    def __init__(self, text="Thought: go\nAction: click[p1]"):
        self.text = text
    def generate(self, messages):
        return TeacherResponse(self.text, "model-a", 200, "req", 4, 2, 0.01, 0, "chat_completions")


class NeverDoneEnv(FakeEnv):
    def step(self, session, response):
        self.steps += 1
        return EnvResult({"action": "search[x]", "observation": f"obs-{self.steps}",
                          "policy_observation": f"obs-{self.steps}\n\n搜索功能是否可用: True\n\n可点击的按钮: []",
                          "done": False,
                          "action_valid": True}, 0.001)


def test_successful_termination_and_fresh_reset(tmp_path):
    env = FakeEnv()
    output = io.StringIO()
    with TeacherLedger(tmp_path, manifest(), profile=True) as ledger:
        logger = ProgressLogger(output)
        guard = GracefulCollectionStop(ledger, "resume", logger)
        result = _run_attempt(ledger=ledger, env=env, client=FakeClient(), task_id="task-1", scenario="single",
                              phase="first_success", attempt_index=1, expected_model="model-a",
                              logger=logger, guard=guard)
        assert result == "success"
        assert len(list(ledger.paths.trajectories.glob("*.json"))) == 1
    assert env.resets == 1
    assert env.releases == 1
    timing = output.getvalue()
    assert "[TIMING] task-1 phase=first_success attempt 1" in timing
    assert "reset=" in timing and "API=" in timing and "env_steps=" in timing and "total=" in timing


def test_max_action_steps_is_30(tmp_path):
    env = NeverDoneEnv()
    with TeacherLedger(tmp_path, manifest(), profile=True) as ledger:
        guard = GracefulCollectionStop(ledger, "resume")
        result = _run_attempt(ledger=ledger, env=env, client=FakeClient(), task_id="task-steps", scenario="single",
                              phase="first_success", attempt_index=1, expected_model="model-a",
                              logger=ProgressLogger(), guard=guard)
        assert result == "max_steps"
        assert env.steps == 30


def test_malformed_action_and_environment_failure_are_separate(tmp_path):
    with TeacherLedger(tmp_path / "m", manifest(), profile=True) as ledger:
        guard = GracefulCollectionStop(ledger, "resume")
        assert _run_attempt(ledger=ledger, env=FakeEnv(), client=FakeClient("plain text"), task_id="t1", scenario="single",
                            phase="first_success", attempt_index=1, expected_model="model-a", logger=ProgressLogger(), guard=guard) == "malformed_action"
    with TeacherLedger(tmp_path / "malformed", manifest(), profile=True) as ledger:
        guard = GracefulCollectionStop(ledger, "resume")
        env = FakeEnv()
        assert _run_attempt(ledger=ledger, env=env, client=FakeClient("Thought only\nAction: foo"), task_id="t-malformed", scenario="single",
                            phase="first_success", attempt_index=1, expected_model="model-a", logger=ProgressLogger(), guard=guard) == "malformed_action"
        assert env.steps == 0
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
    task_ids = [f"t{i}" for i in range(24)]
    task_file.write_text(json.dumps({"metadata": {
        "task_ids_sha256": __import__("hashlib").sha256(("\n".join(task_ids) + "\n").encode()).hexdigest(),
        "source_manifest_sha256": "source", "seed": 1},
        "tasks": [{"task_id": task_id, "official_split": "train", "scenario": "single"}
                  for task_id in task_ids]}))
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


def test_resume_selects_latest_incomplete_run(tmp_path):
    import json
    import sqlite3

    scenario_root = tmp_path / "single"
    for run_id, status, created_at in (
        ("run-old", "infrastructure_interrupted", "2026-01-01T00:00:00+00:00"),
        ("run-new", "stopped", "2026-01-02T00:00:00+00:00"),
    ):
        run_root = scenario_root / run_id
        run_root.mkdir(parents=True)
        (run_root / "run_manifest.json").write_text(json.dumps({
            "purpose": "p3a_profiling", "scenario": "single", "created_at": created_at,
        }))
        db = sqlite3.connect(run_root / "state.sqlite")
        db.execute("CREATE TABLE run_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute(
            "CREATE TABLE attempts (attempt_id TEXT PRIMARY KEY, task_id TEXT, status TEXT, "
            "artifact_path TEXT, started_at TEXT, finished_at TEXT, teacher_attempt INTEGER)"
        )
        db.execute("INSERT INTO run_state VALUES ('status', ?)", (status,))
        db.commit()
        db.close()

    selected, older_count = _find_resume_run(tmp_path, "single", [])
    assert selected is not None
    assert selected.run_id == "run-new"
    assert selected.status == "stopped"
    assert selected.touched_tasks == 0
    assert older_count == 1

"""SQLite ledger、atomic artifact、resume 与 graceful stop 测试。"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from rollout.progress import ProgressLogger
from rollout.storage import GracefulCollectionStop, ResumeConfigMismatch, TeacherLedger


def manifest(**overrides):
    base = {
        "run_id": "run-001", "scenario": "single", "teacher_model": "model-a",
        "api_style": "chat_completions", "reasoning_effort": "high",
        "system_prompt_hash": "prompt-hash", "collection_config_hash": "config-hash",
        "shopsim_source_fingerprint": "source-hash", "environment_version": "task-scoped-v1",
        "reward_deviation_version": "query-match-false-v1",
    }
    base.update(overrides)
    return base


def test_sqlite_atomic_accept_and_resume(tmp_path):
    with TeacherLedger(tmp_path, manifest()) as ledger:
        attempt = ledger.start_attempt("task-1")
        attempt_path = ledger.save_attempt(attempt, {"events": [{"action": "search[x]"}]}, status="success")
        accepted_path = ledger.accept(attempt, {"messages": [], "reward": {"r_succ": 1}})
        assert attempt_path.exists() and accepted_path.exists()
        assert not list(accepted_path.parent.glob("*.tmp"))
        assert ledger.accepted_count() == 1
    with TeacherLedger(tmp_path, manifest(), resume=True) as resumed:
        assert resumed.accepted_count() == 1
        assert len(list(resumed.paths.accepted.glob("*.json"))) == 1


def test_interrupted_attempt_preserved_and_not_accepted(tmp_path):
    with TeacherLedger(tmp_path, manifest()) as ledger:
        attempt = ledger.start_attempt("task-2")
        path = ledger.mark_infrastructure_interrupted(attempt, {"events": [{"step": 4}]})
        record = json.loads(path.read_text())
        assert record["status"] == "infrastructure_interrupted"
        assert record["events"] == [{"step": 4}]
        assert ledger.accepted_count() == 0


def test_config_mismatch_refuses_resume(tmp_path):
    TeacherLedger(tmp_path, manifest()).close()
    with pytest.raises(ResumeConfigMismatch, match="teacher_model"):
        TeacherLedger(tmp_path, manifest(teacher_model="model-b"), resume=True)


def test_ctrl_c_flushes_partial_and_prints_resume(tmp_path):
    output = io.StringIO()
    ledger = TeacherLedger(tmp_path, manifest())
    attempt = ledger.start_attempt("task-3")
    guard = GracefulCollectionStop(ledger, "python collect.py --resume run-001", ProgressLogger(output))
    guard.current_attempt_id = attempt
    guard.current_record = {"events": [{"step": 2}]}
    guard.handle_sigint(2, None)
    record = json.loads(next(ledger.paths.attempts.glob("*.json")).read_text())
    assert guard.stop_requested is True
    assert record["status"] == "infrastructure_interrupted"
    with TeacherLedger(tmp_path, manifest(), resume=True) as resumed:
        assert resumed.get_state("status") == "infrastructure_interrupted"
    assert "Resume with:" in output.getvalue()


def test_profile_storage_is_separate_from_formal_teacher_raw(tmp_path):
    with TeacherLedger(tmp_path / "teacher_profile", manifest(), profile=True) as ledger:
        attempt = ledger.start_attempt("task-profile")
        ledger.save_attempt(attempt, {"success": True}, status="success")
        path = ledger.save_trajectory(attempt, {"success": True}, status="success")
        assert path.parent.name == "trajectories"
        assert not (tmp_path / "teacher_raw").exists()
    with TeacherLedger(tmp_path / "teacher_profile", manifest(), profile=True, resume=True) as resumed:
        assert resumed.completed_task_ids() == {"task-profile"}


def test_profile_extra_immutable_fields_refuse_resume(tmp_path):
    values = manifest(purpose="p3a_profiling", selected_task_list_hash="a")
    TeacherLedger(tmp_path, values, profile=True, extra_immutable_fields=("purpose", "selected_task_list_hash")).close()
    with pytest.raises(ResumeConfigMismatch, match="selected_task_list_hash"):
        TeacherLedger(tmp_path, manifest(purpose="p3a_profiling", selected_task_list_hash="b"), profile=True,
                      resume=True, extra_immutable_fields=("purpose", "selected_task_list_hash"))


def test_invalidated_profile_run_cannot_resume(tmp_path):
    values = manifest(purpose="p3a_profiling", profiler_protocol_version="old")
    with TeacherLedger(tmp_path, values, profile=True,
                       extra_immutable_fields=("purpose", "profiler_protocol_version")) as ledger:
        ledger.set_state("status", "invalidated_by_implementation_bug")
        payload = json.loads(ledger.paths.manifest.read_text(encoding="utf-8"))
        payload["status"] = "invalidated_by_implementation_bug"
        ledger.paths.manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResumeConfigMismatch, match="invalidated"):
        TeacherLedger(tmp_path, values, profile=True, resume=True,
                      extra_immutable_fields=("purpose", "profiler_protocol_version"))


def test_profile_unsolved_updates_artifact_and_ledger(tmp_path):
    with TeacherLedger(tmp_path, manifest(), profile=True) as ledger:
        attempt = ledger.start_attempt("task-unsolved")
        ledger.save_attempt(attempt, {"success": False, "termination_reason": "max_steps"}, status="max_steps")
        ledger.mark_profile_unsolved("task-unsolved")
        row = ledger.db.execute("SELECT status FROM attempts WHERE attempt_id=?", (attempt,)).fetchone()
        assert row[0] == "profile_unsolved"
        payload = json.loads((ledger.paths.attempts / f"{attempt}.json").read_text())
        assert payload["status"] == "profile_unsolved"
        assert payload["termination_reason"] == "profile_unsolved"

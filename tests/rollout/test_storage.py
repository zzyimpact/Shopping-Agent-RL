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
    assert "Resume with:" in output.getvalue()

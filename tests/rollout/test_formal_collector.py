"""Focused, no-paid checks for the P3c formal scheduler and ledger boundary."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading

from env.teacher_env_client import EnvResult
from rollout.collection_policy import load_collection_policy
from rollout.formal_collector import (
    CollectorLog,
    TaskInputs,
    _accept_candidate,
    _apply_attempt_to_state,
    _pass_c_work,
    _pass_b_work,
    initial_state,
    load_state,
    next_work,
    dump_state,
    run_engine,
)
from rollout.prompt import prompt_hash
from rollout.storage import TeacherLedger
from rollout.teacher_client import TeacherResponse


ROOT = Path(__file__).resolve().parents[2]
POLICY = load_collection_policy(ROOT / "configs/teacher/formal_collection_p3b_v1_1.yaml")


def _work_dict(item):
    return {
        "pass": item.pass_name, "task_id": item.task_id,
        "coverage_slot": item.coverage_slot, "source": item.source,
        "attempt_ordinal": item.attempt_ordinal,
    }


def _inputs() -> TaskInputs:
    tasks = [
        {"task_id": "p1", "domain": "家居", "category": "椅子"},
        {"task_id": "p2", "domain": "家居", "category": "桌子"},
        {"task_id": "p3", "domain": "数码", "category": "耳机"},
        {"task_id": "r1", "domain": "家居", "category": "椅子"},
    ]
    return TaskInputs(
        primary=tasks[:3], train=tasks, by_id={item["task_id"]: item for item in tasks},
        reserve_queues={'["家居","椅子"]': ["r1"]},
        primary_manifest_hash="primary", train_manifest_hash="train",
    )


def _ledger(tmp_path: Path) -> TeacherLedger:
    manifest = {
        "purpose": "formal_teacher_collection", "run_id": "test-run", "scenario": "single",
        "teacher_model": "gpt-5.6-sol", "api_style": "chat_completions",
        "reasoning_effort": "high", "system_prompt_hash": "prompt",
        "collection_config_hash": POLICY["identity"]["policy_hash"],
        "shopsim_source_fingerprint": "source", "environment_version": "task-scoped-v3-multisession",
        "reward_deviation_version": "query-match-false-v1", "policy_version": "p3b-v1.1",
        "policy_hash": POLICY["identity"]["policy_hash"], "workers": 2, "seed": 1,
        "sanitized_request_config_hash": "request", "primary_manifest_hash": "primary",
        "train_manifest_hash": "train", "policy_observation_version": "single-eval-policy-v1",
        "profiler_protocol_version": "p3a-visible-action-v2", "max_action_steps": 30,
        "environment_fingerprint": "source", "selected_primary_hash": "selected",
    }
    return TeacherLedger(tmp_path, manifest, extra_immutable_fields=(
        "purpose", "policy_version", "policy_hash", "workers", "seed",
        "sanitized_request_config_hash", "primary_manifest_hash", "train_manifest_hash",
        "policy_observation_version", "profiler_protocol_version", "max_action_steps",
        "environment_fingerprint", "selected_primary_hash",
    ))


def test_policy_v11_transport_delta_is_frozen():
    old = load_collection_policy(ROOT / "configs/teacher/formal_collection_p3b_v1.yaml")
    assert POLICY["identity"]["policy_version"] == "p3b-v1.1"
    assert POLICY["teacher"]["api_style"] == "chat_completions"
    assert POLICY["concurrency"]["workers_default"] == 8
    assert old["teacher"]["api_style"] == "responses"
    assert POLICY["scope"] == old["scope"]
    assert POLICY["targets"] == old["targets"]
    assert POLICY["attempt_budgets"] == old["attempt_budgets"]
    assert POLICY["passes"]["A"]["genuine_attempts_per_task_max"] == old["passes"]["A"]["genuine_attempts_per_task_max"]
    assert POLICY["passes"]["A"]["reserve"]["selection"] == old["passes"]["A"]["reserve"]["selection"]
    assert POLICY["passes"]["A"]["reserve"]["cross_stratum_fallback"] is False
    assert POLICY["passes"]["B"] == old["passes"]["B"]
    assert POLICY["passes"]["C"]["accepted_demo_count_priority"] == old["passes"]["C"]["accepted_demo_count_priority"]
    assert POLICY["passes"]["C"]["stratum_distribution"] == old["passes"]["C"]["stratum_distribution"]
    assert POLICY["quality"] == old["quality"]
    assert POLICY["diversity"] == old["diversity"]


def test_primary_two_failures_allocate_deterministic_same_stratum_reserve(tmp_path):
    inputs = _inputs()
    state = initial_state(inputs, target=3)
    logger = CollectorLog(tmp_path / "collector.log", scenario="single", run_id="test")
    first = next_work(state, inputs, logger, workers=1)
    assert first[0].task_id == "p1" and first[0].attempt_ordinal == 1
    _apply_attempt_to_state(state, _work_dict(first[0]), accepted=False,
                            status="terminal_unsuccessful", attempt_id="a1")
    second = next_work(state, inputs, logger, workers=1)
    assert second[0].task_id == "p1" and second[0].attempt_ordinal == 2
    _apply_attempt_to_state(state, _work_dict(second[0]), accepted=False,
                            status="terminal_unsuccessful", attempt_id="a2")
    reserve = next_work(state, inputs, logger, workers=1)
    assert reserve[0].task_id == "r1" and reserve[0].source == "reserve"
    assert state["reserve_allocations"] == {"r1": 0}


def test_b_duplicate_uses_recovery_and_c_never_overshoots(tmp_path):
    inputs = _inputs()
    state = initial_state(inputs, target=3)
    logger = CollectorLog(tmp_path / "collector.log", scenario="single", run_id="test")
    a = next_work(state, inputs, logger, workers=1)[0]
    _apply_attempt_to_state(state, _work_dict(a), accepted=True, status="accepted", attempt_id="a")
    state["current_pass"] = "B"
    b1 = _pass_b_work(state, workers=1)[0]
    assert b1.task_id == "p1" and b1.attempt_ordinal == 1
    _apply_attempt_to_state(state, _work_dict(b1), accepted=False,
                            status="rejected_exact_duplicate", attempt_id="b1")
    b2 = _pass_b_work(state, workers=1)[0]
    assert b2.task_id == "p1" and b2.attempt_ordinal == 2
    _apply_attempt_to_state(state, _work_dict(b2), accepted=True, status="accepted", attempt_id="b2")
    state["current_pass"] = "C"
    work = _pass_c_work(state, inputs, workers=8, remaining_quota=1)
    assert len(work) == 1
    assert len({item.task_id for item in work}) == len(work)


def test_primary_first_success_then_b1_reaches_two(tmp_path):
    inputs = _inputs()
    state = initial_state(inputs, target=3)
    logger = CollectorLog(tmp_path / "collector.log", scenario="single", run_id="test")
    a1 = next_work(state, inputs, logger, workers=1)[0]
    _apply_attempt_to_state(state, _work_dict(a1), accepted=True, status="accepted", attempt_id="a1")
    assert state["tasks"]["p1"]["accepted_count"] == 1
    state["current_pass"] = "B"
    b1 = _pass_b_work(state, workers=1)[0]
    _apply_attempt_to_state(state, _work_dict(b1), accepted=True, status="accepted", attempt_id="b1")
    assert state["tasks"]["p1"]["accepted_count"] == 2
    assert state["tasks"]["p1"]["b_done"] is True


def test_pass_c_respects_remaining_quota_and_max_three():
    inputs = _inputs()
    state = initial_state(inputs, target=4)
    state["current_pass"] = "C"
    state["tasks"]["p1"].update({"first_success": True, "accepted_count": 2, "post_attempts": 2})
    state["tasks"]["p2"].update({"first_success": True, "accepted_count": 1, "post_attempts": 1})
    state["tasks"]["p3"].update({"first_success": True, "accepted_count": 3, "post_attempts": 3})
    work = _pass_c_work(state, inputs, workers=8, remaining_quota=2)
    assert len(work) == 2
    assert [item.task_id for item in work] == ["p2", "p1"]
    assert all(state["tasks"][item.task_id]["accepted_count"] < 3 for item in work)


def test_state_round_trip_and_log_are_durable(tmp_path):
    inputs = _inputs()
    state = initial_state(inputs, target=3)
    ledger = _ledger(tmp_path)
    try:
        for pass_name in ("A", "B", "C"):
            state["current_pass"] = pass_name
            dump_state(ledger, state)
            restored = load_state(ledger)
            assert restored["target"] == 3 and restored["current_pass"] == pass_name
        logger = CollectorLog(tmp_path / "collector.log", scenario="single", run_id="test")
        logger.event("RUN_START", worker="worker-1", pass_name="A", task_id="p1", attempt=1)
        text = (tmp_path / "collector.log").read_text(encoding="utf-8")
        assert "RUN_START" in text and "worker=worker-1" in text
        assert "Authorization" not in text and "api_key" not in text.lower()
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/inspect_teacher_run.py"),
             "--scenario", "single", "--latest", "--last", "1",
             "--data-root", str(tmp_path)],
            check=True, text=True, capture_output=True,
        )
        assert "Run ID: test-run" in result.stdout
        assert "Recent collector.log entries" in result.stdout
    finally:
        ledger.close()


def test_exact_behavioral_duplicate_is_rejected(tmp_path):
    ledger = _ledger(tmp_path)
    record = {
        "task_id": "p1", "done": True, "terminal_purchase": {"asin": "p1"},
        "reward_metrics": {
            "r_loose": 1, "r_strict": 1, "r_succ": 1, "r_finish": 1,
            "r_category": 1, "r_attribute": 1, "r_option": 1, "r_price": 1,
        },
        "actions": ["click[p1]"], "policy_observations": ["before", "after"],
    }
    try:
        first = ledger.start_attempt("p1")
        ledger.save_attempt(first, record, status="candidate_success")
        assert _accept_candidate(ledger, first, dict(record)) == "accepted"
        second = ledger.start_attempt("p1")
        ledger.save_attempt(second, record, status="candidate_success")
        assert _accept_candidate(ledger, second, dict(record)) == "rejected_exact_duplicate"
        assert ledger.accepted_count() == 1
    finally:
        ledger.close()


def test_fake_client_real_engine_crosses_a_b_c_and_writes_artifacts(tmp_path):
    inputs = _inputs()
    inputs = TaskInputs(
        primary=inputs.primary[:2], train=inputs.train, reserve_queues=inputs.reserve_queues,
        by_id=inputs.by_id, primary_manifest_hash="primary", train_manifest_hash="train",
    )
    state = initial_state(inputs, target=5)
    ledger = _ledger(tmp_path)
    logger = CollectorLog(tmp_path / "collector.log", scenario="single", run_id="test-run")
    counter = iter(range(1, 20))
    lock = threading.Lock()

    class FakeTeacher:
        def generate(self, _messages):
            with lock:
                ordinal = next(counter)
            return TeacherResponse(
                text=f"Thought: fake\nAction: click[item-{ordinal}]", model="gpt-5.6-sol",
                status_code=200, request_id=f"request-{ordinal}", input_tokens=10,
                output_tokens=5, latency_s=0.001, retries=0, api_style="chat_completions",
            )

        def close(self):
            pass

    class FakeEnv:
        def reset(self, scenario, task_id):
            system = "fake system prompt"
            return EnvResult({
                "session_id": f"session-{task_id}", "scenario": scenario, "task_id": task_id,
                "policy_observation": "fake observation", "raw_observation": "fake observation",
                "policy_context": {"system_prompt": system, "source": "test", "prompt_hash": prompt_hash(system)},
            }, 0.001)

        def step(self, session_id, response, **_kwargs):
            task_id = session_id.removeprefix("session-")
            action = response.split("Action:", 1)[1].strip()
            return EnvResult({
                "session_id": session_id, "scenario": "single", "task_id": task_id,
                "action": action, "action_valid": True, "done": True,
                "policy_observation": "terminal", "raw_observation": "terminal",
                "purchase": {"asin": task_id}, "reward": 1.0,
                "reward_detail": {"r_category": 1, "r_attribute": 1, "r_option": 1, "r_price": 1},
            }, 0.001)

        def release(self, _session_id):
            return EnvResult({"released": True}, 0.0)

        def close(self):
            pass

    try:
        summary = run_engine(
            ledger=ledger, state=state, inputs=inputs, policy=POLICY, logger=logger,
            workers=2, client_factory=lambda _on_retry: FakeTeacher(), env_factory=FakeEnv,
        )
        assert summary["status"] == "complete"
        assert summary["accepted_trajectories"] == 5
        assert summary["current_pass"] == "C"
        assert len(list(ledger.paths.accepted.glob("*.json"))) == 5
        assert max(task["accepted_count"] for task in state["tasks"].values()) == 3
        log_text = (tmp_path / "collector.log").read_text(encoding="utf-8")
        assert "pass=A" in log_text and "pass=B" in log_text and "pass=C" in log_text
    finally:
        ledger.close()

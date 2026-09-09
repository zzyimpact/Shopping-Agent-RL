"""Focused, no-paid checks for the P3c formal scheduler and ledger boundary."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

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


# Event-driven scheduler tests: no timer-based ordering and no external clients.
class ScriptedRollouts:
    def __init__(self, monkeypatch, ledger, state, behavior):
        import rollout.formal_collector as formal
        self.ledger, self.state, self.behavior = ledger, state, behavior
        self.active = set()
        self.peak = 0
        self.started = []
        self.completed = []
        self.lock = threading.Lock()
        original_accept = formal._accept_candidate
        main_thread = threading.get_ident()

        def accept(*args):
            assert threading.get_ident() == main_thread
            return original_accept(*args)

        monkeypatch.setattr(formal, '_accept_candidate', accept)
        monkeypatch.setattr(formal, 'execute_rollout', self.execute)

    def execute(self, work, stop, **kwargs):
        from rollout.formal_collector import AttemptResult
        with self.lock:
            assert work.task_id not in self.active
            self.active.add(work.task_id)
            self.peak = max(self.peak, len(self.active))
            self.started.append((work.pass_name, work.task_id, work.attempt_ordinal))
        persisted = load_state(self.ledger)
        assert work.task_id in persisted['active_task_ids']
        assert dict(work.__dict__) in persisted['in_flight_work']
        if work.source == 'reserve':
            assert persisted['reserve_allocations'][work.task_id] == work.coverage_slot
        attempt_id = self.ledger.start_attempt(work.task_id, teacher_attempt=work.attempt_ordinal)

        def finish(status='candidate_success', variant=None):
            variant = variant or f'{work.pass_name}-{work.attempt_ordinal}'
            record = {
                'task_id': work.task_id, 'formal_work': _work_dict(work), 'done': True,
                'terminal_purchase': {'asin': work.task_id},
                'reward_metrics': dict.fromkeys((
                    'r_loose', 'r_strict', 'r_succ', 'r_finish', 'r_category',
                    'r_attribute', 'r_option', 'r_price'), 1),
                'actions': [f'search[{variant}]', f'click[{work.task_id}]'],
                'policy_observations': ['search', 'product', 'terminal'],
            }
            self.ledger.save_attempt(attempt_id, record, status=status)
            return AttemptResult(attempt_id, status)

        try:
            result = self.behavior(work, stop, finish)
            self.completed.append((work.pass_name, work.task_id, work.attempt_ordinal))
            return result
        finally:
            with self.lock:
                self.active.remove(work.task_id)


def _script_engine(tmp_path, monkeypatch, behavior, *, target=3, state_setup=None):
    ledger = _ledger(tmp_path)
    inputs = _inputs()
    state = initial_state(inputs, target=target)
    if state_setup:
        state_setup(ledger, state, inputs)
    logger = CollectorLog(tmp_path / 'collector.log', scenario='single', run_id='test-run')
    script = ScriptedRollouts(monkeypatch, ledger, state, behavior)
    return ledger, inputs, state, logger, script


def _run_script(ledger, inputs, state, logger):
    return run_engine(ledger=ledger, state=state, inputs=inputs, policy=POLICY,
                      logger=logger, workers=2, client_factory=None, env_factory=None)


def test_rolling_refills_before_slow_cleanup_and_serializes_retry(tmp_path, monkeypatch):
    third_started = threading.Event()
    slow_persisted = threading.Event()

    def behavior(work, stop, finish):
        if work.task_id == 'p1':
            result = finish()  # Persisted success, but future/session cleanup still active.
            slow_persisted.set()
            assert third_started.wait(5), 'batch barrier: p3 never started while p1 was active'
            return result
        assert slow_persisted.wait(5)
        if work.task_id == 'p2' and work.attempt_ordinal == 1:
            return finish('terminal_unsuccessful')
        if work.task_id == 'p3':
            assert state['tasks']['p2']['accepted_count'] == 1
            assert state['tasks']['p1']['accepted_count'] == 0
            assert 'p1' in script.active
            third_started.set()
        return finish()

    ledger, inputs, state, logger, script = _script_engine(tmp_path, monkeypatch, behavior)
    try:
        summary = _run_script(ledger, inputs, state, logger)
        assert summary['accepted_trajectories'] == 3
        assert script.peak == 2
        assert state['tasks']['p2']['first_attempts'] == 2
        assert script.completed.index(('A', 'p2', 2)) < script.completed.index(('A', 'p1', 1))
        assert not state['active_task_ids'] and not state['in_flight_work']
    finally:
        ledger.close()


def test_rolling_pass_barriers_duplicate_recovery_and_max_demos(tmp_path, monkeypatch):
    import rollout.formal_collector as formal
    gates = {name: threading.Event() for name in ('A', 'B')}
    observed_barriers = []
    original_next = formal.next_work

    def choose(state, *args, **kwargs):
        before = state['current_pass']
        work = original_next(state, *args, **kwargs)
        if not work and state['active_task_ids'] and before in gates:
            assert state['current_pass'] == before
            observed_barriers.append(before)
            gates[before].set()
        return work

    monkeypatch.setattr(formal, 'next_work', choose)

    def behavior(work, stop, finish):
        if work.task_id == 'p1' and work.pass_name in gates and work.attempt_ordinal == 1:
            assert gates[work.pass_name].wait(5)
        if work.pass_name == 'B':
            assert all(t['first_success'] for t in state['tasks'].values())
            # B1 identical to A; B2 must recover through real duplicate acceptance.
            return finish(variant='A-1' if work.attempt_ordinal == 1 else 'B-2')
        if work.pass_name == 'C':
            assert all(t['b_done'] for t in state['tasks'].values())
            assert all(row[0] == 'C' for row in script.started[-len(script.active):])
        return finish()

    ledger, inputs, state, logger, script = _script_engine(tmp_path, monkeypatch, behavior, target=8)
    try:
        summary = _run_script(ledger, inputs, state, logger)
        assert summary['status'] == 'complete' and summary['accepted_trajectories'] == 8
        assert set(observed_barriers) == {'A', 'B'}
        assert state['exact_duplicate_rejects'] == 3
        assert max(t['accepted_count'] for t in state['tasks'].values()) == 3
        assert max(t['post_attempts'] for t in state['tasks'].values()) == 3
        assert script.peak <= 2
    finally:
        ledger.close()


def test_rolling_pass_c_reserves_inflight_quota_and_refills_failure(tmp_path, monkeypatch):
    import rollout.formal_collector as formal
    release_slow = threading.Event()
    original_next = formal.next_work

    def choose(state, *args, **kwargs):
        work = original_next(state, *args, **kwargs)
        accepted = sum(t['accepted_count'] for t in state['tasks'].values())
        assert accepted + len(state['active_task_ids']) <= state['target']
        if accepted == 4 and state['active_task_ids']:
            assert not work
            release_slow.set()
        return work

    monkeypatch.setattr(formal, 'next_work', choose)

    def setup(ledger, state, inputs):
        # A real durable first success for each task, then resume directly in C.
        for task_id in ('p1', 'p2', 'p3'):
            task = state['tasks'][task_id]
            work = formal.WorkItem('A', task_id, task['coverage_slot'], 'primary', 1)
            attempt_id = ledger.start_attempt(task_id)
            record = {'task_id': task_id, 'formal_work': _work_dict(work),
                      'actions': [f'click[{task_id}]']}
            ledger.accept(attempt_id, record)
            _apply_attempt_to_state(state, _work_dict(work), accepted=True,
                                    status='accepted', attempt_id=attempt_id)
        state['current_pass'] = 'C'

    def behavior(work, stop, finish):
        assert work.pass_name == 'C'
        assert work.task_id != 'p3', 'quota reservation allowed an extra attempt'
        if work.task_id == 'p1':
            assert release_slow.wait(5)
        if work.task_id == 'p2' and work.attempt_ordinal == 1:
            return finish('terminal_unsuccessful')
        return finish()

    ledger, inputs, state, logger, script = _script_engine(
        tmp_path, monkeypatch, behavior, target=5, state_setup=setup)
    try:
        summary = _run_script(ledger, inputs, state, logger)
        assert summary['accepted_trajectories'] == 5
        assert script.started.count(('C', 'p2', 2)) == 1
        assert len(script.started) == 3  # one failure plus exactly two acquired successes
    finally:
        ledger.close()


@pytest.mark.parametrize("competing_slots", [False, True])
def test_rolling_reserve_unique_same_stratum_persisted(tmp_path, monkeypatch, competing_slots):
    def behavior(work, stop, finish):
        if work.task_id in {'p1', 'p2'}:
            return finish('terminal_unsuccessful')
        return finish()

    ledger, inputs, state, logger, script = _script_engine(tmp_path, monkeypatch, behavior, target=2)
    if competing_slots:
        inputs.primary[1]['category'] = '椅子'
        inputs.by_id['r2'] = {'task_id': 'r2', 'domain': '家居', 'category': '椅子'}
        inputs.reserve_queues['["家居","椅子"]'].append('r2')
        state.clear()
        state.update(initial_state(inputs, target=3))
    try:
        summary = _run_script(ledger, inputs, state, logger)
        assert summary['accepted_trajectories'] == (3 if competing_slots else 2)
        if competing_slots:
            assert set(state['reserve_allocations']) == {'r1', 'r2'}
            assert set(state['reserve_allocations'].values()) == {0, 1}
            assert sum(task == 'r2' for _, task, _ in script.started) == 1
        else:
            assert state['reserve_allocations'] == {'r1': 0}
            assert state['slots']['1']['exhausted']  # table slot cannot use chair reserve
        assert sum(task == 'r1' for _, task, _ in script.started) == 1
        assert state['tasks']['p1']['first_attempts'] == 2
        assert state['tasks']['p2']['first_attempts'] == 2
    finally:
        ledger.close()


@pytest.mark.parametrize("fatal", [True, False])
def test_rolling_fatal_drains_and_old_stopped_state_resumes(tmp_path, monkeypatch, fatal):
    from rollout.formal_collector import FormalRunStop
    both_started = threading.Event()
    fatal_saved = threading.Event()

    def behavior(work, stop, finish):
        if work.task_id == 'p1':
            both_started.set()
            assert fatal_saved.wait(5)
            return finish()  # already in-flight success is retained on global stop
        assert work.task_id == 'p2'
        assert both_started.wait(5)
        finish('infrastructure_interrupted')
        stop.set()
        fatal_saved.set()
        if fatal:
            raise FormalRunStop('fake infrastructure')
        from rollout.concurrency import WorkerCancelled
        raise WorkerCancelled('fake Ctrl+C')

    ledger, inputs, state, logger, script = _script_engine(tmp_path, monkeypatch, behavior)
    try:
        summary = _run_script(ledger, inputs, state, logger)
        assert summary['status'] == 'stopped'
        assert summary['accepted_trajectories'] == 1
        assert len(script.started) == 2 and not script.active
        assert state['tasks']['p2']['first_attempts'] == 0
        old_accepted = {p.name: p.read_bytes() for p in ledger.paths.accepted.glob('*.json')}
        old_attempts = set(p.name for p in ledger.paths.attempts.glob('*.json'))
        # A crash can leave a persisted reservation and partial attempt. Neither uses quota.
        stale = ledger.start_attempt('p2', teacher_attempt=1)
        ledger.save_progress(stale, {'task_id': 'p2', 'formal_work': {
            'pass': 'A', 'task_id': 'p2', 'coverage_slot': 1, 'source': 'primary',
            'attempt_ordinal': 1}})
        state['active_task_ids'] = ['p2']
        state.pop('in_flight_work', None)  # fixed-batch-era state has no new optional field
        dump_state(ledger, state)
        manifest = dict(ledger.manifest)
        fields = ledger.extra_immutable_fields
        ledger.close()
        # Legacy stopped state has no scheduler identity requirement; same run is reopened.
        ledger = TeacherLedger(tmp_path, manifest, resume=True, extra_immutable_fields=fields)
        restored = load_state(ledger)
        assert restored['status'] == 'stopped'
        restored['status'] = 'in_progress'
        script = ScriptedRollouts(monkeypatch, ledger, restored, lambda work, stop, finish: finish())
        summary = _run_script(ledger, inputs, restored, logger)
        assert summary['status'] == 'complete' and summary['accepted_trajectories'] == 3
        assert ledger.manifest['run_id'] == 'test-run'
        assert restored['genuine_attempts'] == 3
        assert all(task != 'p1' for _, task, _ in script.started)
        assert old_attempts.issubset(p.name for p in ledger.paths.attempts.glob('*.json'))
        assert all((ledger.paths.accepted / name).read_bytes() == data for name, data in old_accepted.items())
    finally:
        ledger.close()

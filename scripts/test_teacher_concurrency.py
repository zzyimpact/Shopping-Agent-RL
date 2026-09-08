#!/usr/bin/env python3
"""User-triggered concurrency probe (one paid attempt per explicit task).

This is deliberately separate from P3a profiling.  It writes only under
``data/teacher_concurrency_probe`` and has no retry/second-demo/acceptance
policy.  It is safe to run with ``--task-limit 2`` as the first paid smoke.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from env.teacher_env_client import TeacherEnvClient, TeacherEnvError  # noqa: E402
from rollout.concurrency import ConcurrentArtifactStore, GlobalStop, run_bounded  # noqa: E402
from rollout.prompt import (append_turn, assert_no_evaluator_leakage,  # noqa: E402
                            build_initial_messages, policy_context_from_reset)
from rollout.profiler import _metrics, _policy_observation, _valid_visible_action  # noqa: E402
from rollout.teacher_client import TeacherClient, TeacherClientError, load_env_file  # noqa: E402


MAX_ACTION_STEPS = 30
MAX_WORKERS = 32


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "task"


def _hash_ids(task_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest()


class _ApiGauge:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.maximum = 0

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.maximum = max(self.maximum, self.current)

    def leave(self) -> None:
        with self._lock:
            self.current = max(0, self.current - 1)


def _load_tasks(path: Path, limit: int) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("probe task config has no tasks")
    ids = [str(item["task_id"]) for item in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("probe task config contains duplicate task IDs")
    if data.get("metadata", {}).get("task_ids_sha256") != _hash_ids(ids):
        raise ValueError("probe task list hash mismatch")
    if limit < 1 or limit > len(ids):
        raise ValueError(f"--task-limit must be between 1 and {len(ids)}")
    return ids[:limit]


def _finish(record: dict[str, Any], *, outcome: str, started: float) -> dict[str, Any]:
    record["outcome"] = outcome
    record["finished_at"] = _now()
    record["wall_time_s"] = time.monotonic() - started
    return record


def _run_task(*, task_id: str, scenario: str, endpoint: str, cfg: Mapping[str, str],
              store: ConcurrentArtifactStore, stop: GlobalStop, gauge: _ApiGauge,
              expected_model: str) -> dict[str, Any]:
    started = time.monotonic()
    relative = f"tasks/{_safe(task_id)}.json"
    record: dict[str, Any] = {
        "purpose": "p3a_concurrency_probe",
        "scenario": scenario,
        "task_id": task_id,
        "status": "in_progress",
        "started_at": _now(),
        "steps": [],
        "api_calls": 0,
        "api_total_latency_s": 0.0,
        "environment_total_latency_s": 0.0,
        "retries": 0,
        "retry_events": [],
        "session_id": None,
        "internal_slot": None,
        "error": None,
    }
    store.write(relative, record)
    env: TeacherEnvClient | None = None
    client: TeacherClient | None = None
    session_id: str | None = None
    try:
        env = TeacherEnvClient(endpoint, timeout=60.0)
        reset = env.reset(scenario, task_id)
        session_id = str(reset.payload["session_id"])
        record["session_id"] = session_id
        record["internal_slot"] = reset.payload.get("internal_slot")
        context = policy_context_from_reset(reset.payload, scenario)
        messages = build_initial_messages(context, _policy_observation(reset.payload))
        assert_no_evaluator_leakage(messages)
        client = TeacherClient(
            api_url=cfg["TEACHER_API_URL"], api_key=cfg["TEACHER_API_KEY"],
            model=cfg["TEACHER_API_MODEL"], api_style=cfg["TEACHER_API_STYLE"],
            reasoning_effort=cfg["TEACHER_REASONING_EFFORT"],
        )
        for step_index in range(1, MAX_ACTION_STEPS + 1):
            stop.check_before_external_call()
            gauge.enter()
            api_started = time.monotonic()
            try:
                response = client.generate(messages)
            finally:
                gauge.leave()
            api_elapsed = time.monotonic() - api_started
            record["api_calls"] += 1
            record["api_total_latency_s"] += api_elapsed
            record["retries"] += response.retries
            record["retry_events"].extend(response.retry_events)
            if response.model and response.model != expected_model:
                record["error"] = "provider_model_mismatch"
                return _finish(record, outcome="provider_protocol_error", started=started)
            if not _valid_visible_action(response.text):
                record["steps"].append({"step": step_index, "visible_response": response.text,
                                        "outcome": "malformed_action", "api_latency_s": response.latency_s})
                return _finish(record, outcome="malformed_action", started=started)
            env_started = time.monotonic()
            result = env.step(session_id, response.text,
                              expected_task_id=task_id, expected_scenario=scenario)
            env_elapsed = time.monotonic() - env_started
            record["environment_total_latency_s"] += env_elapsed
            step_record = {
                "step": step_index, "visible_response": response.text,
                "action": result.payload.get("action"),
                "action_valid": result.payload.get("action_valid"),
                "available_action_count": len((result.payload.get("available_actions") or {}).get("clickables", [])),
                "api_latency_s": response.latency_s, "environment_latency_s": env_elapsed,
                "retries": response.retries,
            }
            record["steps"].append(step_record)
            if result.payload.get("action_valid") is False:
                return _finish(record, outcome="invalid_action", started=started)
            observation = _policy_observation(result.payload)
            messages = append_turn(messages, response.text, observation)
            if result.payload.get("done") or result.payload.get("over"):
                metrics = _metrics(result.payload)
                record["reward_metrics"] = metrics
                record["reward"] = result.payload.get("reward")
                record["success"] = metrics["r_succ"] == 1.0
                return _finish(record, outcome="success" if record["success"] else "terminal_unsuccessful", started=started)
        return _finish(record, outcome="max_steps", started=started)
    except (TeacherClientError, TeacherEnvError, ValueError, KeyError) as exc:
        kind = getattr(exc, "kind", type(exc).__name__)
        record["error"] = str(kind)
        if isinstance(exc, TeacherClientError):
            record["retries"] += exc.retries
            record["retry_events"].extend(exc.retry_events)
        if kind in {"infrastructure", "provider_protocol_error"}:
            stop.set()
            outcome = "infrastructure_interrupted"
        elif kind == "invalid_action":
            outcome = "invalid_action"
        else:
            outcome = "environment_error"
        return _finish(record, outcome=outcome, started=started)
    finally:
        if session_id and env is not None:
            try:
                env.release(session_id)
            except Exception:
                record.setdefault("release_error", True)
        if client is not None:
            client.close()
        if env is not None:
            env.close()
        store.update(relative, record)


def run_probe(*, scenario: str, endpoint: str, env_file: Path, config: Path,
              output_root: Path, workers: int, task_limit: int) -> int:
    if scenario != "single":
        raise ValueError("the first paid concurrency probe is pinned to the Single task config")
    cfg = load_env_file(str(env_file))
    required = ("TEACHER_API_URL", "TEACHER_API_KEY", "TEACHER_API_MODEL",
                "TEACHER_API_STYLE", "TEACHER_REASONING_EFFORT")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(".env.teacher 缺少: " + ", ".join(missing))
    task_ids = _load_tasks(config, task_limit)
    print("ABOUT TO MAKE REAL PAID TEACHER API CALLS")
    print(f"scenario={scenario}")
    print(f"workers={workers}")
    print(f"tasks={len(task_ids)}")
    print("max_attempts_per_task=1")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = output_root / scenario / f"{stamp}-{_hash_ids(task_ids)[:8]}"
    store = ConcurrentArtifactStore(run_root)
    manifest = {"purpose": "p3a_concurrency_probe", "scenario": scenario,
                "task_ids": task_ids, "task_ids_sha256": _hash_ids(task_ids),
                "workers": workers, "max_action_steps": MAX_ACTION_STEPS,
                "teacher_model": cfg["TEACHER_API_MODEL"],
                "api_style": cfg["TEACHER_API_STYLE"],
                "reasoning_effort": cfg["TEACHER_REASONING_EFFORT"],
                "created_at": _now(), "status": "in_progress"}
    store.write("run_manifest.json", manifest)
    stop = GlobalStop()
    gauge = _ApiGauge()
    started = time.monotonic()
    results = run_bounded(
        task_ids,
        lambda task_id, event: _run_task(task_id=task_id, scenario=scenario, endpoint=endpoint,
                                         cfg=cfg, store=store, stop=event, gauge=gauge,
                                         expected_model=cfg["TEACHER_API_MODEL"]),
        workers=workers, stop=stop,
    )
    rows = []
    for result in results:
        if result.value is not None:
            rows.append(result.value)
        else:
            rows.append({"task_id": result.item, "outcome": "cancelled" if result.cancelled else "worker_error",
                         "error": type(result.error).__name__ if result.error else None})
    makespan = time.monotonic() - started
    sum_wall = sum(float(row.get("wall_time_s", 0.0) or 0.0) for row in rows)
    statuses = Counter(str(row.get("outcome", "unknown")) for row in rows)
    retry_events = [event for row in rows for event in row.get("retry_events", [])]
    http_statuses = Counter(str(event.get("status_code")) for event in retry_events if event.get("status_code") is not None)
    aggregate = {"workers": workers, "tasks": len(task_ids), "makespan_s": makespan,
                 "sum_individual_task_wall_s": sum_wall,
                 "effective_parallel_speedup": (sum_wall / makespan) if makespan else None,
                 "trajectories_per_hour": (len([r for r in rows if r.get("outcome") == "success"]) / makespan * 3600) if makespan else 0,
                 "api_requests": sum(int(row.get("api_calls", 0) or 0) for row in rows),
                 "retry_events": len(retry_events), "http_status_counts": dict(http_statuses),
                 "max_simultaneous_api_requests": gauge.maximum,
                 "session_mismatch_count": sum(1 for row in rows if row.get("error") == "infrastructure"),
                 "environment_error_count": sum(1 for row in rows if row.get("outcome") == "environment_error"),
                 "outcomes": dict(statuses), "completed_at": _now()}
    store.write("aggregate.json", aggregate)
    manifest.update({"status": "stopped" if stop.is_set() else "complete", "aggregate": aggregate})
    store.update("run_manifest.json", manifest)
    print(f"Probe output: {run_root}")
    print(json.dumps(aggregate, ensure_ascii=False, sort_keys=True))
    return 2 if stop.is_set() else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="one-attempt teacher concurrency probe")
    parser.add_argument("--scenario", choices=("single",), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--task-limit", type=int, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:5500")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.teacher")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "teacher" / "p3a_concurrency_probe_tasks.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "teacher_concurrency_probe")
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")
    try:
        return run_probe(scenario=args.scenario, endpoint=args.endpoint, env_file=args.env_file,
                         config=args.config, output_root=args.output_root,
                         workers=args.workers, task_limit=args.task_limit)
    except Exception as exc:  # no provider body/URL in user-facing errors
        print(f"Probe failed locally: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

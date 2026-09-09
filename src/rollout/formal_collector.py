"""P3c formal teacher collector engine（P3b-v1.1）。

复用现有同步 TeacherClient、multi-session HTTP client、bounded thread pool 与
TeacherLedger。Scheduler/state/acceptance 在主线程串行提交；worker 仅执行一个 fresh
rollout，因此 reserve、same-task 与 6000 quota 不依赖分布式锁。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

from env.teacher_env_client import TeacherEnvClient, TeacherEnvError
from rollout.collection_policy import has_deterministic_no_progress_loop
from rollout.collection_policy import load_collection_policy
from rollout.concurrency import GlobalStop, WorkerCancelled, MAX_WORKERS
from rollout.diversity import behavior_fingerprint, exact_duplicate, similarity_features
from rollout.profiler import _metrics, _policy_observation, _valid_visible_action
from rollout.prompt import (append_turn, assert_no_evaluator_leakage,
                            build_initial_messages, policy_context_from_reset)
from rollout.protocol import trace_visible_action
from rollout.storage import TeacherLedger, atomic_json, canonical_hash
from rollout.teacher_client import TeacherClient, TeacherClientError, TeacherResponse


FORMAL_EXTRA_IMMUTABLE = (
    "purpose", "policy_version", "policy_hash", "seed",
    "primary_manifest_hash", "train_manifest_hash",
    "policy_observation_version", "profiler_protocol_version", "max_action_steps",
    "environment_fingerprint", "selected_primary_hash",
)
SCHEDULER_VERSION = "completion-driven-rolling-v1"
TERMINAL_RUN_STATES = {"complete", "quota_unmet"}
INFRA_STATUSES = {
    "infrastructure_interrupted", "provider_config_error", "provider_model_mismatch",
    "environment_fatal_error", "cancelled",
}
GENUINE_FAILURE_STATUSES = {
    "teacher_failure", "teacher_empty_response", "malformed_action", "invalid_action",
    "terminal_unsuccessful", "max_steps", "hygiene_reject", "rejected_exact_duplicate",
    "rejected_acceptance_integrity",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def task_ids_hash(tasks: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(item["task_id"]) for item in tasks).encode()).hexdigest()


def stratum_of(task: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(task.get("domain_zh", task.get("domain", "<missing>")) or "<missing>"),
        str(task.get("category", "<missing>") or "<missing>"),
    )


def stratum_key(task: Mapping[str, Any]) -> str:
    return json.dumps(stratum_of(task), ensure_ascii=False, separators=(",", ":"))


def load_manifest(path: Path, *, expected_count: int, expected_hash: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少 task manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    tasks = value.get("tasks") if isinstance(value, Mapping) else None
    if not isinstance(tasks, list) or len(tasks) != expected_count:
        raise ValueError(f"task manifest count 不一致: {path}")
    normalized = [dict(item) for item in tasks]
    ids = [str(item.get("task_id", "")) for item in normalized]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"task manifest task_id 缺失或重复: {path}")
    actual = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    recorded = value.get("metadata", {}).get("task_ids_sha256")
    if actual != expected_hash or recorded != expected_hash:
        raise ValueError(f"task manifest hash 不一致: {path}")
    for item, task_id in zip(normalized, ids):
        item["task_id"] = task_id
        item["domain_zh"] = stratum_of(item)[0]
    return normalized


@dataclass(frozen=True)
class TaskInputs:
    primary: list[dict[str, Any]]
    train: list[dict[str, Any]]
    reserve_queues: dict[str, list[str]]
    by_id: dict[str, dict[str, Any]]
    primary_manifest_hash: str
    train_manifest_hash: str


def load_task_inputs(
    manifest_dir: Path, policy: Mapping[str, Any], scenario: str,
    *, selected_primary_ids: Sequence[str] | None = None,
    allow_non_primary_selected: bool = False,
) -> TaskInputs:
    source = policy["task_sources"][scenario]
    train = load_manifest(
        manifest_dir / source["train_manifest"],
        expected_count=int(source["train_task_count"]),
        expected_hash=str(source["train_task_ids_sha256"]),
    )
    full_primary = load_manifest(
        manifest_dir / source["primary_manifest"],
        expected_count=int(source["primary_task_count"]),
        expected_hash=str(source["primary_task_ids_sha256"]),
    )
    by_id = {str(item["task_id"]): item for item in train}
    if selected_primary_ids is None:
        primary = full_primary
    else:
        wanted = [str(item) for item in selected_primary_ids]
        if len(wanted) != len(set(wanted)):
            raise ValueError("selected primary tasks 含重复 task_id")
        full_ids = {str(item["task_id"]) for item in full_primary}
        if not set(wanted).issubset(by_id):
            raise ValueError("selected primary task 不属于 frozen TRAIN manifest")
        if not allow_non_primary_selected and not set(wanted).issubset(full_ids):
            raise ValueError("selected primary task 不属于 frozen primary manifest")
        primary = [by_id[task_id] for task_id in wanted]
    primary_ids = {str(item["task_id"]) for item in full_primary}
    grouped: dict[str, list[str]] = defaultdict(list)
    seed = int(policy["identity"]["seed"])
    for item in train:
        task_id = str(item["task_id"])
        if task_id not in primary_ids:
            grouped[stratum_key(item)].append(task_id)
    # Hash rank is deterministic across Python/process versions and does not
    # mutate the frozen manifests.
    for key, ids in grouped.items():
        ids.sort(key=lambda task_id: hashlib.sha256(
            f"{seed}\0{key}\0{task_id}".encode("utf-8")
        ).hexdigest())
    return TaskInputs(
        primary=primary, train=train, reserve_queues=dict(grouped), by_id=by_id,
        primary_manifest_hash=str(source["primary_task_ids_sha256"]),
        train_manifest_hash=str(source["train_task_ids_sha256"]),
    )


class CollectorLog:
    """单个 append-only collector.log；只写 sanitized scalar context。"""

    def __init__(self, path: Path, *, scenario: str, run_id: str, stream: Any = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.scenario = scenario
        self.run_id = run_id
        self.stream = stream
        self._lock = threading.RLock()
        self.task_api_profiles: dict[str, str] = {}

    @staticmethod
    def _safe(value: Any) -> str:
        text = str(value if value is not None else "-").replace("\n", " ").replace("\r", " ")
        return "".join(ch for ch in text if ch.isalnum() or ch in "._:/=-")[:180] or "-"

    def event(self, event: str, **fields: Any) -> None:
        base = {
            "timestamp": utc_now(), "event": event, "scenario": self.scenario,
            "run_id": self.run_id, "worker": fields.pop("worker", "-"),
            "pass": fields.pop("pass_name", "-"), "task": fields.pop("task_id", "-"),
            "attempt": fields.pop("attempt", "-"),
        }
        base.update(fields)
        line = " ".join(f"{key}={self._safe(value)}" for key, value in base.items())
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
            if self.stream is not None:
                terminal = self._terminal_line(event, base)
                if terminal:
                    print(terminal, file=self.stream, flush=True)

    def _terminal_line(self, event: str, fields: Mapping[str, Any]) -> str | None:
        """Keep live output close to P3a profiling without echoing the full log."""
        if event in {"RUN_START", "RESUME"}:
            return f"[{event}] scenario={self.scenario} run_id={self.run_id} workers={fields.get('workers', '-')}"
        if event == "PASS_START":
            return f"[{self.scenario}] Pass {fields.get('pass', '-')} started"
        profile = self.task_api_profiles.get(str(fields.get("task", "")))
        prefix = f"[{self.scenario}]" + (f"[api={profile}]" if profile else "")
        if event == "ATTEMPT_START":
            return (
                f"{prefix} Task {fields.get('task', '-')} | Pass {fields.get('pass', '-')} "
                f"| Attempt {fields.get('attempt', '-')}"
            )
        if event == "API_RETRY":
            return (
                f"[API] task={fields.get('task', '-')} {fields.get('classification', 'retry')} "
                f"| retry={fields.get('retry_ordinal', '-')}"
            )
        if event in {"ACCEPT", "ATTEMPT_FAILURE", "REJECT_EXACT_DUPLICATE"}:
            outcome = {
                "ACCEPT": "accepted", "ATTEMPT_FAILURE": fields.get("classification", "failed"),
                "REJECT_EXACT_DUPLICATE": "exact_duplicate",
            }[event]
            return (
                f"{prefix} Task {fields.get('task', '-')} finished | outcome={outcome} "
                f"| wall={float(fields.get('wall_s', 0.0)):.2f}s "
                f"| API={float(fields.get('api_s', 0.0)):.2f}s | steps={fields.get('steps', 0)}"
            )
        if event == "PROGRESS":
            return f"[PROGRESS] pass={fields.get('pass', '-')} accepted={fields.get('accepted', 0)}"
        if event in {"CTRL_C", "INFRA_STOP", "QUOTA_UNMET", "COMPLETE"}:
            return f"[{event}] pass={fields.get('pass', '-')} accepted={fields.get('accepted', '-')}"
        return None

    def terminal_progress(self, state: Mapping[str, Any], *, accepted: int) -> None:
        """Print cumulative debugging progress without writing another log event."""
        if self.stream is None:
            return
        slots = state.get("slots", {})
        completed = sum(
            1 for slot in slots.values() if slot.get("fulfilled") or slot.get("exhausted")
        )
        total_tasks = len(slots)
        trajectories = int(state.get("genuine_attempts", 0))
        target = int(state.get("target", 0))
        task_pct = 100.0 * completed / total_tasks if total_tasks else 0.0
        success_rate = 100.0 * accepted / trajectories if trajectories else 0.0
        target_pct = 100.0 * accepted / target if target else 0.0
        with self._lock:
            print(
                f"[STATUS] tasks={completed}/{total_tasks} ({task_pct:.2f}%) "
                f"| successful_trajectories={accepted} | total_trajectories={trajectories} "
                f"| success_rate={success_rate:.2f}% "
                f"| success_target={accepted}/{target} ({target_pct:.2f}%)",
                file=self.stream, flush=True,
            )


@dataclass(frozen=True)
class WorkItem:
    pass_name: str
    task_id: str
    coverage_slot: int
    source: str
    attempt_ordinal: int


@dataclass(frozen=True)
class AttemptResult:
    attempt_id: str
    status: str


class FormalRunStop(RuntimeError):
    """Worker 已持久化 fatal/infra attempt，请求 global graceful stop。"""


def initial_state(inputs: TaskInputs, *, target: int) -> dict[str, Any]:
    slots: dict[str, Any] = {}
    tasks: dict[str, Any] = {}
    for index, item in enumerate(inputs.primary):
        task_id = str(item["task_id"])
        key = str(index)
        slots[key] = {
            "primary_task_id": task_id, "stratum": stratum_key(item),
            "current_task_id": task_id, "fulfilled": False, "exhausted": False,
            "reserve_chain": [],
        }
        tasks[task_id] = {
            "coverage_slot": index, "source": "primary", "stratum": stratum_key(item),
            "first_attempts": 0, "first_success": False, "accepted_count": 0,
            "post_attempts": 0, "b_attempts": 0, "b_done": False,
        }
    return {
        "state_version": "p3c-formal-state-v1", "status": "in_progress",
        "current_pass": "A", "target": target, "slots": slots, "tasks": tasks,
        "reserve_allocations": {}, "reserve_exhausted_strata": [],
        "processed_attempt_ids": [], "active_task_ids": [],
        "genuine_attempts": 0, "exact_duplicate_rejects": 0,
        "last_progress_accepted": 0,
        "started_at": utc_now(), "last_update": utc_now(), "last_success": None,
        "last_error": None,
    }


def dump_state(ledger: TeacherLedger, state: Mapping[str, Any]) -> None:
    value = dict(state)
    value["last_update"] = utc_now()
    ledger.set_state("formal_state", json.dumps(value, ensure_ascii=False, sort_keys=True))
    ledger.set_state("status", str(value["status"]))


def load_state(ledger: TeacherLedger) -> dict[str, Any] | None:
    raw = ledger.get_state("formal_state")
    return json.loads(raw) if raw else None


def _read_attempt(ledger: TeacherLedger, relative_path: str) -> dict[str, Any]:
    return json.loads((ledger.paths.root / relative_path).read_text(encoding="utf-8"))


def _accepted_for_task(ledger: TeacherLedger, task_id: str) -> list[dict[str, Any]]:
    with ledger._lock:
        rows = ledger.db.execute(
            "SELECT artifact_path FROM accepted WHERE task_id=? ORDER BY rowid", (task_id,)
        ).fetchall()
    return [_read_attempt(ledger, str(row[0])) for row in rows]


def _valid_purchase(record: Mapping[str, Any]) -> bool:
    purchase = record.get("terminal_purchase")
    return isinstance(purchase, Mapping) and bool(str(purchase.get("asin", "")).strip())


def _accept_candidate(ledger: TeacherLedger, attempt_id: str, record: dict[str, Any]) -> str:
    metrics = record.get("reward_metrics")
    reward_keys = {
        "r_loose", "r_strict", "r_succ", "r_finish", "r_category", "r_attribute",
        "r_option", "r_price",
    }
    if (record.get("done") is not True or not _valid_purchase(record)
            or not isinstance(metrics, Mapping) or not reward_keys.issubset(metrics)
            or float(metrics.get("r_succ", 0.0)) != 1.0):
        record.update({"accepted": False, "rejection_reason": "acceptance_integrity"})
        ledger.save_attempt(attempt_id, record, status="rejected_acceptance_integrity")
        return "rejected_acceptance_integrity"
    if has_deterministic_no_progress_loop(record["actions"], record["policy_observations"]):
        record.update({"accepted": False, "rejection_reason": "deterministic_no_progress_loop"})
        ledger.save_attempt(attempt_id, record, status="hygiene_reject")
        return "hygiene_reject"
    fingerprint = behavior_fingerprint(
        record["actions"], final_purchase_asin=str(record["terminal_purchase"]["asin"])
    )
    prior = _accepted_for_task(ledger, str(record["task_id"]))
    prior_fingerprints = [item.get("behavior_fingerprint", {}) for item in prior]
    is_exact = any(exact_duplicate(fingerprint, item) for item in prior_fingerprints)
    similarities = [similarity_features(fingerprint, item) for item in prior_fingerprints]
    near = any(bool(item.get("provisional_near_duplicate")) for item in similarities)
    record.update({
        "behavior_fingerprint": fingerprint, "exact_duplicate": is_exact,
        "provisional_near_duplicate": near, "similarity_features": similarities,
    })
    if is_exact:
        record.update({"accepted": False, "rejection_reason": "exact_duplicate"})
        ledger.save_attempt(attempt_id, record, status="rejected_exact_duplicate")
        return "rejected_exact_duplicate"
    record.update({"accepted": True, "rejection_reason": None})
    ledger.save_attempt(attempt_id, record, status="candidate_success")
    ledger.accept(attempt_id, record)
    return "accepted"


def _apply_attempt_to_state(
    state: dict[str, Any], work: Mapping[str, Any], *, accepted: bool, status: str,
    attempt_id: str,
) -> None:
    task_id = str(work["task_id"])
    task = state["tasks"][task_id]
    pass_name = str(work["pass"])
    if status not in INFRA_STATUSES:
        state["genuine_attempts"] += 1
        if pass_name == "A":
            task["first_attempts"] += 1
        else:
            task["post_attempts"] += 1
            if pass_name == "B":
                task["b_attempts"] += 1
    if accepted:
        task["accepted_count"] += 1
        state["last_success"] = utc_now()
        if pass_name == "A":
            task["first_success"] = True
            state["slots"][str(work["coverage_slot"])]["fulfilled"] = True
        elif pass_name == "B":
            task["b_done"] = True
    elif pass_name == "B" and task["b_attempts"] >= 2:
        task["b_done"] = True
    if status == "rejected_exact_duplicate":
        state["exact_duplicate_rejects"] += 1
    state["processed_attempt_ids"].append(attempt_id)


def reconcile_attempts(
    ledger: TeacherLedger, state: dict[str, Any], logger: CollectorLog,
) -> None:
    """Exactly-once apply terminal attempts; also repairs crash-after-accept."""
    processed = set(state["processed_attempt_ids"])
    active = set(state.get("active_task_ids", []))
    with ledger._lock:
        rows = ledger.db.execute(
            "SELECT attempt_id,status,artifact_path,task_id FROM attempts ORDER BY rowid"
        ).fetchall()
    for attempt_id, status, relative_path, task_id in rows:
        # A worker may have persisted its result but still be releasing its session.
        # Reconcile only after its future has returned, never an active sibling.
        if task_id in active or attempt_id in processed or status == "in_progress":
            continue
        record = _read_attempt(ledger, str(relative_path))
        work = record.get("formal_work")
        if not isinstance(work, Mapping):
            continue
        applied_status = str(status)
        accepted = status == "accepted"
        if status == "candidate_success":
            applied_status = _accept_candidate(ledger, str(attempt_id), record)
            accepted = applied_status == "accepted"
        _apply_attempt_to_state(
            state, work, accepted=accepted, status=applied_status, attempt_id=str(attempt_id)
        )
        if applied_status == "accepted":
            logger.event("ACCEPT", pass_name=work["pass"], task_id=work["task_id"],
                         attempt=work["attempt_ordinal"],
                         wall_s=record.get("trajectory_attempt_wall_time_s", 0.0),
                         api_s=record.get("api_latency_total_s", 0.0),
                         steps=len(record.get("actions", [])))
        elif applied_status == "rejected_exact_duplicate":
            logger.event("REJECT_EXACT_DUPLICATE", pass_name=work["pass"],
                         task_id=work["task_id"], attempt=work["attempt_ordinal"],
                         wall_s=record.get("trajectory_attempt_wall_time_s", 0.0),
                         api_s=record.get("api_latency_total_s", 0.0),
                         steps=len(record.get("actions", [])))
        elif applied_status not in INFRA_STATUSES:
            logger.event("ATTEMPT_FAILURE", pass_name=work["pass"], task_id=work["task_id"],
                         attempt=work["attempt_ordinal"], classification=applied_status,
                         wall_s=record.get("trajectory_attempt_wall_time_s", 0.0),
                         api_s=record.get("api_latency_total_s", 0.0),
                         steps=len(record.get("actions", [])))
        if applied_status not in INFRA_STATUSES:
            logger.terminal_progress(state, accepted=ledger.accepted_count())
    dump_state(ledger, state)


def mark_stale_in_progress(ledger: TeacherLedger) -> None:
    """Crash/reboot 后 partial rollout 保留，但不消耗 genuine quota。"""
    rows = ledger.db.execute(
        "SELECT attempt_id,artifact_path FROM attempts WHERE status='in_progress'"
    ).fetchall()
    for attempt_id, relative_path in rows:
        record = _read_attempt(ledger, str(relative_path))
        record.update({"interruption": "process_restart", "failure_class": "infrastructure"})
        ledger.mark_infrastructure_interrupted(str(attempt_id), record)


def _allocate_reserve(
    state: dict[str, Any], slot_id: str, inputs: TaskInputs, logger: CollectorLog,
) -> bool:
    slot = state["slots"][slot_id]
    key = str(slot["stratum"])
    allocated = state["reserve_allocations"]
    candidate = next(
        (task_id for task_id in inputs.reserve_queues.get(key, []) if task_id not in allocated),
        None,
    )
    if candidate is None:
        slot["exhausted"] = True
        if key not in state["reserve_exhausted_strata"]:
            state["reserve_exhausted_strata"].append(key)
        logger.event("RESERVE_EXHAUSTED", pass_name="A", task_id=slot["current_task_id"])
        return False
    allocated[candidate] = int(slot_id)
    slot["current_task_id"] = candidate
    slot["reserve_chain"].append(candidate)
    state["tasks"][candidate] = {
        "coverage_slot": int(slot_id), "source": "reserve", "stratum": key,
        "first_attempts": 0, "first_success": False, "accepted_count": 0,
        "post_attempts": 0, "b_attempts": 0, "b_done": False,
    }
    logger.event("RESERVE_ALLOCATE", pass_name="A", task_id=candidate,
                 coverage_slot=slot_id)
    return True


def _pass_a_work(
    state: dict[str, Any], inputs: TaskInputs, logger: CollectorLog, *, workers: int,
    first_cap: int,
) -> list[WorkItem]:
    work: list[WorkItem] = []
    for slot_id in sorted(state["slots"], key=int):
        slot = state["slots"][slot_id]
        if (slot["fulfilled"] or slot["exhausted"]
                or slot["current_task_id"] in state.get("active_task_ids", [])):
            continue
        while True:
            task_id = str(slot["current_task_id"])
            task = state["tasks"][task_id]
            if task["first_attempts"] < first_cap:
                work.append(WorkItem(
                    "A", task_id, int(slot_id), str(task["source"]),
                    int(task["first_attempts"]) + 1,
                ))
                break
            if not _allocate_reserve(state, slot_id, inputs, logger):
                break
        if len(work) >= workers:
            break
    return work


def _pass_b_work(state: dict[str, Any], *, workers: int) -> list[WorkItem]:
    rows = []
    for task_id, task in state["tasks"].items():
        if (task_id not in state.get("active_task_ids", [])
                and task["first_success"] and not task["b_done"] and task["b_attempts"] < 2):
            rows.append(WorkItem(
                "B", task_id, int(task["coverage_slot"]), str(task["source"]),
                int(task["b_attempts"]) + 1,
            ))
    rows.sort(key=lambda item: (item.coverage_slot, item.task_id))
    return rows[:workers]


def largest_remainder_targets(primary: Sequence[Mapping[str, Any]], target: int) -> dict[str, int]:
    counts = Counter(stratum_key(item) for item in primary)
    total = sum(counts.values())
    raw = {key: target * count / total for key, count in counts.items()}
    result = {key: int(value) for key, value in raw.items()}
    remaining = target - sum(result.values())
    for key in sorted(counts, key=lambda item: (-(raw[item] - int(raw[item])), item))[:remaining]:
        result[key] += 1
    return result


def _pass_c_work(
    state: dict[str, Any], inputs: TaskInputs, *, workers: int, remaining_quota: int,
) -> list[WorkItem]:
    targets = largest_remainder_targets(inputs.primary, int(state["target"]))
    accepted_by_stratum = Counter()
    for task in state["tasks"].values():
        accepted_by_stratum[str(task["stratum"])] += int(task["accepted_count"])
    eligible = []
    for task_id, task in state["tasks"].items():
        if (task_id not in state.get("active_task_ids", [])
                and task["first_success"] and task["accepted_count"] < 3
                and task["post_attempts"] < 3):
            deficit = targets.get(str(task["stratum"]), 0) - accepted_by_stratum[str(task["stratum"])]
            eligible.append((int(task["accepted_count"]), -deficit,
                             int(task["coverage_slot"]), task_id, task))
    eligible.sort(key=lambda row: row[:4])
    limit = min(workers, max(0, remaining_quota))
    return [WorkItem(
        "C", task_id, int(task["coverage_slot"]), str(task["source"]),
        int(task["post_attempts"]) + 1,
    ) for _count, _deficit, _slot, task_id, task in eligible[:limit]]


def next_work(
    state: dict[str, Any], inputs: TaskInputs, logger: CollectorLog, *, workers: int,
    first_cap: int = 2, stop_after_pass: str | None = None,
) -> list[WorkItem]:
    if workers <= 0:
        return []
    while True:
        active_count = len(state.get("active_task_ids", []))
        accepted = sum(int(task["accepted_count"]) for task in state["tasks"].values())
        if accepted >= int(state["target"]):
            state["status"] = "complete"
            return []
        current = state["current_pass"]
        if current == "A":
            work = _pass_a_work(state, inputs, logger, workers=workers, first_cap=first_cap)
            if work:
                return work
            if active_count:
                return []
            if stop_after_pass == "A":
                state["status"] = "complete" if accepted >= int(state["target"]) else "quota_unmet"
                return []
            state["current_pass"] = "B"
            logger.event("PASS_START", pass_name="B")
            continue
        if current == "B":
            work = _pass_b_work(state, workers=workers)
            if work:
                return work
            if active_count:
                return []
            state["current_pass"] = "C"
            logger.event("PASS_START", pass_name="C")
            continue
        # Every in-flight attempt conservatively reserves one possible success.
        remaining = int(state["target"]) - accepted - active_count
        work = _pass_c_work(state, inputs, workers=workers, remaining_quota=remaining)
        if work:
            return work
        if active_count:
            return []
        state["status"] = "quota_unmet"
        return []


def _safe_error_class(exc: BaseException) -> str:
    if isinstance(exc, TeacherClientError):
        return exc.kind
    if isinstance(exc, TeacherEnvError):
        return exc.kind
    return type(exc).__name__


def execute_rollout(
    work: WorkItem, stop: GlobalStop, *, ledger: TeacherLedger,
    scenario: str, expected_model: str, max_action_steps: int,
    logger: CollectorLog, client_factory: Callable[[Callable[..., None]], Any],
    env_factory: Callable[[], Any], sanitized_teacher_config: Mapping[str, Any],
) -> AttemptResult:
    """执行并持久化一个 fresh rollout；不在 worker 内做 duplicate/accept。"""
    attempt_id = ledger.start_attempt(work.task_id, teacher_attempt=work.attempt_ordinal)
    worker_name = threading.current_thread().name
    formal_work = {
        "pass": work.pass_name, "task_id": work.task_id,
        "coverage_slot": work.coverage_slot, "source": work.source,
        "attempt_ordinal": work.attempt_ordinal,
    }
    record: dict[str, Any] = {
        "run_id": ledger.manifest["run_id"], "scenario": scenario,
        "task_id": work.task_id, "formal_work": formal_work,
        "teacher_config": dict(sanitized_teacher_config), "started_at": utc_now(),
        "messages": [], "visible_responses": [], "actions": [],
        "policy_observations": [], "raw_observations": [], "api_diagnostics": [],
        "environment_diagnostics": [], "total_input_tokens": 0,
        "total_output_tokens": 0, "api_latency_total_s": 0.0,
        "environment_latency_total_s": 0.0, "infrastructure_retries": 0,
        "accepted": False, "rejection_reason": None,
        "evaluator_only": {"task_id": work.task_id},
    }
    ledger.save_progress(attempt_id, record)
    logger.event("ATTEMPT_START", worker=worker_name, pass_name=work.pass_name,
                 task_id=work.task_id, attempt=work.attempt_ordinal)
    env = None
    client = None
    session_id = None
    started = time.monotonic()
    retry_events: list[dict[str, Any]] = []

    def on_retry(kind: str, ordinal: int, maximum: int, delay: float) -> None:
        event = {"reason": kind, "retry_ordinal": ordinal, "max_retries": maximum,
                 "backoff_s": delay, "timestamp": utc_now()}
        retry_events.append(event)
        logger.event("API_RETRY", worker=worker_name, pass_name=work.pass_name,
                     task_id=work.task_id, attempt=work.attempt_ordinal,
                     classification=kind, retry_ordinal=ordinal)

    def finish(status: str, *, reason: str | None = None) -> AttemptResult:
        record.update({
            "termination_reason": reason or status, "finished_at": utc_now(),
            "trajectory_attempt_wall_time_s": time.monotonic() - started,
            "retry_events": retry_events,
        })
        ledger.save_attempt(attempt_id, record, status=status)
        return AttemptResult(attempt_id, status)

    try:
        stop.check_before_external_call()
        env = env_factory()
        reset = env.reset(scenario, work.task_id)
        session_id = reset.payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise TeacherEnvError("reset missing session identity", kind="infrastructure")
        context = policy_context_from_reset(reset.payload, scenario)
        observation = _policy_observation(reset.payload)
        messages = build_initial_messages(context, observation)
        assert_no_evaluator_leakage(messages)
        record["messages"] = list(messages)
        record["policy_observations"] = [observation]
        record["raw_observations"] = [reset.payload.get("raw_observation", "")]
        record["environment_latency_total_s"] += float(reset.latency_s)
        ledger.save_progress(attempt_id, record)
        client = client_factory(on_retry)
        for step in range(max_action_steps):
            stop.check_before_external_call()
            response: TeacherResponse = client.generate(messages)
            if response.model and response.model != expected_model:
                record["returned_model"] = response.model
                result = finish("provider_model_mismatch")
                raise FormalRunStop("returned model mismatch")
            record["visible_responses"].append(response.text)
            record.setdefault("action_traces", []).append(trace_visible_action(response.text))
            diagnostics = {
                "step": step + 1, "latency_s": response.latency_s,
                "retries": response.retries, "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "request_id": response.request_id or "N/A", "model": response.model or "N/A",
                "http_status": response.status_code, "retry_events": list(response.retry_events),
            }
            record["api_diagnostics"].append(diagnostics)
            record["infrastructure_retries"] += int(response.retries)
            record["total_input_tokens"] += int(response.input_tokens or 0)
            record["total_output_tokens"] += int(response.output_tokens or 0)
            record["api_latency_total_s"] += float(response.latency_s)
            for item in response.retry_events:
                if dict(item) not in retry_events:
                    retry_events.append(dict(item))
            if not _valid_visible_action(response.text):
                return finish("malformed_action")
            result = env.step(
                session_id, response.text, expected_task_id=work.task_id,
                expected_scenario=scenario,
            )
            action = str(result.payload.get("action", ""))
            record["actions"].append(action)
            next_observation = _policy_observation(result.payload)
            record["policy_observations"].append(next_observation)
            record["raw_observations"].append(result.payload.get("raw_observation", ""))
            record["environment_latency_total_s"] += float(result.latency_s)
            record["environment_diagnostics"].append({
                "step": step + 1, "latency_s": result.latency_s,
                "action_valid": result.payload.get("action_valid"),
                "internal_slot": result.payload.get("internal_slot"),
            })
            messages = append_turn(messages, response.text, next_observation)
            record["messages"] = list(messages)
            ledger.save_progress(attempt_id, record)
            if result.payload.get("action_valid") is False:
                return finish("invalid_action")
            if has_deterministic_no_progress_loop(
                record["actions"], record["policy_observations"]
            ):
                record["rejection_reason"] = "deterministic_no_progress_loop"
                return finish("hygiene_reject")
            if result.payload.get("done") or result.payload.get("over"):
                metrics = _metrics(result.payload)
                purchase = result.payload.get("purchase", {})
                record.update({
                    "done": True, "reward_metrics": metrics,
                    "terminal_purchase": purchase,
                    "evaluator_only": {
                        "task_id": work.task_id, "goal": result.payload.get("goal", {}),
                        "query_match": (
                            result.payload.get("reward_detail", {}).get("query_match")
                            if isinstance(result.payload.get("reward_detail"), Mapping) else None
                        ),
                    },
                })
                if metrics["r_succ"] == 1.0:
                    logger.event("ATTEMPT_SUCCESS", worker=worker_name,
                                 pass_name=work.pass_name, task_id=work.task_id,
                                 attempt=work.attempt_ordinal)
                    return finish("candidate_success")
                return finish("terminal_unsuccessful")
        return finish("max_steps")
    except WorkerCancelled:
        finish("infrastructure_interrupted", reason="global_stop")
        raise
    except TeacherClientError as exc:
        record["failure_class"] = exc.kind
        record["infrastructure_retries"] += int(exc.retries)
        retry_events.extend(dict(item) for item in exc.retry_events if dict(item) not in retry_events)
        if exc.kind in {"infrastructure", "provider_protocol_error"}:
            finish("infrastructure_interrupted", reason=exc.kind)
        elif exc.kind == "teacher_empty_response":
            return finish("teacher_empty_response")
        else:
            finish("provider_config_error", reason=exc.kind)
        raise FormalRunStop(exc.kind) from exc
    except TeacherEnvError as exc:
        record["failure_class"] = exc.kind
        if exc.kind == "invalid_action":
            return finish("invalid_action")
        finish("environment_fatal_error", reason=exc.kind)
        raise FormalRunStop(exc.kind) from exc
    except FormalRunStop:
        raise
    except (ValueError, KeyError) as exc:
        record["failure_class"] = type(exc).__name__
        finish("environment_fatal_error", reason=type(exc).__name__)
        raise FormalRunStop(type(exc).__name__) from exc
    finally:
        if session_id is not None and env is not None:
            try:
                env.release(session_id)
            except Exception:
                record["release_error"] = True
        if client is not None:
            client.close()
        if env is not None:
            env.close()


def summary_from_state(ledger: TeacherLedger, state: Mapping[str, Any]) -> dict[str, Any]:
    accepted = ledger.accepted_count()
    retries = 0
    api_calls = 0
    input_tokens = 0
    output_tokens = 0
    http_429 = 0
    http_5xx = 0
    for _attempt_id, relative_path in ledger.db.execute(
        "SELECT attempt_id,artifact_path FROM attempts"
    ).fetchall():
        try:
            record = _read_attempt(ledger, str(relative_path))
        except (OSError, ValueError):
            continue
        api_calls += len(record.get("api_diagnostics", []))
        retries += int(record.get("infrastructure_retries", 0) or 0)
        input_tokens += int(record.get("total_input_tokens", 0) or 0)
        output_tokens += int(record.get("total_output_tokens", 0) or 0)
        for event in record.get("retry_events", []):
            code = event.get("status_code")
            http_429 += int(code == 429)
            http_5xx += int(isinstance(code, int) and 500 <= code <= 599)
    unique = sum(1 for task in state["tasks"].values() if task["first_success"])
    reserve_used = len(state["reserve_allocations"])
    return {
        "run_id": ledger.manifest["run_id"], "status": state["status"],
        "scenario": ledger.manifest["scenario"], "policy_version": ledger.manifest["policy_version"],
        "workers": state.get("runtime_workers", ledger.manifest["workers"]), "current_pass": state["current_pass"],
        "coverage_slots_completed": sum(
            1 for slot in state["slots"].values() if slot["fulfilled"] or slot["exhausted"]
        ),
        "coverage_slots_total": len(state["slots"]), "unique_successful_tasks": unique,
        "accepted_trajectories": accepted, "genuine_attempts": state["genuine_attempts"],
        "reserve_used": reserve_used, "exact_duplicate_rejects": state["exact_duplicate_rejects"],
        "api_calls": api_calls, "retries": retries, "http_429": http_429,
        "http_5xx": http_5xx, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "started_at": state["started_at"], "last_success": state["last_success"],
        "last_error": state["last_error"], "last_update": state["last_update"],
    }


def run_engine(
    *, ledger: TeacherLedger, state: dict[str, Any], inputs: TaskInputs,
    policy: Mapping[str, Any], logger: CollectorLog, workers: int,
    client_factory: Callable[[Callable[..., None]], Any], env_factory: Callable[[], Any],
    stop: GlobalStop | None = None, first_cap: int = 2,
    stop_after_pass: str | None = None, api_style: str | None = None,
    api_profiles: Mapping[str, tuple[int, Callable[..., Any], str]] | None = None,
) -> dict[str, Any]:
    """运行 A/B/C；测试与 paid smoke 均调用这一入口。"""
    stop = stop or GlobalStop()
    if not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    profiles = dict(api_profiles) if api_profiles is not None else {
        "legacy": (workers, client_factory, api_style or policy["teacher"]["api_style"])
    }
    if (not profiles or any(capacity < 1 for capacity, _, _ in profiles.values())
            or sum(capacity for capacity, _, _ in profiles.values()) != workers):
        raise ValueError("API profile capacities must be positive and sum to workers")
    profile_active = dict.fromkeys(profiles, 0)
    state["runtime_workers"] = workers
    mark_stale_in_progress(ledger)
    state["active_task_ids"] = []
    state["in_flight_work"] = []
    reconcile_attempts(ledger, state, logger)
    max_steps = int(policy["protocol"]["max_action_steps"])
    sanitized = {
        "model": policy["teacher"]["model"],
        "api_style": api_style or policy["teacher"]["api_style"],
        "reasoning_effort": policy["teacher"]["reasoning_effort"],
        "request_semantics": policy["teacher"]["request_semantics"],
    }
    logger.event("PASS_START", pass_name=state["current_pass"])
    def invoke(item: WorkItem, profile_id: str) -> AttemptResult:
        try:
            stop.check_before_external_call()
            return execute_rollout(
                item, stop, ledger=ledger, scenario=str(ledger.manifest["scenario"]),
                expected_model=str(policy["teacher"]["model"]), max_action_steps=max_steps,
                logger=logger, client_factory=profiles[profile_id][1], env_factory=env_factory,
                sanitized_teacher_config={**sanitized, "api_style": profiles[profile_id][2]},
            )
        except BaseException:
            stop.set()  # Signal siblings immediately, not when main consumes this future.
            raise

    in_flight: dict[Future[AttemptResult], WorkItem] = {}
    future_profiles: dict[Future[AttemptResult], str] = {}

    def persist_reservations() -> None:
        state["active_task_ids"] = [item.task_id for item in in_flight.values()]
        state["in_flight_work"] = [dict(item.__dict__) for item in in_flight.values()]
        dump_state(ledger, state)

    # One executor survives refills (and drained Pass boundaries). No static backlog.
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="teacher-worker") as pool:
            try:
                while in_flight or (not stop.is_set() and state["status"] == "in_progress"):
                    while (not stop.is_set() and state["status"] == "in_progress"
                           and len(in_flight) < workers):
                        profile_id = next(
                            key for key, (capacity, _, _) in profiles.items()
                            if profile_active[key] < capacity
                        )
                        work = next_work(
                            state, inputs, logger, workers=1, first_cap=first_cap,
                            stop_after_pass=stop_after_pass,
                        )
                        if not work:
                            break
                        item = work[0]
                        if item.task_id in state["active_task_ids"]:
                            raise RuntimeError("scheduler invariant: same task dispatched concurrently")
                        # Reservation survives a crash even before the worker starts its ledger attempt.
                        state["active_task_ids"].append(item.task_id)
                        state["in_flight_work"].append(dict(item.__dict__))
                        dump_state(ledger, state)
                        if stop.is_set():
                            persist_reservations()
                            break
                        if profile_id != "legacy":
                            logger.task_api_profiles[item.task_id] = profile_id
                        future = pool.submit(invoke, item, profile_id)
                        in_flight[future] = item
                        future_profiles[future] = profile_id
                        profile_active[profile_id] += 1
                    if not in_flight:
                        break
                    completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    # Stable order only among already-completed futures; never wait on a sibling.
                    for future in sorted(completed, key=lambda f: (
                        in_flight[f].coverage_slot, in_flight[f].task_id
                    )):
                        in_flight.pop(future)
                        profile_active[future_profiles.pop(future)] -= 1
                        try:
                            future.result()
                        except WorkerCancelled:
                            stop.set()
                        except BaseException as exc:
                            stop.set()
                            state["last_error"] = _safe_error_class(exc)
                            logger.event("INFRA_STOP", classification=_safe_error_class(exc))
                    persist_reservations()
                    reconcile_attempts(ledger, state, logger)
                    for task_id in list(logger.task_api_profiles):
                        if task_id not in state["active_task_ids"]:
                            del logger.task_api_profiles[task_id]
                    accepted = ledger.accepted_count()
                    last_progress = int(state.get("last_progress_accepted", 0))
                    if accepted >= last_progress + 25:
                        logger.event("PROGRESS", pass_name=state["current_pass"],
                                     accepted=accepted, in_flight=len(in_flight))
                        state["last_progress_accepted"] = accepted - (accepted % 25)
                    dump_state(ledger, state)
            except BaseException:
                stop.set()  # Join/persist in-flight work before the caller can close SQLite.
                raise
    except BaseException:
        state["active_task_ids"] = []
        state["in_flight_work"] = []
        reconcile_attempts(ledger, state, logger)
        state["status"] = "stopped"
        dump_state(ledger, state)
        raise
    if stop.is_set() and state["status"] == "in_progress":
        state["status"] = "stopped"
        dump_state(ledger, state)
    if state["status"] == "complete":
        logger.event("COMPLETE", pass_name=state["current_pass"], accepted=ledger.accepted_count())
    elif state["status"] == "quota_unmet":
        logger.event("QUOTA_UNMET", pass_name=state["current_pass"], accepted=ledger.accepted_count())
    return summary_from_state(ledger, state)


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def new_run_id(*, smoke: bool = False) -> str:
    prefix = "formal-smoke" if smoke else "formal"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def find_resume_run(
    data_root: Path, scenario: str, expected: Mapping[str, Any], *, run_id: str | None = None,
) -> str:
    scenario_root = data_root / scenario
    compatible: list[str] = []
    incompatible: list[str] = []
    for directory in sorted(scenario_root.glob("*")) if scenario_root.exists() else []:
        manifest_path = directory / "run_manifest.json"
        database = directory / "state.sqlite"
        if not manifest_path.exists() or not database.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("purpose") != "formal_teacher_collection":
            continue
        if run_id is not None and directory.name != run_id:
            continue
        import sqlite3
        db = sqlite3.connect(database)
        row = db.execute("SELECT value FROM run_state WHERE key='status'").fetchone()
        db.close()
        if row and row[0] in TERMINAL_RUN_STATES:
            continue
        if all(manifest.get(key) == value for key, value in expected.items()):
            compatible.append(directory.name)
        else:
            incompatible.append(directory.name)
    if run_id is not None:
        if compatible:
            return compatible[0]
        raise ValueError(f"run_id 不存在、已 terminal 或 immutable config 不兼容: {run_id}")
    if len(compatible) == 1:
        return compatible[0]
    if len(compatible) > 1:
        raise ValueError("存在多个 compatible incomplete formal run；请用 --run-id: " + ", ".join(compatible))
    if incompatible:
        raise ValueError("存在 incomplete formal run，但 immutable config 不兼容: " + ", ".join(incompatible))
    raise FileNotFoundError("没有 compatible incomplete formal run；请去掉 --resume 创建新 run")


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    atomic_json(path, dict(summary))


def print_summary(summary: Mapping[str, Any], *, resume_command: str | None = None) -> None:
    for key in (
        "scenario", "current_pass", "unique_successful_tasks", "accepted_trajectories",
        "genuine_attempts", "reserve_used", "exact_duplicate_rejects", "retries",
        "input_tokens", "output_tokens", "status",
        "elapsed_seconds",
    ):
        print(f"{key}={summary.get(key)}")
    if resume_command and summary.get("status") == "stopped":
        print(f"Resume with: {resume_command}")


def parse_api_workers(value: str) -> dict[str, int]:
    capacities: dict[str, int] = {}
    for entry in value.split(","):
        parts = entry.strip().split(":")
        if (len(parts) != 2 or not all(p.isascii() and p.isdecimal() for p in parts)
                or int(parts[0]) < 1 or int(parts[1]) < 1):
            raise ValueError("--api-workers 格式为 1:8,2:12；profile 和 workers 必须为正整数")
        profile_id, count = str(int(parts[0])), int(parts[1])
        if profile_id in capacities:
            raise ValueError("--api-workers 含重复 profile ID")
        capacities[profile_id] = count
    if sum(capacities.values()) > MAX_WORKERS:
        raise ValueError(f"total workers must be <= {MAX_WORKERS}")
    return capacities


def run_from_configuration(
    *, project_root: Path, scenario: str, policy_path: Path, manifest_dir: Path,
    data_root: Path, env_file: Path, endpoint: str, workers: int,
    resume: bool = False, run_id: str | None = None,
    selected_primary_ids: Sequence[str] | None = None, smoke: bool = False,
    api_workers: str | None = None,
) -> dict[str, Any]:
    """完成 no-paid preflight 后构造/恢复 run 并调用真正 formal engine。"""
    from rollout.teacher_client import load_env_file

    policy = load_collection_policy(policy_path)
    if policy["identity"]["policy_version"] != "p3b-v1.1":
        raise ValueError("P3c 默认只接受 p3b-v1.1")
    if scenario not in policy["scope"]["scenarios"]:
        raise ValueError(f"policy 不支持 scenario: {scenario}")
    capacities = parse_api_workers(api_workers) if api_workers is not None else {"legacy": workers}
    workers = sum(capacities.values())
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    if not env_file.is_file():
        raise FileNotFoundError(f"缺少 teacher env file: {env_file}")
    cfg = load_env_file(str(env_file))
    required = ("TEACHER_API_MODEL", "TEACHER_REASONING_EFFORT")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(".env.teacher 缺少: " + ", ".join(missing))
    expected_cfg = {
        "TEACHER_API_MODEL": policy["teacher"]["model"],
        "TEACHER_REASONING_EFFORT": policy["teacher"]["reasoning_effort"],
    }
    mismatched = [key for key, value in expected_cfg.items() if cfg.get(key) != value]
    if mismatched:
        raise ValueError("teacher config 与 p3b-v1.1 不一致: " + ", ".join(mismatched))
    # Credentials stay inside client factory closures; never enter run/trajectory metadata.
    transports = {}
    for profile_id in capacities:
        suffix = "" if profile_id == "legacy" else f"_{profile_id}"
        url, key = cfg.get(f"TEACHER_API_URL{suffix}"), cfg.get(f"TEACHER_API_KEY{suffix}")
        style = cfg.get(f"TEACHER_API_STYLE{suffix}", cfg.get("TEACHER_API_STYLE", ""))
        if not url or not key:
            raise ValueError(f"缺少 TEACHER_API_URL{suffix} / TEACHER_API_KEY{suffix}")
        if style not in {"chat_completions", "responses"}:
            raise ValueError(f"TEACHER_API_STYLE{suffix} 必须是 chat_completions 或 responses")
        transports[profile_id] = (url, key, style)
    first_style = next(iter(transports.values()))[2]

    inputs = load_task_inputs(
        manifest_dir, policy, scenario, selected_primary_ids=selected_primary_ids,
        allow_non_primary_selected=smoke,
    )
    if smoke:
        # The paid smoke is exactly two known-short TRAIN tasks. It exercises
        # the formal engine without silently expanding into reserve work.
        inputs = TaskInputs(
            primary=inputs.primary, train=inputs.train, reserve_queues={}, by_id=inputs.by_id,
            primary_manifest_hash=inputs.primary_manifest_hash,
            train_manifest_hash=inputs.train_manifest_hash,
        )
    # Remote protocol/prompt parity preflight. No teacher request is made.
    env = TeacherEnvClient(
        endpoint, timeout=60.0,
        expected_environment_version=policy["protocol"]["environment_version"],
    )
    try:
        health = env.health().payload
        if health.get("status") != "ok":
            raise ValueError("ShopSimulator health check failed")
        if health.get("environment_version") != policy["protocol"]["environment_version"]:
            raise ValueError("environment version 与 policy 不一致")
        probe = env.reset(scenario, str(inputs.primary[0]["task_id"]))
        try:
            context = policy_context_from_reset(probe.payload, scenario)
            _policy_observation(probe.payload)
            if probe.payload.get("policy_observation_version") != policy["protocol"]["policy_observation_version"]:
                raise ValueError("policy observation version 与 policy 不一致")
            if probe.payload.get("profiler_protocol_version") != policy["protocol"]["action_protocol_version"]:
                raise ValueError("action protocol version 与 policy 不一致")
        finally:
            env.release(str(probe.payload["session_id"]))
    finally:
        env.close()

    data_root.mkdir(parents=True, exist_ok=True)
    write_probe = data_root / f".write-test-{uuid.uuid4().hex}"
    try:
        write_probe.write_text("ok", encoding="utf-8")
    finally:
        write_probe.unlink(missing_ok=True)
    sanitized_request = {
        "model": policy["teacher"]["model"],
        "reasoning_effort": policy["teacher"]["reasoning_effort"],
        "request_semantics": policy["teacher"]["request_semantics"],
        "omitted": list(policy["teacher"]["explicitly_omitted_request_fields"]),
    }
    selected_hash = task_ids_hash(inputs.primary)
    purpose = "formal_teacher_smoke" if smoke else "formal_teacher_collection"
    expected_resume = {
        "purpose": purpose, "scenario": scenario,
        "policy_version": policy["identity"]["policy_version"],
        "policy_hash": policy["identity"]["policy_hash"],
        "teacher_model": policy["teacher"]["model"],
        "reasoning_effort": policy["teacher"]["reasoning_effort"],
        "seed": policy["identity"]["seed"],
        "primary_manifest_hash": inputs.primary_manifest_hash,
        "train_manifest_hash": inputs.train_manifest_hash,
        "system_prompt_hash": context.source_hash,
        "selected_primary_hash": selected_hash,
    }
    if resume:
        if smoke:
            raise ValueError("formal smoke 不支持 resume；重新运行会创建独立 smoke run")
        resolved_run_id = find_resume_run(
            data_root, scenario, expected_resume, run_id=run_id
        )
    else:
        if run_id is not None:
            raise ValueError("--run-id 只能与 --resume 一起使用")
        resolved_run_id = new_run_id(smoke=smoke)
    persisted_transport = None
    if resume:
        persisted_transport = json.loads(
            (data_root / scenario / resolved_run_id / "run_manifest.json").read_text(encoding="utf-8")
        )
    manifest = {
        **expected_resume, "run_id": resolved_run_id, "workers": workers,
        # Keep historical manifest provenance untouched while allowing the
        # current relay transport to change between resume sessions.
        "api_style": (
            persisted_transport.get("api_style", first_style)
            if persisted_transport else first_style
        ),
        "sanitized_request_config_hash": (
            persisted_transport.get("sanitized_request_config_hash", canonical_hash(sanitized_request))
            if persisted_transport else canonical_hash(sanitized_request)
        ),
        "collection_config_hash": str(policy["identity"]["policy_hash"]),
        "shopsim_source_fingerprint": health.get("source_fingerprint", "unknown"),
        "environment_fingerprint": health.get("source_fingerprint", "unknown"),
        "environment_version": policy["protocol"]["environment_version"],
        "policy_observation_version": policy["protocol"]["policy_observation_version"],
        "profiler_protocol_version": policy["protocol"]["action_protocol_version"],
        "reward_deviation_version": policy["protocol"]["reward_deviation_version"],
        "max_action_steps": policy["protocol"]["max_action_steps"],
        "primary_manifest": policy["task_sources"][scenario]["primary_manifest"],
        "train_manifest": policy["task_sources"][scenario]["train_manifest"],
        "reserve_strategy": "deterministic_unused_same_stratum",
        "git_commit": git_commit(project_root), "created_at": utc_now(),
        "smoke_bounds": ({"tasks": len(inputs.primary), "first_attempts_max": 1,
                           "stop_after_pass": "A"} if smoke else None),
    }
    ledger = TeacherLedger(
        data_root, manifest, resume=resume,
        extra_immutable_fields=FORMAL_EXTRA_IMMUTABLE,
    )
    log_path = data_root / "collector.log"
    logger = CollectorLog(log_path, scenario=scenario, run_id=resolved_run_id, stream=sys.stdout)
    capacity_arg = (
        "--api-workers " + ",".join(f"{key}:{count}" for key, count in capacities.items())
        if api_workers is not None else f"--workers {workers}"
    )
    resume_command = (
        f"python3 scripts/collect_teacher.py --scenario {scenario} "
        f"{capacity_arg} --run-id {resolved_run_id} --resume"
    )
    stop = GlobalStop()
    old_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(_signum: int, _frame: Any) -> None:
        stop.set()
        logger.event("CTRL_C")

    signal.signal(signal.SIGINT, handle_sigint)
    try:
        state = load_state(ledger)
        if state is None:
            target = len(inputs.primary) if smoke else int(
                policy["targets"]["accepted_trajectories_per_scenario"]["target"]
            )
            state = initial_state(inputs, target=target)
            dump_state(ledger, state)
            logger.event("RUN_START", workers=workers, policy=policy["identity"]["policy_version"])
        else:
            if state["status"] not in TERMINAL_RUN_STATES:
                state["status"] = "in_progress"
            logger.event("RESUME", workers=workers, policy=policy["identity"]["policy_version"])

        state["runtime_workers"] = workers
        history = state.setdefault("scheduler_history", [])
        history.append({
            "scheduler_version": SCHEDULER_VERSION, "git_commit": git_commit(project_root),
            "started_at": utc_now(), "run_id": resolved_run_id,
            "policy_version": policy["identity"]["policy_version"],
            "previous_scheduler": history[-1]["scheduler_version"] if history else ("fixed-micro-batch" if resume else None),
            "previous_manifest_commit": ledger.manifest.get("git_commit"),
            "attempt_rowid_boundary": ledger.db.execute(
                "SELECT COALESCE(MAX(rowid), 0) FROM attempts"
            ).fetchone()[0],
            "accepted_before_resume": ledger.accepted_count(),
            "genuine_attempts_before_resume": state["genuine_attempts"],
            "current_pass": state["current_pass"], "workers": workers,
            "api_workers": capacities,
        })
        dump_state(ledger, state)
        logger.event("SCHEDULER_START", scheduler=SCHEDULER_VERSION,
                     commit=history[-1]["git_commit"],
                     attempt_rowid_boundary=history[-1]["attempt_rowid_boundary"])

        def make_client_factory(transport: tuple[str, str, str]) -> Callable[..., TeacherClient]:
            url, key, style = transport

            def factory(on_retry: Callable[..., None]) -> TeacherClient:
                return TeacherClient(
                    api_url=url, api_key=key, model=cfg["TEACHER_API_MODEL"], api_style=style,
                    reasoning_effort=cfg["TEACHER_REASONING_EFFORT"], on_retry=on_retry,
                )
            return factory

        profiles = {
            profile_id: (count, make_client_factory(transports[profile_id]), transports[profile_id][2])
            for profile_id, count in capacities.items()
        }
        client_factory = next(iter(profiles.values()))[1]

        def env_factory() -> TeacherEnvClient:
            return TeacherEnvClient(
                endpoint, timeout=60.0,
                expected_environment_version=policy["protocol"]["environment_version"],
            )

        summary = run_engine(
            ledger=ledger, state=state, inputs=inputs, policy=policy, logger=logger,
            workers=workers, client_factory=client_factory, env_factory=env_factory,
            stop=stop, first_cap=1 if smoke else 2,
            stop_after_pass="A" if smoke else None,
            api_style=first_style, api_profiles=profiles,
        )
        summary["elapsed_seconds"] = max(
            0.0,
            (datetime.now(timezone.utc) - datetime.fromisoformat(state["started_at"])).total_seconds(),
        )
        summary["resume_command"] = resume_command if summary["status"] == "stopped" else None
        write_summary(ledger.paths.root / "summary.json", summary)
        print_summary(summary, resume_command=resume_command)
        return summary
    finally:
        signal.signal(signal.SIGINT, old_handler)
        ledger.close()

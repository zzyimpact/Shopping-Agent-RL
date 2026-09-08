"""Single-worker P3a teacher rollout profiler.

This module deliberately contains no collection-policy filtering and no
concurrency.  It orchestrates the already-pinned teacher API adapter and the
remote ShopSimulator HTTP protocol, while keeping every attempt auditable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from env.teacher_env_client import TeacherEnvClient, TeacherEnvError
from rollout.diversity import behavior_fingerprint, exact_duplicate, similarity_features
from rollout.prompt import (append_turn, assert_no_evaluator_leakage,
                            build_initial_messages, policy_context_from_reset)
from rollout.protocol import (
    POLICY_OBSERVATION_VERSION,
    PROFILER_PROTOCOL_VERSION,
    trace_visible_action,
)
from rollout.progress import ProgressLogger
from rollout.runtime import CollectionInfrastructureInterrupted, generate_or_interrupt
from rollout.storage import GracefulCollectionStop, TeacherLedger, canonical_hash
from rollout.teacher_client import TeacherClient, TeacherClientError, TeacherResponse


PROFILE_LIMITS = {
    "first_success_max_attempts": 3,
    "second_demo_max_attempts": 2,
    "max_action_steps": 30,
    "workers": 1,
}
EXTRA_IMMUTABLE = (
    "purpose", "selected_task_list_hash", "source_sft_manifest_hash", "upstream_prompt_hash",
    "max_action_steps", "profiling_limits", "query_match_deviation", "persona_pool_deviation",
    "environment_fingerprint", "reward_compatibility_version",
    "policy_observation_version", "profiler_protocol_version",
)


class ProfileStop(RuntimeError):
    """Run-level stop that preserves artifacts and does not count a teacher attempt."""


class TeacherModelMismatch(ProfileStop):
    pass


@dataclass(frozen=True)
class ResumeRun:
    run_id: str
    status: str
    created_at: str
    attempts: int
    touched_tasks: int
    terminal_tasks: int
    successes: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_ids_hash(task_ids: list[str], *, trailing_newline: bool = False) -> str:
    """Hash the frozen ID sequence using the manifest's line serialization.

    Early P3a task-list generation used ``printf '%s\\n'`` (a final newline),
    while the P2 helper hashes ``"\\n".join(ids)`` without one.  Both are
    deterministic representations of the already-frozen IDs; accept either so
    validation does not force a task-list rewrite or re-sampling.
    """
    serialized = "\n".join(task_ids)
    if trailing_newline:
        serialized += "\n"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _find_resume_run(
    data_root: Path,
    scenario: str,
    task_ids: list[str],
    expected_config: Mapping[str, Any] | None = None,
    *,
    purpose: str = "p3a_profiling",
) -> tuple[ResumeRun | None, int]:
    candidates: list[ResumeRun] = []
    incompatible: list[str] = []
    scenario_root = data_root / scenario
    if not scenario_root.exists():
        return None, 0
    for directory in sorted(p for p in scenario_root.iterdir() if p.is_dir()):
        manifest_path = directory / "run_manifest.json"
        state_path = directory / "state.sqlite"
        if not manifest_path.exists() or not state_path.exists():
            continue
        try:
            manifest = _read_json(manifest_path)
            import sqlite3
            db = sqlite3.connect(state_path)
            state = db.execute("SELECT value FROM run_state WHERE key='status'").fetchone()
            if (manifest.get("purpose") != purpose
                    or manifest.get("scenario") != scenario
                    or manifest.get("status") == "invalidated_by_implementation_bug"
                    or (state and state[0] in {"complete", "invalidated_by_implementation_bug"})):
                db.close()
                continue
            if list(manifest.get("selected_task_ids", task_ids)) != list(task_ids):
                db.close()
                incompatible.append(directory.name)
                continue
            if expected_config and any(manifest.get(key) != value for key, value in expected_config.items()):
                db.close()
                incompatible.append(directory.name)
                continue
            had_infrastructure_stop = db.execute(
                "SELECT 1 FROM attempts WHERE status='infrastructure_interrupted' LIMIT 1"
            ).fetchone()
            status_text = str(state[0]) if state else (
                "infrastructure_interrupted" if had_infrastructure_stop else "incomplete"
            )
            attempts = int(db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])
            touched = int(db.execute("SELECT COUNT(DISTINCT task_id) FROM attempts").fetchone()[0])
            terminal = int(db.execute(
                "SELECT COUNT(DISTINCT task_id) FROM attempts WHERE status='profile_unsolved'"
            ).fetchone()[0])
            successes = int(db.execute("SELECT COUNT(*) FROM attempts WHERE status='success'").fetchone()[0])
            db.close()
            candidates.append(ResumeRun(
                run_id=directory.name,
                status=status_text,
                created_at=str(manifest.get("created_at", directory.name)),
                attempts=attempts,
                touched_tasks=touched,
                terminal_tasks=terminal,
                successes=successes,
            ))
        except Exception:
            continue
    if not candidates and incompatible:
        raise ValueError(
            "存在未完成 profiling run，但都与当前 task/API 配置不兼容；请恢复原配置或新建 run。"
        )
    candidates.sort(key=lambda item: (item.created_at, item.run_id))
    return (candidates[-1] if candidates else None), max(0, len(candidates) - 1)


def _validate_resume_task_manifest(ledger: TeacherLedger, task_ids: list[str]) -> None:
    recorded = ledger.manifest.get("selected_task_ids")
    if recorded is not None and list(recorded) != list(task_ids):
        raise ValueError("拒绝 resume：selected task IDs 与 run manifest 不一致")


def _metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    detail = payload.get("reward_detail") if isinstance(payload.get("reward_detail"), Mapping) else {}
    category = float(detail.get("r_category", detail.get("r_type", 0.0)) or 0.0)
    attribute = float(detail.get("r_attribute", detail.get("r_att", 0.0)) or 0.0)
    option = float(detail.get("r_option", 0.0) or 0.0)
    price = float(detail.get("r_price", 0.0) or 0.0)
    loose = float(payload.get("reward", 0.0) or 0.0)
    strict = category * attribute * option * price
    finished = bool(payload.get("done") or payload.get("over"))
    return {
        "r_loose": loose, "r_strict": strict, "r_succ": float(strict == 1.0),
        "r_finish": float(finished), "r_category": category, "r_attribute": attribute,
        "r_option": option, "r_price": price,
    }


def _terminal_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics(payload)
    return {"metrics": metrics, "query_match":
            (payload.get("reward_detail", {}).get("query_match")
             if isinstance(payload.get("reward_detail"), Mapping) else None),
            "purchase": payload.get("purchase", {}), "goal": payload.get("goal", {})}


def _task_rows(ledger: TeacherLedger, task_id: str) -> list[tuple[Any, ...]]:
    return ledger.db.execute(
        "SELECT attempt_id, status, teacher_attempt, artifact_path FROM attempts "
        "WHERE task_id=? ORDER BY rowid", (task_id,)
    ).fetchall()


def _success_records(ledger: TeacherLedger, task_id: str) -> list[dict[str, Any]]:
    paths = []
    for attempt_id, status, _index, artifact_path in _task_rows(ledger, task_id):
        if status != "success":
            continue
        path = ledger.paths.root / artifact_path
        if path.exists():
            try:
                record = _read_json(path)
            except (OSError, ValueError):
                continue
            if record.get("profiling_status") == "success":
                paths.append(record)
    return paths


def _valid_visible_action(text: str) -> bool:
    # The upstream extractor/parser is the source of truth.  This local gate
    # only prevents an obviously malformed response from consuming an env
    # step; it deliberately does not repair markdown, case, or whitespace.
    return bool(trace_visible_action(text)["canonical"])


def _policy_observation(payload: Mapping[str, Any]) -> str:
    """Read the explicit model-visible observation contract from the service."""

    for key in ("policy_observation", "user_message"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError(
        "remote response missing canonical policy observation "
        "(expected policy_observation or user_message)"
    )


def _manifest_for(run_id: str, scenario: str, task_data: Mapping[str, Any], env_health: Mapping[str, Any],
                  context_hash: str, env_endpoint: str, env_payload: Mapping[str, Any], cfg: Mapping[str, str]) -> dict[str, Any]:
    metadata = task_data.get("metadata", {})
    purpose = str(metadata.get("purpose", "p3a_profiling"))
    return {
        "purpose": purpose, "run_id": run_id, "scenario": scenario,
        "teacher_model": cfg["TEACHER_API_MODEL"], "api_style": cfg["TEACHER_API_STYLE"],
        "reasoning_effort": cfg["TEACHER_REASONING_EFFORT"], "system_prompt_hash": context_hash,
        "upstream_prompt_hash": context_hash, "selected_task_list_hash": metadata.get("task_ids_sha256"),
        "source_sft_manifest_hash": metadata.get("source_primary_manifest_sha256") or metadata.get("source_manifest_sha256"),
        "collection_config_hash": canonical_hash({"purpose": purpose, "scenario": scenario, "seed": metadata.get("seed"), "task_ids": [str(item["task_id"]) for item in task_data.get("tasks", [])], "limits": PROFILE_LIMITS}),
        "shopsim_source_fingerprint": env_health.get("source_fingerprint", "unknown"),
        "environment_fingerprint": env_health.get("source_fingerprint", "unknown"),
        "environment_version": env_payload.get("environment_version", "task-scoped-v3-multisession"),
        "policy_observation_version": env_payload.get("policy_observation_version", POLICY_OBSERVATION_VERSION),
        "profiler_protocol_version": env_payload.get("profiler_protocol_version", PROFILER_PROTOCOL_VERSION),
        "reward_deviation_version": env_payload.get("reward_deviation_version", "query-match-false-v1"),
        "reward_compatibility_version": "query-match-false-v1",
        "query_match_deviation": "missing query -> query_match=False",
        "persona_pool_deviation": "paper 3383 -> project actual upstream snapshot 3323",
        "profiling_limits": PROFILE_LIMITS,
        "max_action_steps": PROFILE_LIMITS["max_action_steps"],
        "selected_task_ids": [str(item["task_id"]) for item in task_data.get("tasks", [])],
        "prompt_source": env_payload.get("policy_context", {}).get("source"),
        "created_at": _now(),
    }


def _run_attempt(*, ledger: TeacherLedger, env: TeacherEnvClient, client: TeacherClient,
                 task_id: str, scenario: str, phase: str, attempt_index: int,
                 expected_model: str, logger: ProgressLogger, guard: GracefulCollectionStop,
                 resume_command: str | None = None) -> str:
    timings = {"reset": 0.0, "api": 0.0, "step": 0.0, "release": 0.0}
    started = time.monotonic()
    outcome = "interrupted"
    try:
        outcome = _execute_attempt(
            ledger=ledger, env=env, client=client, task_id=task_id, scenario=scenario,
            phase=phase, attempt_index=attempt_index, expected_model=expected_model,
            logger=logger, guard=guard, timings=timings,
            resume_command=resume_command,
        )
        return outcome
    finally:
        total = time.monotonic() - started
        accounted = sum(timings.values())
        logger.line(
            f"[TIMING] {task_id} phase={phase} attempt {attempt_index} | outcome={outcome} | "
            f"reset={timings['reset']:.2f}s | API={timings['api']:.2f}s | "
            f"env_steps={timings['step']:.2f}s | release={timings['release']:.2f}s | "
            f"local/storage={max(0.0, total - accounted):.2f}s | total={total:.2f}s"
        )


def _execute_attempt(*, ledger: TeacherLedger, env: TeacherEnvClient, client: TeacherClient,
                     task_id: str, scenario: str, phase: str, attempt_index: int,
                     expected_model: str, logger: ProgressLogger, guard: GracefulCollectionStop,
                     timings: dict[str, float], resume_command: str | None = None) -> str:
    attempt_id = ledger.start_attempt(task_id, teacher_attempt=attempt_index)
    session_id: str | None = None
    try:
        call_started = time.monotonic()
        try:
            reset = env.reset(scenario, task_id)
        finally:
            timings["reset"] += time.monotonic() - call_started
        session_id = reset.payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            ledger.save_attempt(attempt_id, {
                "task_id": task_id, "scenario": scenario, "attempt_index": attempt_index,
                "attempt_phase": phase, "termination_reason": "environment_failure",
                "success": False, "observations": [], "actions": [],
                "environment_error": "reset response missing session_id",
            }, status="environment_failure")
            return "environment_failure"
    except TeacherEnvError as exc:
        base = {"task_id": task_id, "scenario": scenario, "attempt_index": attempt_index,
                "attempt_phase": phase, "termination_reason": "environment_failure",
                "success": False, "observations": [], "actions": []}
        if getattr(exc, "kind", None) == "infrastructure":
            ledger.mark_infrastructure_interrupted(attempt_id, {**base, "failure_class": "environment_infrastructure"})
            ledger.set_state("status", "infrastructure_interrupted")
            raise ProfileStop("ShopSimulator infrastructure unavailable during reset") from exc
        ledger.save_attempt(attempt_id, base, status="environment_failure")
        return "environment_failure"
    context = policy_context_from_reset(reset.payload, scenario)
    initial_policy_observation = _policy_observation(reset.payload)
    messages = build_initial_messages(context, initial_policy_observation)
    assert_no_evaluator_leakage(messages)
    record: dict[str, Any] = {
        "task_id": task_id, "scenario": scenario, "attempt_index": attempt_index,
        "attempt_phase": phase, "started_at": _now(), "termination_reason": None,
        "success": False, "messages": list(messages), "visible_responses": [],
        "actions": [], "observations": [initial_policy_observation],
        "raw_observations": [reset.payload.get("raw_observation", reset.payload.get("observation", ""))],
        "api_diagnostics": [], "environment_diagnostics": [], "infrastructure_retries": 0,
        "malformed_action_count": 0, "invalid_action_count": 0, "action_steps": 0,
        "total_input_tokens": 0, "total_output_tokens": 0, "api_latency_total_s": 0.0,
        "environment_latency_total_s": reset.latency_s,
        "evaluator_only": {"task_id": task_id, "reset_payload": {
            "instruction": reset.payload.get("instruction"), "scenario": scenario,
        }},
    }
    guard.current_attempt_id, guard.current_record = attempt_id, record
    try:
        for step in range(PROFILE_LIMITS["max_action_steps"]):
            if guard.stop_requested:
                raise ProfileStop("SIGINT")
            partial = {**record, "action_steps": step}
            try:
                call_started = time.monotonic()
                try:
                    response: TeacherResponse = generate_or_interrupt(
                        client, messages, ledger=ledger, attempt_id=attempt_id, partial_record=partial,
                        resume_command=resume_command or f"python3 scripts/profile_teacher.py --scenario {scenario} --resume", logger=logger,
                    )
                finally:
                    timings["api"] += time.monotonic() - call_started
            except TeacherClientError as exc:
                if exc.kind == "teacher_empty_response":
                    record["termination_reason"] = "teacher_empty_response"
                    ledger.save_attempt(attempt_id, record, status="teacher_failure")
                    return "teacher_empty_response"
                # Request/config errors are not silently retried.  Persist the
                # redacted classification and stop before another paid call.
                record["termination_reason"] = exc.kind
                ledger.save_attempt(attempt_id, record, status="provider_config_error")
                raise ProfileStop(f"teacher request/config failure ({exc.kind})") from exc
            if response.model and response.model != expected_model:
                ledger.save_attempt(attempt_id, {**record, "termination_reason": "provider_model_mismatch"},
                                    status="provider_model_mismatch")
                raise TeacherModelMismatch(f"teacher returned model {response.model!r}, expected {expected_model!r}")
            record["visible_responses"].append(response.text)
            record.setdefault("action_traces", []).append(trace_visible_action(response.text))
            record["api_diagnostics"].append({
                "latency_s": response.latency_s, "retries": response.retries,
                "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                "request_id": response.request_id or "N/A", "model": response.model or "N/A",
                "http_status": response.status_code,
                "retry_events": list(response.retry_events),
            })
            record["infrastructure_retries"] += response.retries
            record["total_input_tokens"] += response.input_tokens or 0
            record["total_output_tokens"] += response.output_tokens or 0
            record["api_latency_total_s"] += response.latency_s
            if not _valid_visible_action(response.text):
                record["malformed_action_count"] += 1
                record["termination_reason"] = "malformed_action"
                ledger.save_attempt(attempt_id, record, status="malformed_action")
                return "malformed_action"
            try:
                call_started = time.monotonic()
                try:
                    result = env.step(session_id, response.text)
                finally:
                    timings["step"] += time.monotonic() - call_started
            except TeacherEnvError as exc:
                if getattr(exc, "kind", None) == "infrastructure":
                    ledger.mark_infrastructure_interrupted(attempt_id, {**record, "failure_class": "environment_infrastructure"})
                    ledger.set_state("status", "infrastructure_interrupted")
                    raise ProfileStop("ShopSimulator infrastructure unavailable") from exc
                if "malformed_action" in str(exc).lower() or "invalid_action" in str(exc).lower():
                    record["invalid_action_count"] += 1
                    record["termination_reason"] = "invalid_action"
                    ledger.save_attempt(attempt_id, record, status="invalid_action")
                    return "invalid_action"
                record["termination_reason"] = "environment_failure"
                record["environment_error"] = "remote environment request failed"
                ledger.save_attempt(attempt_id, record, status="environment_failure")
                return "environment_failure"
            record["actions"].append(result.payload.get("action", ""))
            observation = _policy_observation(result.payload)
            record["observations"].append(observation)
            record.setdefault("raw_observations", []).append(
                result.payload.get("raw_observation", result.payload.get("observation", ""))
            )
            record["environment_diagnostics"].append({
                "latency_s": result.latency_s,
                "action_valid": result.payload.get("action_valid", "N/A"),
                "pre_step_available_actions": result.payload.get("pre_step_available_actions"),
                "post_step_available_actions": result.payload.get(
                    "post_step_available_actions", result.payload.get("available_actions")
                ),
            })
            record["environment_latency_total_s"] += result.latency_s
            record["action_steps"] = step + 1
            if result.payload.get("action_valid") is False:
                record["invalid_action_count"] += 1
                record["termination_reason"] = "invalid_action"
                ledger.save_attempt(attempt_id, record, status="invalid_action")
                return "invalid_action"
            messages = append_turn(messages, response.text, observation)
            # Keep the artifact's conversation exactly aligned with the next
            # request; avoid duplicating the assistant turn in the audit copy.
            record["messages"] = list(messages)
            ledger.save_progress(attempt_id, record)
            if result.payload.get("done") or result.payload.get("over"):
                terminal = _terminal_payload(result.payload)
                record["reward_metrics"] = terminal["metrics"]
                record["query_match"] = terminal["query_match"]
                record["evaluator_only"].update({"purchase": terminal["purchase"], "goal": terminal["goal"]})
                record["success"] = terminal["metrics"]["r_succ"] == 1.0
                record["termination_reason"] = "success" if record["success"] else "terminal_unsuccessful"
                status = "success" if record["success"] else "teacher_failure"
                if record["success"]:
                    fp = behavior_fingerprint(record["actions"], final_purchase_asin=(terminal["purchase"] or {}).get("asin"))
                    prior = [_read_json(p) for p in ledger.paths.trajectories.glob("*.json")] if ledger.paths.trajectories.exists() else []
                    prior_fps = [x.get("behavior_fingerprint", {}) for x in prior if isinstance(x, Mapping)]
                    record["behavior_fingerprint"] = fp
                    record["exact_duplicate"] = any(exact_duplicate(fp, old) for old in prior_fps)
                    record["similarity_features"] = [similarity_features(fp, old) for old in prior_fps]
                    record["potential_near_duplicate"] = any(item.get("provisional_near_duplicate") for item in record["similarity_features"])
                ledger.save_attempt(attempt_id, record, status=status)
                if record["success"]:
                    ledger.save_trajectory(attempt_id, record, status="success")
                return status
        record["termination_reason"] = "max_steps"
        record["max_steps"] = True
        ledger.save_attempt(attempt_id, record, status="max_steps")
        return "max_steps"
    finally:
        if session_id is not None:
            try:
                call_started = time.monotonic()
                try:
                    env.release(session_id)
                finally:
                    timings["release"] += time.monotonic() - call_started
            except Exception:  # best effort; the attempt artifact is authoritative
                pass
        if guard.current_attempt_id == attempt_id:
            guard.current_attempt_id = None


def run_profile(*, scenario: str, env_endpoint: str, task_file: Path, data_root: Path,
                env_file: Path, resume: bool = False, run_id: str | None = None) -> int:
    from rollout.teacher_client import load_env_file
    cfg = load_env_file(str(env_file))
    required = ("TEACHER_API_URL", "TEACHER_API_KEY", "TEACHER_API_MODEL", "TEACHER_API_STYLE", "TEACHER_REASONING_EFFORT")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(".env.teacher 缺少: " + ", ".join(missing))
    task_data = _read_json(task_file)
    tasks = task_data.get("tasks", [])
    purpose = str(task_data.get("metadata", {}).get("purpose", "p3a_profiling"))
    expected_count = 2 if purpose == "p3a_repair" else 24
    if purpose not in {"p3a_profiling", "p3a_repair"}:
        raise ValueError("task list purpose 必须是 p3a_profiling 或 p3a_repair")
    if len(tasks) != expected_count or any(item.get("official_split") != "train" or item.get("scenario") not in {None, scenario} for item in tasks):
        raise ValueError(f"{purpose} task list 必须是固定 {expected_count} 条 TRAIN task")
    task_ids = [str(item["task_id"]) for item in tasks]
    if len(set(task_ids)) != expected_count:
        raise ValueError("P3a task list 含重复 task_id")
    if purpose == "p3a_repair":
        source_name = task_data.get("metadata", {}).get("source_task_file")
        source_path = task_file.parent / str(source_name) if source_name else None
        if source_path is None or not source_path.exists():
            raise ValueError("repair task list 缺少可验证的 source_task_file")
        source_data = _read_json(source_path)
        source_metadata = source_data.get("metadata", {})
        expected_source_hash = task_data.get("metadata", {}).get("source_task_ids_sha256")
        actual_source_hash = source_metadata.get("task_ids_sha256")
        if expected_source_hash and expected_source_hash != actual_source_hash:
            raise ValueError("repair task list 的 source task-list hash 不一致")
        source_tasks = source_data.get("tasks", [])
        source_ids = {str(item.get("task_id")) for item in source_tasks
                      if item.get("official_split") == "train"
                      and item.get("scenario") in {None, scenario}}
        if not set(task_ids).issubset(source_ids):
            raise ValueError("repair task IDs 不属于 frozen Persona profiling task list")
    recorded_ids_hash = task_data.get("metadata", {}).get("task_ids_sha256")
    valid_ids_hashes = {_task_ids_hash(task_ids), _task_ids_hash(task_ids, trailing_newline=True)}
    if recorded_ids_hash and recorded_ids_hash not in valid_ids_hashes:
        raise ValueError("P3a task list hash 与 metadata.task_ids_sha256 不一致")
    logger = ProgressLogger()
    env = TeacherEnvClient(env_endpoint, timeout=60.0)
    health = env.health().payload
    if health.get("status") != "ok":
        raise RuntimeError("ShopSimulator health check failed")
    probe = env.reset(scenario, task_ids[0])
    context = policy_context_from_reset(probe.payload, scenario)
    _policy_observation(probe.payload)
    env.release(probe.payload["session_id"])
    import uuid
    if resume and run_id is None:
        selected, other_count = _find_resume_run(data_root, scenario, task_ids, {
            "teacher_model": cfg["TEACHER_API_MODEL"],
            "api_style": cfg["TEACHER_API_STYLE"],
            "reasoning_effort": cfg["TEACHER_REASONING_EFFORT"],
            "selected_task_list_hash": recorded_ids_hash,
            "policy_observation_version": probe.payload.get(
                "policy_observation_version", POLICY_OBSERVATION_VERSION
            ),
            "profiler_protocol_version": probe.payload.get(
                "profiler_protocol_version", PROFILER_PROTOCOL_VERSION
            ),
        }, purpose=purpose)
        if selected is None:
            raise FileNotFoundError("没有找到该 scenario 的唯一未完成 profiling run；请去掉 --resume 新建 run")
        run_id = selected.run_id
        logger.line(f"[RESUME] 自动选择最新且配置兼容的未完成 run: {run_id}")
        logger.line(
            f"[RESUME] status={selected.status} | touched={selected.touched_tasks}/{expected_count} | "
            f"terminal={selected.terminal_tasks}/{expected_count} | attempts={selected.attempts} | successes={selected.successes}"
        )
        if other_count:
            logger.line(f"[RESUME] 已排除 {other_count} 个更早的未完成 run；completed run 不参与 resume。")
    if run_id is None:
        prefix = "p3a-repair-" if purpose == "p3a_repair" else "p3a-"
        run_id = prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    logger.line(f"[RUN] purpose={purpose} run_id={run_id} tasks={len(task_ids)}")
    manifest = _manifest_for(run_id, scenario, task_data, health, context.source_hash,
                             env_endpoint, probe.payload, cfg)
    extra = EXTRA_IMMUTABLE
    ledger = TeacherLedger(data_root, manifest, resume=resume, extra_immutable_fields=extra, profile=True)
    _validate_resume_task_manifest(ledger, task_ids)
    resume_command = (
        f"python3 scripts/profile_teacher.py --scenario {scenario} "
        f"--task-file {task_file} --run-id {run_id} --resume"
    )
    guard = GracefulCollectionStop(ledger, resume_command, logger)
    guard.install()
    client = TeacherClient(api_url=cfg["TEACHER_API_URL"], api_key=cfg["TEACHER_API_KEY"], model=cfg["TEACHER_API_MODEL"],
                           api_style=cfg["TEACHER_API_STYLE"], reasoning_effort=cfg["TEACHER_REASONING_EFFORT"],
                           on_retry=lambda kind, n, maximum, delay: logger.line(f"[API] {kind} | retry {n}/{maximum} | next in {delay:.1f}s"))
    try:
        for ordinal, task_id in enumerate(task_ids, 1):
            rows = _task_rows(ledger, task_id)
            successes = _success_records(ledger, task_id)
            # Determine phase from persisted artifacts rather than treating an
            # infrastructure interruption as a teacher attempt.
            first_attempts = 0
            second_attempts = 0
            for aid, status, _idx, path in rows:
                if status == "infrastructure_interrupted":
                    continue
                p = ledger.paths.root / path
                phase = _read_json(p).get("attempt_phase") if p.exists() else "first_success"
                if phase == "second_demo": second_attempts += 1
                else: first_attempts += 1
            if not successes:
                while not successes and first_attempts < PROFILE_LIMITS["first_success_max_attempts"]:
                    logger.line(f"[{scenario}] Task {ordinal:02d}/{expected_count} | Phase: first_success | Attempt: {first_attempts + 1}/3")
                    status = _run_attempt(ledger=ledger, env=env, client=client, task_id=task_id, scenario=scenario,
                                          phase="first_success", attempt_index=first_attempts + 1, expected_model=cfg["TEACHER_API_MODEL"],
                                          logger=logger, guard=guard, resume_command=resume_command)
                    first_attempts += 1
                    successes = _success_records(ledger, task_id)
                    logger.line(f"[{scenario}] {task_id}: {status}")
                if not successes and first_attempts >= PROFILE_LIMITS["first_success_max_attempts"]:
                    # A completed unsolved task is terminal for resume, but is
                    # still a profiling observation and never enters trajectories.
                    ledger.mark_profile_unsolved(task_id)
                    continue
            while successes and second_attempts < PROFILE_LIMITS["second_demo_max_attempts"]:
                logger.line(f"[{scenario}] Task {ordinal:02d}/{expected_count} | Phase: second_demo | Attempt: {second_attempts + 1}/2")
                _run_attempt(ledger=ledger, env=env, client=client, task_id=task_id, scenario=scenario,
                             phase="second_demo", attempt_index=second_attempts + 1, expected_model=cfg["TEACHER_API_MODEL"],
                             logger=logger, guard=guard, resume_command=resume_command)
                second_attempts += 1
                successes = _success_records(ledger, task_id)
        ledger.set_state("status", "complete")
    except (CollectionInfrastructureInterrupted, TeacherModelMismatch, ProfileStop) as exc:
        if not ledger.closed and ledger.get_state("status") is None:
            ledger.set_state(
                "status",
                "infrastructure_interrupted" if isinstance(exc, CollectionInfrastructureInterrupted) else "stopped",
            )
        logger.line(f"[STOP] {exc}")
        return 2
    finally:
        guard.close()
        client.close()
        env.close()
    return 0

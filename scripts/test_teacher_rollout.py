#!/usr/bin/env python3
"""Run exactly one paid teacher episode for post-fix validation.

This is deliberately separate from :mod:`scripts.profile_teacher`.  It does
not open a profiling ledger, does not select a task list, and never retries an
episode.  The configured ``TeacherClient`` may still retry an individual HTTP
request according to the normal infrastructure policy, but this command has
one reset and at most one trajectory for the explicitly supplied task.

The command is user-triggered and therefore may make real paid API calls.  It
is not imported by the test suite with a live client; the helper functions are
kept small and deterministic so the safety properties can be tested locally.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from env.teacher_env_client import TeacherEnvClient, TeacherEnvError  # noqa: E402
from rollout.prompt import (  # noqa: E402
    append_turn,
    assert_no_evaluator_leakage,
    build_initial_messages,
    policy_context_from_reset,
)
from rollout.storage import atomic_json  # noqa: E402
from rollout.teacher_client import (  # noqa: E402
    TeacherClient,
    TeacherClientError,
    TeacherResponse,
    load_env_file,
)


MAX_ACTION_STEPS = 30


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_component(value: str) -> str:
    """Return a filesystem-safe, non-empty artifact path component."""

    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return component or "task"


def _policy_observation(payload: Mapping[str, Any]) -> str:
    """Read the canonical model-visible observation from an env response.

    New remote services expose ``policy_observation`` (or the equivalent
    ``user_message``).  Raw ``observation`` and the legacy ``instruction``
    field are intentionally *not* fallbacks: accepting either could silently
    reintroduce the available-actions/Persona protocol bug this smoke is
    intended to catch.  The remote service may retain those fields for
    diagnostics, but this command must consume the explicit canonical field.
    """

    for key in ("policy_observation", "user_message"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError(
        "remote response missing canonical policy observation "
        "(expected policy_observation or user_message)"
    )


def _extract_action_for_display(text: str) -> tuple[str | None, str | None, str]:
    """Mirror the pinned upstream extraction/parse semantics for diagnostics.

    The remote service remains the source of truth and receives the original
    visible response.  This helper is only for the terminal line and local
    artifact, so a malformed response is represented without being repaired.
    """

    normalized = text.replace("\\n", "\n")
    action_text = normalized.split("\nAction: ", 1)[1] if "\nAction: " in normalized else normalized
    action_text = action_text.strip()
    match = re.match(r"(.+)\[(.+)\]", action_text)
    if match is None:
        return None, None, action_text
    name, argument = match.groups()
    return name, argument, action_text


def _available_summary(value: Any) -> dict[str, Any]:
    """Keep diagnostics compact while preserving the legal-action signal."""

    if not isinstance(value, Mapping):
        return {"clickable_count": "N/A", "has_search_bar": "N/A"}
    clickables = value.get("clickables")
    count = len(clickables) if isinstance(clickables, (list, tuple)) else "N/A"
    return {"clickable_count": count, "has_search_bar": value.get("has_search_bar", "N/A")}


def _new_artifact_dir(output_root: Path, scenario: str, task_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = output_root / _safe_component(scenario) / f"{stamp}-{_safe_component(task_id)}"
    # A same-second invocation must never overwrite a previous debug run.
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _write_record(path: Path, record: Mapping[str, Any]) -> None:
    """Publish a debug record atomically after every meaningful state change."""

    atomic_json(path, dict(record))


def _print_response(step: int, response: TeacherResponse, action_name: str | None,
                    action_arg: str | None, action_text: str) -> None:
    print(f"\n[step {step}] Teacher visible response:")
    print(response.text)
    normalized_name = action_name.strip().lower() if isinstance(action_name, str) else "N/A"
    normalized_arg = action_arg.strip().lower() if isinstance(action_arg, str) else "N/A"
    print(
        f"[step {step}] extracted_action={action_text or 'N/A'} "
        f"normalized_type={normalized_name} normalized_arg={normalized_arg}"
    )
    print(
        f"[step {step}] API latency={response.latency_s:.2f}s "
        f"retries={response.retries}"
    )


def run_one(*, scenario: str, task_id: str, endpoint: str, env_file: Path,
            output_root: Path, max_action_steps: int = MAX_ACTION_STEPS) -> int:
    """Execute one task/attempt and persist a debug-only artifact.

    Return codes intentionally distinguish a normal terminal episode (0), a
    teacher/environment outcome that needs inspection (1), and preflight or
    infrastructure failures (2).  No code path starts another task.
    """

    cfg = load_env_file(str(env_file))
    required = (
        "TEACHER_API_URL", "TEACHER_API_KEY", "TEACHER_API_MODEL",
        "TEACHER_API_STYLE", "TEACHER_REASONING_EFFORT",
    )
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        print("Missing teacher config: " + ", ".join(missing), file=sys.stderr)
        return 2
    if max_action_steps < 1 or max_action_steps > MAX_ACTION_STEPS:
        print(f"--max-action-steps must be between 1 and {MAX_ACTION_STEPS}", file=sys.stderr)
        return 2

    # Make the paid-call boundary unmistakable before touching the relay.
    print("ABOUT TO MAKE REAL PAID TEACHER API CALLS")
    print(f"scenario={scenario}")
    print(f"task_id={task_id}")
    print("max_attempts=1")

    try:
        artifact_dir = _new_artifact_dir(output_root, scenario, task_id)
    except OSError as exc:
        print(f"Cannot create debug artifact directory: {type(exc).__name__}", file=sys.stderr)
        return 2
    record_path = artifact_dir / "attempt.json"
    manifest_path = artifact_dir / "run_manifest.json"
    manifest = {
        "purpose": "p3a_one_task_paid_debug_smoke",
        "scenario": scenario,
        "task_id": str(task_id),
        "teacher_model": cfg["TEACHER_API_MODEL"],
        "api_style": cfg["TEACHER_API_STYLE"],
        "reasoning_effort": cfg["TEACHER_REASONING_EFFORT"],
        # Do not persist a potentially user-specific URL (it may contain a
        # relay token/query string).  The endpoint is printed only by the
        # caller's tunnel script; this debug artifact needs no copy of it.
        "max_action_steps": max_action_steps,
        "created_at": _utc_now(),
        "status": "in_progress",
    }
    _write_record(manifest_path, manifest)
    record: dict[str, Any] = {
        "purpose": manifest["purpose"],
        "scenario": scenario,
        "task_id": str(task_id),
        "started_at": _utc_now(),
        "status": "in_progress",
        "steps": [],
        "visible_responses": [],
        "api_calls": 0,
        "environment_steps": 0,
        "error": None,
    }
    _write_record(record_path, record)

    env: TeacherEnvClient | None = None
    client: TeacherClient | None = None
    session_id: str | None = None
    messages: list[dict[str, str]] = []
    outcome = "infrastructure_failure"
    started = time.monotonic()
    try:
        env = TeacherEnvClient(endpoint, timeout=60.0)
        health = env.health()
        record["health"] = {"status": health.payload.get("status"), "latency_s": health.latency_s}
        if health.payload.get("status") != "ok":
            raise TeacherEnvError("ShopSimulator health check failed", kind="infrastructure")
        reset = env.reset(scenario, str(task_id))
        session_id = reset.payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise TeacherEnvError("reset response missing session_id", kind="environment")
        context = policy_context_from_reset(reset.payload, scenario)
        initial_observation = _policy_observation(reset.payload)
        messages = build_initial_messages(context, initial_observation)
        assert_no_evaluator_leakage(messages)
        record["reset"] = {
            "latency_s": reset.latency_s,
            "environment_version": reset.payload.get("environment_version"),
            "policy_observation": initial_observation,
            "available_actions": _available_summary(reset.payload.get("available_actions")),
        }
        record["messages"] = list(messages)
        _write_record(record_path, record)

        client = TeacherClient(
            api_url=cfg["TEACHER_API_URL"], api_key=cfg["TEACHER_API_KEY"],
            model=cfg["TEACHER_API_MODEL"], api_style=cfg["TEACHER_API_STYLE"],
            reasoning_effort=cfg["TEACHER_REASONING_EFFORT"],
            on_retry=lambda kind, n, maximum, delay: print(
                f"[API] {kind} | retry {n}/{maximum} | next in {delay:.1f}s"
            ),
        )
        for step_number in range(1, max_action_steps + 1):
            api_started = time.monotonic()
            response = client.generate(messages)
            api_elapsed = time.monotonic() - api_started
            record["api_calls"] += 1
            record["visible_responses"].append(response.text)
            action_name, action_arg, action_text = _extract_action_for_display(response.text)
            _print_response(step_number, response, action_name, action_arg, action_text)
            step_record: dict[str, Any] = {
                "step": step_number,
                "teacher_response": response.text,
                "extracted_action": action_text,
                "normalized_action_type": action_name.strip().lower() if action_name else None,
                "normalized_action_argument": action_arg.strip().lower() if action_arg else None,
                "api_latency_s": response.latency_s,
                "api_wall_latency_s": api_elapsed,
                "api_retries": response.retries,
                "api_model": response.model or "N/A",
                "http_status": response.status_code,
                "request_id": response.request_id or "N/A",
                "input_tokens": response.input_tokens if response.input_tokens is not None else "N/A",
                "output_tokens": response.output_tokens if response.output_tokens is not None else "N/A",
            }
            if not action_text or action_name is None or action_arg is None:
                step_record["outcome"] = "malformed_action"
                record["steps"].append(step_record)
                record["status"] = "malformed_action"
                record["finished_at"] = _utc_now()
                outcome = "malformed_action"
                _write_record(record_path, record)
                print(f"[step {step_number}] malformed action; stopping one-task smoke")
                break
            env_started = time.monotonic()
            try:
                result = env.step(session_id, response.text)
            except TeacherEnvError as exc:
                step_record.update({
                    "outcome": getattr(exc, "kind", "environment"),
                    "status_code": getattr(exc, "status_code", None),
                })
                record["steps"].append(step_record)
                record["status"] = getattr(exc, "kind", "environment")
                record["finished_at"] = _utc_now()
                record["error"] = {"kind": getattr(exc, "kind", "environment"),
                                    "status_code": getattr(exc, "status_code", None)}
                outcome = "infrastructure_failure" if getattr(exc, "kind", None) == "infrastructure" else record["status"]
                _write_record(record_path, record)
                print(f"[step {step_number}] environment error: {record['status']}", file=sys.stderr)
                break
            env_elapsed = time.monotonic() - env_started
            payload = result.payload
            policy_observation = None
            if isinstance(payload.get("policy_observation"), str):
                policy_observation = payload["policy_observation"]
            elif isinstance(payload.get("user_message"), str):
                policy_observation = payload["user_message"]
            step_record.update({
                "environment_latency_s": result.latency_s,
                "environment_wall_latency_s": env_elapsed,
                "action": payload.get("action", action_text),
                "action_valid": payload.get("action_valid", "N/A"),
                "available_actions": _available_summary(payload.get("available_actions")),
                "policy_observation": policy_observation,
                "done": bool(payload.get("done") or payload.get("over")),
                "reward": payload.get("reward", "N/A"),
                "reward_detail": payload.get("reward_detail", {}),
            })
            record["steps"].append(step_record)
            record["environment_steps"] += 1
            print(
                f"[step {step_number}] action_valid={step_record['action_valid']} "
                f"available_clickables={step_record['available_actions']['clickable_count']} "
                f"env latency={result.latency_s:.2f}s"
            )
            if "action_valid" not in payload:
                outcome = "protocol_error"
                record["status"] = outcome
                record["finished_at"] = _utc_now()
                record["error"] = {"kind": "missing_action_valid"}
                _write_record(record_path, record)
                print(f"[step {step_number}] missing action_valid; stopping", file=sys.stderr)
                break
            if step_record["action_valid"] is False:
                outcome = "invalid_action"
                record["status"] = outcome
                record["finished_at"] = _utc_now()
                _write_record(record_path, record)
                print(f"[step {step_number}] invalid_action; stopping one-task smoke")
                break
            if step_record["done"]:
                detail = payload.get("reward_detail")
                success_value = detail.get("r_succ") if isinstance(detail, Mapping) else None
                if success_value is None:
                    success_value = payload.get("reward", 0)
                outcome = "success" if success_value == 1 else "terminal_unsuccessful"
                record["status"] = outcome
                record["finished_at"] = _utc_now()
                _write_record(record_path, record)
                print(f"[step {step_number}] terminal outcome={outcome} reward={payload.get('reward', 'N/A')}")
                break
            if not isinstance(policy_observation, str) or not policy_observation:
                outcome = "protocol_error"
                record["status"] = outcome
                record["finished_at"] = _utc_now()
                record["error"] = {"kind": "missing_policy_observation"}
                _write_record(record_path, record)
                print(f"[step {step_number}] missing canonical policy observation; stopping", file=sys.stderr)
                break
            messages = append_turn(messages, response.text, policy_observation)
            assert_no_evaluator_leakage(messages)
            record["messages"] = list(messages)
            _write_record(record_path, record)
        else:
            outcome = "max_steps"
            record["status"] = outcome
            record["finished_at"] = _utc_now()
            _write_record(record_path, record)
    except (TeacherClientError, TeacherEnvError) as exc:
        outcome = getattr(exc, "kind", "infrastructure")
        record["status"] = outcome
        record["error"] = {"kind": outcome, "status_code": getattr(exc, "status_code", None),
                            "retries": getattr(exc, "retries", None)}
        record["finished_at"] = _utc_now()
        _write_record(record_path, record)
        print(f"One-task smoke failed: {outcome}", file=sys.stderr)
    except KeyboardInterrupt:
        outcome = "interrupted"
        record["status"] = outcome
        record["error"] = {"kind": "SIGINT"}
        record["finished_at"] = _utc_now()
        _write_record(record_path, record)
        print("[STOP] Ctrl+C; debug artifact preserved", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - redact provider/environment details
        outcome = "local_error"
        record["status"] = outcome
        record["error"] = {"kind": type(exc).__name__}
        record["finished_at"] = _utc_now()
        _write_record(record_path, record)
        print(f"One-task smoke failed locally: {type(exc).__name__}", file=sys.stderr)
    finally:
        if session_id and env is not None:
            try:
                env.release(session_id)
            except Exception:  # noqa: BLE001 - release is best effort
                pass
        if client is not None:
            client.close()
        if env is not None:
            env.close()
        record.setdefault("finished_at", _utc_now())
        record["status"] = outcome
        record["total_wall_latency_s"] = time.monotonic() - started
        _write_record(record_path, record)
        manifest["status"] = outcome
        manifest["finished_at"] = record["finished_at"]
        _write_record(manifest_path, manifest)

    print(f"Debug artifact: {record_path}")
    return 0 if outcome == "success" else 1 if outcome in {
        "invalid_action", "malformed_action", "terminal_unsuccessful", "max_steps",
    } else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-task/one-attempt paid teacher debug smoke")
    parser.add_argument("--scenario", choices=("single", "single_persona"), required=True)
    parser.add_argument("--task-id", required=True, help="one frozen TRAIN task id; no task list is inferred")
    parser.add_argument("--endpoint", default="http://127.0.0.1:5500")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.teacher")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "teacher_debug")
    parser.add_argument("--max-action-steps", type=int, default=MAX_ACTION_STEPS)
    args = parser.parse_args(argv)
    return run_one(
        scenario=args.scenario, task_id=str(args.task_id), endpoint=args.endpoint,
        env_file=args.env_file, output_root=args.output_root,
        max_action_steps=args.max_action_steps,
    )


if __name__ == "__main__":
    raise SystemExit(main())

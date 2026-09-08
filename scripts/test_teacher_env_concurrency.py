#!/usr/bin/env python3
"""Non-paid ten-session remote environment isolation/stress test.

The script uses only textual scripted actions and the normal remote parser.  It
never constructs a TeacherClient and never writes profiling/teacher data.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from env.teacher_env_client import TeacherEnvClient, TeacherEnvError  # noqa: E402


def _tasks(path: Path, limit: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = list(data.get("tasks", []))
    if len(rows) < limit:
        raise ValueError(f"stress config has only {len(rows)} tasks")
    return [{"scenario": str(item["scenario"]), "task_id": str(item["task_id"]),
             "actions": list(item["actions"])} for item in rows[:limit]]


def _reset(endpoint: str, item: dict[str, Any]) -> dict[str, Any]:
    scenario, task_id = item["scenario"], item["task_id"]
    client = TeacherEnvClient(endpoint, timeout=60.0)
    result = client.reset(scenario, task_id)
    return {"scenario": scenario, "task_id": task_id, "client": client,
            "session_id": result.payload["session_id"], "slot": result.payload.get("internal_slot"),
            "reset": result.payload, "actions": item["actions"]}


def _scripted_path(row: dict[str, Any]) -> dict[str, Any]:
    client: TeacherEnvClient = row["client"]
    session = row["session_id"]
    task_id = row["task_id"]
    scenario = row["scenario"]
    final = None
    for index, action in enumerate(row["actions"]):
        result = client.step(session, f"Thought: scripted isolation check\nAction: {action}",
                             expected_task_id=task_id, expected_scenario=scenario)
        if index == 0 and result.payload.get("action_valid") is False:
            raise TeacherEnvError(f"{scenario}/{task_id}: scripted search invalid")
        final = result
    if final is None:
        raise TeacherEnvError(f"{scenario}/{task_id}: empty scripted path")
    purchase = final.payload.get("purchase") or {}
    if not final.payload.get("done") or str(purchase.get("asin", "")).lower() != task_id.lower():
        raise TeacherEnvError(f"{scenario}/{task_id}: terminal purchase/reward mismatch")
    return {"task_id": task_id, "scenario": scenario, "session_id": session,
            "slot": row["slot"], "steps": len(row["actions"]), "reward": final.payload.get("reward"),
            "query_match": (final.payload.get("reward_detail") or {}).get("query_match")}


def main() -> int:
    parser = argparse.ArgumentParser(description="non-paid remote ten-session isolation stress")
    parser.add_argument("--endpoint", default="http://127.0.0.1:5500")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.limit < 2 or args.limit > 10:
        parser.error("--limit must be between 2 and 10")
    items = _tasks(ROOT / "configs/teacher/p3a_environment_stress_tasks.json", args.limit)
    if len(items) != args.limit:
        raise SystemExit(f"could not form {args.limit} mixed TRAIN tasks")
    with TeacherEnvClient(args.endpoint) as probe:
        baseline = probe.health().payload
    rows: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.limit) as pool:
            rows = list(pool.map(lambda item: _reset(args.endpoint, item), items))
        sessions = [row["session_id"] for row in rows]
        slots = [row["slot"] for row in rows]
        if len(set(sessions)) != args.limit or len(set(slots)) != args.limit:
            raise TeacherEnvError("session or internal slot collision")
        with TeacherEnvClient(args.endpoint) as probe:
            active = probe.health().payload
        if active.get("active_sessions") != args.limit:
            raise TeacherEnvError(f"expected {args.limit} active sessions, got {active.get('active_sessions')}")
        if active.get("shared_runtime") is not True or active.get("runtime_loaded") is not True:
            raise TeacherEnvError("remote service is not reporting one shared loaded runtime")
        peak_memory = active
        # Release one exact session before the remaining paths.  The other
        # nine must stay alive and keep their own task/goal bindings.
        first = rows.pop(0)
        first["client"].release(first["session_id"])
        first["client"].close()
        with TeacherEnvClient(args.endpoint) as probe:
            after_release = probe.health().payload
        if after_release.get("active_sessions") != args.limit - 1:
            raise TeacherEnvError("releasing one session affected another session")
        with ThreadPoolExecutor(max_workers=len(rows)) as pool:
            results = list(pool.map(_scripted_path, rows))
        for row in rows:
            row["client"].release(row["session_id"])
            row["client"].close()
        with TeacherEnvClient(args.endpoint) as probe:
            final = probe.health().payload
        if final.get("active_sessions") != 0:
            raise TeacherEnvError(f"session leak: active_sessions={final.get('active_sessions')}")
        print(json.dumps({"status": "PASS", "sessions": args.limit,
                          "unique_slots": sorted(slots),
                          "baseline_memory": {k: baseline.get(k) for k in baseline if "memory" in k or "rss" in k},
                          "peak_memory": {k: peak_memory.get(k) for k in peak_memory if "memory" in k or "rss" in k},
                          "results": results}, ensure_ascii=False))
        print("Non-paid remote isolation stress: PASS")
        return 0
    except Exception as exc:
        for row in rows:
            try:
                row["client"].release(row["session_id"])
                row["client"].close()
            except Exception:
                pass
        print(f"Non-paid remote isolation stress failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

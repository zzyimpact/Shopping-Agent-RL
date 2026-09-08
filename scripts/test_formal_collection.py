#!/usr/bin/env python3
"""User-triggered 2-worker / 2-task paid formal-pipeline smoke.

This calls the real P3c engine but writes only to ``data/teacher_formal_smoke``.
Smoke bounds are Pass A only and at most one teacher attempt per known-short task;
they do not alter p3b-v1.1 or enter formal/P3a/SFT data.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout.collection_policy import DEFAULT_POLICY_PATH  # noqa: E402
from rollout.formal_collector import run_from_configuration  # noqa: E402


def main() -> int:
    task_config = ROOT / "configs" / "teacher" / "p3a_concurrency_probe_tasks.json"
    value = json.loads(task_config.read_text(encoding="utf-8"))
    task_ids = [str(item["task_id"]) for item in value["tasks"][:2]]
    if len(task_ids) != 2 or len(set(task_ids)) != 2:
        print("Formal smoke task config must contain two unique tasks", file=sys.stderr)
        return 2
    print("ABOUT TO MAKE REAL PAID TEACHER API CALLS")
    print("purpose=formal_collector_smoke_only")
    print("policy=p3b-v1.1")
    print("scenario=single")
    print("workers=2")
    print("tasks=2")
    print("max_attempts_per_task=1")
    print("output=data/teacher_formal_smoke (never formal/P3a/SFT)")
    try:
        summary = run_from_configuration(
            project_root=ROOT, scenario="single", policy_path=DEFAULT_POLICY_PATH,
            manifest_dir=ROOT / ".cache" / "teacher_manifests",
            data_root=ROOT / "data" / "teacher_formal_smoke",
            env_file=ROOT / ".env.teacher", endpoint="http://127.0.0.1:5500",
            workers=2, selected_primary_ids=task_ids, smoke=True,
        )
    except Exception as exc:
        print(f"Formal smoke failed: {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr)
        return 2
    print("Smoke pipeline completed. Artifacts are isolated from formal collection.")
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

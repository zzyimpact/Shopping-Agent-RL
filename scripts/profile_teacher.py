#!/usr/bin/env python3
"""Run the small, single-worker P3a teacher profiling plan.

This command is intentionally user-triggered: it can call the configured paid
relay, but no test or Codex validation invokes it with the real endpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout.profiler import run_profile


def main() -> int:
    parser = argparse.ArgumentParser(description="P3a single-worker teacher rollout profiling")
    parser.add_argument("--scenario", choices=("single", "single_persona"), required=True)
    parser.add_argument("--resume", action="store_true", help="resume the unambiguous incomplete run")
    parser.add_argument("--run-id", help="explicit profiling run id (normally unnecessary)")
    parser.add_argument("--endpoint", default="http://127.0.0.1:5500")
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.teacher")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "teacher_profile")
    args = parser.parse_args()
    task_file = args.task_file or ROOT / "configs" / "teacher" / f"p3a_{args.scenario}_tasks.json"
    try:
        return run_profile(scenario=args.scenario, env_endpoint=args.endpoint, task_file=task_file,
                           data_root=args.data_root, env_file=args.env_file, resume=args.resume,
                           run_id=args.run_id)
    except KeyboardInterrupt:
        print("[STOP] Ctrl+C received before an attempt could be started; no new API request was sent.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - user-facing preflight error, no provider body
        print(f"Preflight/profile failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

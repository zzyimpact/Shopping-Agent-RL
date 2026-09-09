#!/usr/bin/env python3
"""P3c formal teacher collection CLI (p3b-v1.1 only)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rollout.collection_policy import DEFAULT_POLICY_PATH, load_collection_policy  # noqa: E402
from env.teacher_env_client import TeacherEnvError  # noqa: E402
from rollout.formal_collector import auto_resumable_error, run_from_configuration  # noqa: E402


def main() -> int:
    policy = load_collection_policy(DEFAULT_POLICY_PATH)
    parser = argparse.ArgumentParser(description="p3b-v1.1 formal teacher collector")
    parser.add_argument("--scenario", choices=("single", "single_persona"), required=True)
    capacity = parser.add_mutually_exclusive_group()
    capacity.add_argument("--api-workers", help="numbered API profiles, e.g. 1:8,2:12")
    capacity.add_argument("--workers", type=int, default=int(policy["concurrency"]["workers_default"]))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--auto-resume", action="store_true",
                        help="after recoverable infrastructure stop, resume the same run every 120s")
    parser.add_argument("--run-id")
    parser.add_argument("--endpoint", default="http://127.0.0.1:5500")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.teacher")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / ".cache" / "teacher_manifests")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "teacher_raw")
    args = parser.parse_args()
    if args.run_id and not args.resume:
        parser.error("--run-id 只能与 --resume 一起使用")
    recovering = False
    try:
        while True:
            try:
                summary = run_from_configuration(
                    project_root=ROOT, scenario=args.scenario, policy_path=DEFAULT_POLICY_PATH,
                    manifest_dir=args.manifest_dir, data_root=args.data_root,
                    env_file=args.env_file, endpoint=args.endpoint, workers=args.workers,
                    resume=args.resume, run_id=args.run_id, api_workers=args.api_workers,
                )
            except TeacherEnvError as exc:
                # A retry's preflight may still find the tunnel unavailable.
                # Only continue after a gracefully closed, pinned prior run.
                if not (recovering and auto_resumable_error(exc)):
                    raise
                print("[AUTO_RESUME] environment network still unavailable", flush=True)
            else:
                if summary["status"] in {"complete", "quota_unmet"}:
                    return 0
                if not (args.auto_resume and summary.get("auto_resume_eligible", False)):
                    return 2
                args.run_id = summary["run_id"]
                args.resume = True
                recovering = True
            # run_from_configuration has drained workers and closed SQLite.
            # Keep every runtime argument; never select or create another run.
            print(f"[AUTO_RESUME] scenario={args.scenario} run_id={args.run_id} "
                  "waiting 120s; Ctrl+C to stop", flush=True)
            time.sleep(120)
    except KeyboardInterrupt:
        print("\n[AUTO_RESUME] cancelled; no restart", flush=True)
        return 130
    except Exception as exc:
        # TeacherClient/Env errors intentionally contain no provider body,
        # request headers, key, or URL. Never dump arbitrary object repr here.
        print(f"Formal collection preflight/run failed: {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

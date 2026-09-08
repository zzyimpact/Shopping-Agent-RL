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
from rollout.formal_collector import run_from_configuration  # noqa: E402


def main() -> int:
    policy = load_collection_policy(DEFAULT_POLICY_PATH)
    parser = argparse.ArgumentParser(description="p3b-v1.1 formal teacher collector")
    parser.add_argument("--scenario", choices=("single", "single_persona"), required=True)
    parser.add_argument("--workers", type=int, default=int(policy["concurrency"]["workers_default"]))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--endpoint", default="http://127.0.0.1:5500")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.teacher")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / ".cache" / "teacher_manifests")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "teacher_raw")
    args = parser.parse_args()
    if args.run_id and not args.resume:
        parser.error("--run-id 只能与 --resume 一起使用")
    resume = args.resume
    run_id = args.run_id
    while True:
        try:
            summary = run_from_configuration(
                project_root=ROOT, scenario=args.scenario, policy_path=DEFAULT_POLICY_PATH,
                manifest_dir=args.manifest_dir, data_root=args.data_root,
                env_file=args.env_file, endpoint=args.endpoint, workers=args.workers,
                resume=resume, run_id=run_id,
            )
            status = summary["status"]
            if status in {"complete", "quota_unmet"}:
                return 0
            # An infrastructure stop has already flushed the run state and
            # printed its exact resume command. Retry that same immutable run.
            if status == "stopped" and summary.get("last_error"):
                run_id = str(summary["run_id"])
                resume = True
                print("Infrastructure stop detected; automatic resume in 15 seconds...", flush=True)
                time.sleep(15)
                continue
            return 2
        except KeyboardInterrupt:
            print("Automatic resume cancelled.", file=sys.stderr)
            return 130
        except Exception as exc:
            # Once explicitly resuming a known run, transient environment/API
            # disconnects are retried without changing its immutable config.
            if resume and run_id:
                print(
                    f"Resume attempt failed ({type(exc).__name__}); automatic resume in 15 seconds...",
                    file=sys.stderr, flush=True,
                )
                time.sleep(15)
                continue
            # TeacherClient/Env errors intentionally contain no provider body,
            # request headers, key, or URL. Never dump arbitrary object repr here.
            print(f"Formal collection preflight/run failed: {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only one-screen inspection for the latest formal teacher run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]


def _runs(data_root: Path, scenario: str) -> list[tuple[str, Path]]:
    result = []
    for directory in (data_root / scenario).glob("*") if (data_root / scenario).exists() else []:
        path = directory / "run_manifest.json"
        if not path.exists():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("purpose") == "formal_teacher_collection":
            result.append((str(manifest.get("created_at", directory.name)), directory))
    return sorted(result)


def _attempt_totals(run_root: Path, db: sqlite3.Connection) -> dict[str, int]:
    totals = {"api_calls": 0, "retries": 0, "http_429": 0, "http_5xx": 0,
              "input_tokens": 0, "output_tokens": 0}
    for relative_path, in db.execute("SELECT artifact_path FROM attempts"):
        try:
            record = json.loads((run_root / relative_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        totals["api_calls"] += len(record.get("api_diagnostics", []))
        totals["retries"] += int(record.get("infrastructure_retries", 0) or 0)
        totals["input_tokens"] += int(record.get("total_input_tokens", 0) or 0)
        totals["output_tokens"] += int(record.get("total_output_tokens", 0) or 0)
        for event in record.get("retry_events", []):
            code = event.get("status_code")
            totals["http_429"] += int(code == 429)
            totals["http_5xx"] += int(isinstance(code, int) and 500 <= code <= 599)
    return totals


def inspect(data_root: Path, scenario: str, run_id: str | None, last: int) -> int:
    runs = _runs(data_root, scenario)
    if run_id is None:
        if not runs:
            print(f"No formal runs found for scenario={scenario}", file=sys.stderr)
            return 2
        run_root = runs[-1][1]
    else:
        run_root = data_root / scenario / run_id
        if not (run_root / "run_manifest.json").exists():
            print(f"Formal run not found: {run_id}", file=sys.stderr)
            return 2
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    db = sqlite3.connect(run_root / "state.sqlite")
    try:
        row = db.execute("SELECT value FROM run_state WHERE key='formal_state'").fetchone()
        if row is None:
            print("Formal state is missing", file=sys.stderr)
            return 2
        state = json.loads(row[0])
        accepted = int(db.execute("SELECT COUNT(*) FROM accepted").fetchone()[0])
        totals = _attempt_totals(run_root, db)
    finally:
        db.close()
    unique = sum(1 for task in state["tasks"].values() if task["first_success"])
    completed = sum(1 for slot in state["slots"].values() if slot["fulfilled"] or slot["exhausted"])
    try:
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(state["started_at"])).total_seconds()
    except (TypeError, ValueError):
        elapsed = 0.0
    history = state.get("scheduler_history", [])
    runtime = history[-1] if history else {}
    workers = state.get("runtime_workers", runtime.get("workers", manifest["workers"]))
    capacities = runtime.get("api_workers", {})
    capacity_arg = (
        "--api-workers " + ",".join(f"{key}:{count}" for key, count in capacities.items())
        if capacities and "legacy" not in capacities else f"--workers {workers}"
    )
    resume = (
        f"python3 scripts/collect_teacher.py --scenario {scenario} {capacity_arg} "
        f"--run-id {manifest['run_id']} --resume"
    )
    rows = [
        ("Run ID", manifest["run_id"]), ("Status", state["status"]),
        ("Scenario", scenario),
        ("Policy", f"{manifest['policy_version']} / {manifest['policy_hash'][:12]}…"),
        ("Workers", workers), ("Current pass", state["current_pass"]),
        ("Coverage slots", f"{completed}/{len(state['slots'])}"),
        ("Unique successful", unique), ("Accepted", accepted),
        ("Genuine attempts", state["genuine_attempts"]),
        ("Reserve used", len(state["reserve_allocations"])),
        ("API calls / retries", f"{totals['api_calls']} / {totals['retries']}"),
        ("HTTP 429 / 5xx", f"{totals['http_429']} / {totals['http_5xx']}"),
        ("Input / output tokens", f"{totals['input_tokens']} / {totals['output_tokens']}"),
        ("Elapsed", f"{elapsed:.0f}s"), ("Last success", state.get("last_success") or "N/A"),
        ("Last error", state.get("last_error") or "N/A"),
        ("Last update", state.get("last_update") or "N/A"),
        ("Resume command", resume),
    ]
    print("\n".join(f"{name}: {value}" for name, value in rows))
    log_path = data_root / "collector.log"
    if log_path.exists() and last:
        matching = [line for line in log_path.read_text(encoding="utf-8").splitlines()
                    if f"run_id={manifest['run_id']}" in line]
        print(f"\nRecent collector.log entries (last {last}):")
        print("\n".join(matching[-last:]) or "N/A")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="inspect a formal teacher collection run")
    parser.add_argument("--scenario", choices=("single", "single_persona"), required=True)
    parser.add_argument("--latest", action="store_true", help="inspect latest run (default)")
    parser.add_argument("--run-id")
    parser.add_argument("--last", type=int, default=20, metavar="N")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "teacher_raw")
    args = parser.parse_args()
    if args.last < 0 or args.last > 200:
        parser.error("--last must be between 0 and 200")
    if args.latest and args.run_id:
        parser.error("--latest and --run-id are mutually exclusive")
    return inspect(args.data_root, args.scenario, args.run_id, args.last)


if __name__ == "__main__":
    raise SystemExit(main())

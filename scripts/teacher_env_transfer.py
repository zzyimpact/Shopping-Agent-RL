#!/usr/bin/env python3
"""Visible, time-bounded SCP for environment setup; never contacts a teacher API."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def transfer(source: str, destination: str, *, timeout: float = 90) -> int:
    command = [
        "scp", "-o", "ConnectTimeout=15", "-o", "ConnectionAttempts=1",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
        source, destination,
    ]
    # Own only this transfer's process group, including the child ssh process.
    process = subprocess.Popen(command, start_new_session=True)
    started = time.monotonic()
    try:
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                return process.wait(timeout=min(10, remaining))
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    raise
                print(f"SCP still running ({elapsed:.0f}s / {timeout:g}s limit)...",
                      file=sys.stderr, flush=True)
    except subprocess.TimeoutExpired:
        print(f"SCP timed out after {timeout:g}s; environment setup stopped. "
              "No collector was started. Retry setup after checking SSH connectivity.",
              file=sys.stderr, flush=True)
        return 124
    finally:
        # Also cleans up on Ctrl+C. Never targets an existing service/tunnel.
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def manifest_valid(path: Path, *, expected_count: int, expected_hash: str) -> bool:
    # Use the same frozen manifest validator as formal preflight, without creating clients.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from rollout.formal_collector import load_manifest
    try:
        load_manifest(path, expected_count=expected_count, expected_hash=expected_hash)
        return True
    except (ValueError, OSError, TypeError, KeyError):
        return False


def check_manifest(path: Path, name: str) -> bool:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from rollout.collection_policy import load_collection_policy
    policy = load_collection_policy(root / "configs/teacher/formal_collection_p3b_v1_1.yaml")
    for scenario in policy["scope"]["scenarios"]:
        source = policy["task_sources"][scenario]
        for kind in ("train", "primary"):
            if source[kind + "_manifest"] == name:
                return manifest_valid(path, expected_count=source[kind + "_task_count"],
                                      expected_hash=source[kind + "_task_ids_sha256"])
    return False


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--check-manifest":
        raise SystemExit(0 if check_manifest(Path(sys.argv[2]), sys.argv[3]) else 1)
    if len(sys.argv) != 3:
        raise SystemExit("usage: teacher_env_transfer.py SOURCE DESTINATION")
    try:
        raise SystemExit(transfer(sys.argv[1], sys.argv[2]))
    except KeyboardInterrupt:
        raise SystemExit(130)

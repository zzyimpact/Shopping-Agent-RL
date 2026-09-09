#!/usr/bin/env python3
"""Visible, time-bounded SCP for environment setup; never contacts a teacher API."""
from __future__ import annotations

import os
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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: teacher_env_transfer.py SOURCE DESTINATION")
    try:
        raise SystemExit(transfer(sys.argv[1], sys.argv[2]))
    except KeyboardInterrupt:
        raise SystemExit(130)

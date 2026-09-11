#!/usr/bin/env python3
"""Linux-only lifecycle and bounded admission check for the one localhost TEST service."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT.parent / "run"
PIDFILE = STATE / "shop_env_test_5200.pid"
LOGFILE = ROOT.parent / "logs/shop_env_test_5200.log"
URL = "http://127.0.0.1:5200"
MANIFESTS = Path("/root/data/shopsim/manifests")
INTERPRETER = "/root/miniconda3/bin/python"


def service_command():
    return [INTERPRETER, "-u", str(ROOT / "scripts/remote_teacher_env_service.py"),
            "--host", "127.0.0.1", "--port", "5200", "--task-split", "test",
            "--manifests", str(MANIFESTS)]


def request(path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def health():
    code, body = request("/health", timeout=3)
    if code != 200 or body.get("status") != "ok" or body.get("task_split") != "test" or (
            body.get("environment_version") != "task-scoped-v3-multisession"):
        raise RuntimeError(f"not a compatible TEST service: {body}")
    return body


def process_identity(pid):
    proc = Path(f"/proc/{pid}")
    args = (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    # starttime distinguishes a reused PID; split after comm, which can contain spaces.
    start = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
    return {"pid": pid, "starttime": start, "argv": [arg.decode() for arg in args]}


def owned_identity():
    saved = json.loads(PIDFILE.read_text())
    current = process_identity(saved["pid"])
    if saved != current or current["argv"] != service_command():
        raise RuntimeError("pidfile ownership mismatch; no signal sent")
    return saved


def start():
    if PIDFILE.exists():
        owned_identity()
        print(json.dumps(health()))
        print("Already running; no restart")
        return
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 5200))  # Fail on occupied port; never replace its owner.
    LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    with LOGFILE.open("ab") as log:
        proc = subprocess.Popen(service_command(), cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                env={**os.environ, "UPSTREAM_ROOT": "/root/ShopSimulator",
                                     "OMP_NUM_THREADS": "2"})
    try:
        identity = process_identity(proc.pid)
        if identity["argv"] != service_command():
            raise RuntimeError("unexpected service command")
        PIDFILE.write_text(json.dumps(identity) + "\n")
        for _ in range(30):
            if proc.poll() is not None:
                raise RuntimeError(f"service exited; see {LOGFILE}")
            try:
                body = health()
            except (OSError, RuntimeError):
                time.sleep(1)
                continue
            print(json.dumps(body))
            print(f"Started PID {proc.pid}; log={LOGFILE}; pidfile={PIDFILE}")
            return
        raise RuntimeError(f"health timeout; see {LOGFILE}")
    except BaseException:
        if proc.poll() is None:
            proc.terminate()  # This Popen child only; never a discovered process.
            proc.wait(timeout=10)
        PIDFILE.unlink(missing_ok=True)
        raise


def stop():
    if not PIDFILE.exists():
        print("No owned TEST service pidfile; nothing stopped")
        return
    saved = owned_identity()
    # Linux pidfd pins the process identity across the final check and signal.
    fd = os.pidfd_open(saved["pid"])
    try:
        owned_identity()
        if health().get("active_sessions") != 0:
            raise RuntimeError("active eval sessions; stop evaluation first")
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        import select
        if not select.select([fd], [], [], 10)[0]:
            raise RuntimeError("TEST service did not exit; pidfile retained, no forced kill")
    finally:
        os.close(fd)
    PIDFILE.unlink()
    print(f"Stopped owned TEST PID {saved['pid']}; 5100 untouched")


def check():
    print(json.dumps(health()))
    for scenario in ("single", "single_persona"):
        test_id = json.loads((MANIFESTS / f"eval_128_{scenario}.json").read_text())["tasks"][0]["task_id"]
        train_id = json.loads((MANIFESTS / f"train_{scenario}.json").read_text())["tasks"][0]["task_id"]
        code, body = request("/reset", {"scenario": scenario, "task_id": test_id})
        session = body.get("session_id")
        try:
            if code != 200 or not session or body.get("task_id") != test_id or body.get("scenario") != scenario:
                raise RuntimeError(f"TEST reset failed: HTTP {code}: {body}")
        finally:
            if session:
                release_code, released = request("/release", {"session_id": session})
                if release_code != 200 or not released.get("released"):
                    raise RuntimeError("TEST session release failed")
        for rejected in (train_id, "__unknown_eval_gate_id__"):
            code, body = request("/reset", {"scenario": scenario, "task_id": rejected})
            if body.get("session_id"):
                request("/release", {"session_id": body["session_id"]})
            if code != 400 or body.get("error") != "task_id is not in frozen TEST manifest":
                raise RuntimeError(f"admission rejection failed: HTTP {code}: {body}")
        print(json.dumps({"scenario": scenario, "test_id": test_id, "train_id": train_id,
                          "test_reset_release": "PASS", "train_unknown_rejection": "PASS"}))
    print(json.dumps(health()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "stop", "check"))
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "shop_env_test_5200.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"start": start, "stop": stop, "check": check}[args.command]()


if __name__ == "__main__":
    main()

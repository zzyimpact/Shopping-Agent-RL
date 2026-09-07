"""Teacher collection 的 SQLite ledger 与原子 JSON artifact。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
from typing import Any, Mapping
import uuid
from rollout.progress import ProgressLogger


IMMUTABLE_FIELDS = (
    "scenario", "teacher_model", "api_style", "reasoning_effort",
    "system_prompt_hash", "collection_config_hash",
    "shopsim_source_fingerprint", "environment_version",
    "reward_deviation_version",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """先 fsync 临时文件，再以 os.replace 原子发布 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


class ResumeConfigMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class RunPaths:
    root: Path
    manifest: Path
    database: Path
    accepted: Path
    trajectories: Path
    attempts: Path
    logs: Path


class TeacherLedger:
    """单个 run 的持久化账本；不会把 partial attempt 当作 accepted。"""

    def __init__(self, data_root: str | Path, manifest: Mapping[str, Any], *, resume: bool = False,
                 extra_immutable_fields: tuple[str, ...] = (), profile: bool = False):
        required = {"run_id", *IMMUTABLE_FIELDS}
        missing = required - manifest.keys()
        if missing:
            raise ValueError(f"run manifest 缺少字段: {', '.join(sorted(missing))}")
        scenario = str(manifest["scenario"])
        run_id = str(manifest["run_id"])
        root = Path(data_root) / scenario / run_id
        self.extra_immutable_fields = tuple(extra_immutable_fields)
        required.update(self.extra_immutable_fields)
        missing = required - manifest.keys()
        if missing:
            raise ValueError(f"run manifest 缺少字段: {', '.join(sorted(missing))}")
        self.paths = RunPaths(
            root=root, manifest=root / "run_manifest.json", database=root / "state.sqlite",
            accepted=root / "accepted", trajectories=root / "trajectories",
            attempts=root / "attempts", logs=root / "logs",
        )
        self.profile = profile
        artifact_directory = self.paths.trajectories if profile else self.paths.accepted
        for directory in (artifact_directory, self.paths.attempts, self.paths.logs):
            directory.mkdir(parents=True, exist_ok=True)
        desired = dict(manifest)
        desired.setdefault("created_at", utc_now())
        hash_fields = (*IMMUTABLE_FIELDS, *self.extra_immutable_fields)
        desired["immutable_config_hash"] = canonical_hash({k: desired[k] for k in hash_fields})
        if self.paths.manifest.exists():
            if not resume:
                raise FileExistsError(f"run 已存在；请使用 --resume: {root}")
            existing = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
            if existing.get("status") == "invalidated_by_implementation_bug":
                raise ResumeConfigMismatch(
                    "拒绝 resume：该 run 已被标记 invalidated_by_implementation_bug；请创建新的 run_id"
                )
            compared_fields = (*IMMUTABLE_FIELDS, *self.extra_immutable_fields)
            mismatched = [k for k in compared_fields if existing.get(k) != desired.get(k)]
            if mismatched:
                raise ResumeConfigMismatch(
                    "拒绝 resume：immutable config 不一致: " + ", ".join(mismatched)
                )
            self.manifest = existing
        else:
            if resume:
                raise FileNotFoundError(f"resume run 不存在: {root}")
            atomic_json(self.paths.manifest, desired)
            self.manifest = desired
        self.db = sqlite3.connect(self.paths.database)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS attempts (
              attempt_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, scenario TEXT NOT NULL,
              status TEXT NOT NULL, artifact_path TEXT NOT NULL, started_at TEXT NOT NULL,
              finished_at TEXT, teacher_attempt INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS accepted (
              accepted_id TEXT PRIMARY KEY, attempt_id TEXT UNIQUE NOT NULL, task_id TEXT NOT NULL,
              artifact_path TEXT NOT NULL, accepted_at TEXT NOT NULL,
              FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id)
            );
            CREATE TABLE IF NOT EXISTS run_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        self.db.commit()
        self.closed = False

    def start_attempt(self, task_id: str, *, teacher_attempt: int = 1) -> str:
        attempt_id = uuid.uuid4().hex
        path = self.paths.attempts / f"{attempt_id}.json"
        record = {
            "attempt_id": attempt_id, "task_id": task_id,
            "scenario": self.manifest["scenario"], "status": "in_progress",
            "teacher_attempt": teacher_attempt, "started_at": utc_now(), "events": [],
        }
        atomic_json(path, record)
        with self.db:
            self.db.execute(
                "INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (attempt_id, task_id, self.manifest["scenario"], "in_progress",
                 str(path.relative_to(self.paths.root)), record["started_at"], None, teacher_attempt),
            )
        return attempt_id

    def save_attempt(self, attempt_id: str, record: Mapping[str, Any], *, status: str) -> Path:
        row = self.db.execute(
            "SELECT task_id, started_at, artifact_path, teacher_attempt FROM attempts WHERE attempt_id=?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        task_id, started_at, relative_path, teacher_attempt = row
        path = self.paths.root / relative_path
        complete = dict(record)
        complete.update({
            "attempt_id": attempt_id, "task_id": task_id,
            "scenario": self.manifest["scenario"], "status": status,
            "teacher_attempt": teacher_attempt, "started_at": started_at,
            "finished_at": utc_now(),
        })
        atomic_json(path, complete)
        with self.db:
            self.db.execute(
                "UPDATE attempts SET status=?, finished_at=? WHERE attempt_id=?",
                (status, complete["finished_at"], attempt_id),
            )
        return path

    def save_progress(self, attempt_id: str, record: Mapping[str, Any]) -> Path:
        """Durably update an in-progress attempt without marking it finished."""
        row = self.db.execute(
            "SELECT task_id, started_at, artifact_path, teacher_attempt FROM attempts WHERE attempt_id=?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        task_id, started_at, relative_path, teacher_attempt = row
        complete = dict(record)
        complete.update({
            "attempt_id": attempt_id, "task_id": task_id,
            "scenario": self.manifest["scenario"], "status": "in_progress",
            "teacher_attempt": teacher_attempt, "started_at": started_at,
        })
        atomic_json(self.paths.root / relative_path, complete)
        with self.db:
            self.db.execute("UPDATE attempts SET status='in_progress' WHERE attempt_id=?", (attempt_id,))
        return self.paths.root / relative_path

    def accept(self, attempt_id: str, trajectory: Mapping[str, Any]) -> Path:
        """先发布 accepted JSON，随后在一个 SQLite transaction 中标记。"""
        row = self.db.execute(
            "SELECT task_id FROM attempts WHERE attempt_id=?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        accepted_id = uuid.uuid4().hex
        path = self.paths.accepted / f"{accepted_id}.json"
        payload = dict(trajectory)
        payload.update({
            "accepted_id": accepted_id, "attempt_id": attempt_id,
            "task_id": row[0], "scenario": self.manifest["scenario"],
            "accepted_at": utc_now(),
        })
        atomic_json(path, payload)
        with self.db:
            self.db.execute(
                "INSERT INTO accepted VALUES (?, ?, ?, ?, ?)",
                (accepted_id, attempt_id, row[0], str(path.relative_to(self.paths.root)), payload["accepted_at"]),
            )
            self.db.execute(
                "UPDATE attempts SET status='accepted', finished_at=? WHERE attempt_id=?",
                (payload["accepted_at"], attempt_id),
            )
        return path

    def save_trajectory(self, attempt_id: str, trajectory: Mapping[str, Any], *, status: str = "success") -> Path:
        """Persist a profiling trajectory without giving it formal ``accepted`` status."""
        if not self.profile:
            raise RuntimeError("save_trajectory 仅用于 profiling ledger")
        row = self.db.execute(
            "SELECT task_id FROM attempts WHERE attempt_id=?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        path = self.paths.trajectories / f"{attempt_id}.json"
        payload = dict(trajectory)
        payload.update({
            "attempt_id": attempt_id, "task_id": row[0],
            "scenario": self.manifest["scenario"],
            "profiling_status": status, "saved_at": utc_now(),
        })
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已存在 profiling trajectory: {path}")
        atomic_json(path, payload)
        with self.db:
            self.db.execute(
                "UPDATE attempts SET status=?, artifact_path=?, finished_at=? WHERE attempt_id=?",
                (status, str(path.relative_to(self.paths.root)), payload["saved_at"], attempt_id),
            )
        return path

    def completed_task_ids(self) -> set[str]:
        rows = self.db.execute(
            "SELECT DISTINCT task_id FROM attempts WHERE status IN "
            "('success','profile_unsolved','teacher_failure','malformed_action','invalid_action','environment_failure','max_steps')"
        ).fetchall()
        return {str(row[0]) for row in rows}

    def mark_profile_unsolved(self, task_id: str) -> None:
        """Mark the last completed teacher attempt as the profiling terminal outcome."""
        row = self.db.execute(
            "SELECT attempt_id, artifact_path FROM attempts WHERE task_id=? AND status != 'infrastructure_interrupted' "
            "ORDER BY rowid DESC LIMIT 1", (task_id,)
        ).fetchone()
        if row:
            attempt_id, relative_path = row
            # Keep the JSON audit artifact and SQLite ledger in sync.  The
            # summarizer reads artifacts, while resume reads SQLite; updating
            # only one would silently lose the unsolved count in reports.
            path = self.paths.root / relative_path
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record = {"task_id": task_id, "termination_reason": "profile_unsolved", "success": False}
            self.save_attempt(attempt_id, {**record, "termination_reason": "profile_unsolved", "success": False},
                              status="profile_unsolved")

    def mark_infrastructure_interrupted(self, attempt_id: str, record: Mapping[str, Any]) -> Path:
        return self.save_attempt(attempt_id, record, status="infrastructure_interrupted")

    def accepted_count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM accepted").fetchone()[0])

    def set_state(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO run_state(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value)
            )

    def get_state(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM run_state WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else None

    def close(self) -> None:
        if not self.closed:
            self.db.commit()
            self.db.close()
            self.closed = True

    def __enter__(self) -> "TeacherLedger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class GracefulCollectionStop:
    """SIGINT 时保存当前 partial attempt，并阻止继续发新请求。"""

    def __init__(self, ledger: TeacherLedger, resume_command: str, logger: ProgressLogger | None = None):
        self.ledger = ledger
        self.resume_command = resume_command
        self.logger = logger or ProgressLogger()
        self.stop_requested = False
        self.current_attempt_id: str | None = None
        self.current_record: dict[str, Any] = {}
        self._old_handler: Any = None

    def install(self) -> None:
        self._old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handle_sigint)

    def handle_sigint(self, _signum: int, _frame: Any) -> None:
        self.stop_requested = True
        if self.current_attempt_id:
            self.ledger.mark_infrastructure_interrupted(
                self.current_attempt_id,
                {**self.current_record, "interruption": "SIGINT"},
            )
            self.current_attempt_id = None
        self.ledger.set_state("status", "infrastructure_interrupted")
        self.ledger.close()
        self.logger.line("[STOP] Ctrl+C received. Current state preserved.")
        self.logger.line(f"Resume with: {self.resume_command}")

    def close(self) -> None:
        if self._old_handler is not None:
            signal.signal(signal.SIGINT, self._old_handler)
        self.ledger.close()

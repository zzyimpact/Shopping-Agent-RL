"""Bounded synchronous worker primitives for teacher probes/collection.

The environment client and teacher adapter are intentionally synchronous.  A
small thread-pool boundary is enough to overlap expensive API waits while the
remote environment serializes its short mutable operations.  This module has
no collection policy and never knows about a teacher provider.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TypeVar

from rollout.storage import atomic_json


T = TypeVar("T")
R = TypeVar("R")
MAX_WORKERS = 32


class WorkerCancelled(RuntimeError):
    """A task was queued before a global stop and never started."""


@dataclass(frozen=True)
class WorkerResult:
    item: Any
    value: Any = None
    error: BaseException | None = None
    cancelled: bool = False


class GlobalStop:
    """Thread-safe stop signal shared by all workers."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def check_before_external_call(self) -> None:
        if self.is_set():
            raise WorkerCancelled("global stop is set")


def run_bounded(
    items: Iterable[T],
    worker: Callable[[T, GlobalStop], R],
    *,
    workers: int,
    stop: GlobalStop | None = None,
) -> list[WorkerResult]:
    """Run bounded work and set global stop on the first worker exception.

    Every wrapper checks the stop event before invoking ``worker``.  Thus work
    already submitted to the executor may be cancelled logically without
    making a new external request.  In-flight calls are allowed to finish and
    their result/error is returned to the caller.
    """

    if not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    stop = stop or GlobalStop()
    values = list(items)
    results: list[WorkerResult] = []
    lock = threading.Lock()

    def invoke(item: T) -> WorkerResult:
        if stop.is_set():
            return WorkerResult(item=item, cancelled=True,
                                error=WorkerCancelled("global stop is set"))
        try:
            return WorkerResult(item=item, value=worker(item, stop))
        except WorkerCancelled as exc:
            return WorkerResult(item=item, cancelled=True, error=exc)
        except BaseException as exc:  # preserve failure for caller, then stop siblings
            stop.set()
            return WorkerResult(item=item, error=exc)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="teacher-worker") as pool:
        futures: list[Future[WorkerResult]] = [pool.submit(invoke, item) for item in values]
        for future in futures:
            result = future.result()
            with lock:
                results.append(result)
    return results


class ConcurrentArtifactStore:
    """Atomic JSON artifacts with a process-local lock.

    Each artifact path is written through :func:`atomic_json`; the lock keeps
    directory-level bookkeeping deterministic when multiple workers finish at
    once.  It is deliberately independent of the profiling SQLite ledger.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def write(self, relative_path: str, value: Mapping[str, Any]) -> Path:
        target = self.root / relative_path
        with self._lock:
            if target.exists():
                raise FileExistsError(target)
            atomic_json(target, value)
        return target

    def update(self, relative_path: str, value: Mapping[str, Any]) -> Path:
        """Atomically replace an existing per-task snapshot."""
        target = self.root / relative_path
        with self._lock:
            atomic_json(target, value)
        return target

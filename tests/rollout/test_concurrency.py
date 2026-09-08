from __future__ import annotations

import threading
import time

from rollout.concurrency import ConcurrentArtifactStore, GlobalStop, WorkerCancelled, run_bounded


def test_bounded_workers_and_global_stop_preserve_inflight_results():
    started = []
    item0_started = threading.Event()
    item1_started = threading.Event()

    def worker(item, stop):
        stop.check_before_external_call()
        started.append(item)
        if item == 0:
            item0_started.set()
            item1_started.wait(1)
            raise RuntimeError("fixture infrastructure failure")
        if item == 1:
            item1_started.set()
            item0_started.wait(1)
            time.sleep(0.1)
        else:
            time.sleep(0.01)
        return item * 2

    # The first two workers begin; the failure stops queued work before it can
    # make an external call, while the other in-flight item is preserved.
    results = run_bounded(range(6), worker, workers=2)
    assert {row.item for row in results} == set(range(6))
    assert any(row.error and isinstance(row.error, RuntimeError) for row in results)
    assert set(started).issubset({0, 1})
    assert any(row.cancelled for row in results)


def test_concurrent_artifact_store_atomic_updates(tmp_path):
    store = ConcurrentArtifactStore(tmp_path)

    def write(index):
        path = f"tasks/{index}.json"
        store.write(path, {"index": index})
        store.update(path, {"index": index, "done": True})

    run_bounded(range(10), lambda item, _stop: write(item), workers=10)
    assert len(list((tmp_path / "tasks").glob("*.json"))) == 10
    assert all('"done": true' in p.read_text() for p in (tmp_path / "tasks").glob("*.json"))


def test_worker_guard_rejects_invalid_count():
    try:
        run_bounded([], lambda item, stop: None, workers=0)
    except ValueError as exc:
        assert "workers" in str(exc)
    else:
        raise AssertionError("invalid worker count accepted")

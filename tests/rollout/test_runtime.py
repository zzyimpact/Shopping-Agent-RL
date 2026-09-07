from __future__ import annotations

import httpx
import pytest

from rollout.runtime import CollectionInfrastructureInterrupted, generate_or_interrupt
from rollout.storage import TeacherLedger
from rollout.teacher_client import TeacherClient
from tests.rollout.test_storage import manifest


def test_retry_exhaustion_stops_collection_without_accepting(tmp_path):
    def timeout(request):
        raise httpx.ReadTimeout("offline", request=request)

    ledger = TeacherLedger(tmp_path, manifest())
    attempt = ledger.start_attempt("task-1")
    client = TeacherClient(
        api_url="https://relay.invalid", api_key="fixture-key", model="m", max_retries=1,
        client=httpx.Client(transport=httpx.MockTransport(timeout)), sleep=lambda _: None, jitter=lambda: 0.0,
    )
    with pytest.raises(CollectionInfrastructureInterrupted):
        generate_or_interrupt(
            client, [{"role": "user", "content": "x"}], ledger=ledger, attempt_id=attempt,
            partial_record={"events": []}, resume_command="collect --resume run-001",
        )
    client.close()
    with TeacherLedger(tmp_path, manifest(), resume=True) as resumed:
        assert resumed.accepted_count() == 0
        assert resumed.get_state("status") == "infrastructure_interrupted"
        row = resumed.db.execute("SELECT status FROM attempts WHERE attempt_id=?", (attempt,)).fetchone()
        assert row[0] == "infrastructure_interrupted"

"""正式 collector 将复用的基础设施中断边界。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from rollout.progress import ProgressLogger
from rollout.storage import TeacherLedger
from rollout.teacher_client import TeacherClient, TeacherClientError, TeacherResponse


class CollectionInfrastructureInterrupted(RuntimeError):
    """表示整个 collector 应保存状态并停止，而不是消耗 teacher attempt。"""


def generate_or_interrupt(
    client: TeacherClient,
    messages: Sequence[Mapping[str, str]],
    *,
    ledger: TeacherLedger,
    attempt_id: str,
    partial_record: Mapping[str, Any],
    resume_command: str,
    logger: ProgressLogger | None = None,
) -> TeacherResponse:
    """API 基础设施失败耗尽时，持久化 partial 并要求整体 graceful stop。"""
    try:
        return client.generate(messages)
    except TeacherClientError as exc:
        if exc.kind not in {"infrastructure", "provider_protocol_error"}:
            raise
        ledger.mark_infrastructure_interrupted(
            attempt_id,
            {**partial_record, "failure_class": exc.kind, "retries": exc.retries},
        )
        ledger.set_state("status", "infrastructure_interrupted")
        ledger.close()
        (logger or ProgressLogger()).infrastructure_stop(resume_command)
        raise CollectionInfrastructureInterrupted(str(exc)) from exc

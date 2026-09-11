"""Environment HTTP client serialization 与错误处理测试。"""

from __future__ import annotations

import json
import httpx
import pytest

from env.teacher_env_client import TeacherEnvClient, TeacherEnvError


def test_health_reset_and_step_serialization():
    seen = []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        if request.url.path == "/health":
            return httpx.Response(200, request=request, json={"status": "ok"})
        if request.url.path == "/reset":
            return httpx.Response(200, request=request, json={"session_id": "s1", "task_id": "834368861472", "scenario": "single", "environment_version": "task-scoped-v3-multisession", "observation": "start"})
        return httpx.Response(200, request=request, json={"session_id": "s1", "task_id": "834368861472", "scenario": "single", "environment_version": "task-scoped-v3-multisession", "done": False, "observation": "results"})

    raw = httpx.Client(transport=httpx.MockTransport(handler))
    with TeacherEnvClient("http://localhost:5500", client=raw) as client:
        assert client.health().payload["status"] == "ok"
        assert client.reset("single", "834368861472").payload["session_id"] == "s1"
        client.step("s1", "Thought: x\nAction: search[x]")
    assert seen[1] == ("POST", "/reset", {"scenario": "single", "task_id": "834368861472"})
    assert seen[2][2]["response"].endswith("search[x]")


def test_remote_error_classification():
    def handler(request):
        return httpx.Response(409, request=request, json={"error": "expired"})

    raw = httpx.Client(transport=httpx.MockTransport(handler))
    with TeacherEnvClient(client=raw) as client, pytest.raises(TeacherEnvError, match="HTTP 409: expired"):
        client.step("bad", "Action: search[x]")


def test_invalid_action_error_is_classified_without_provider_body():
    def handler(request):
        return httpx.Response(422, request=request, json={"error": "malformed_action"})
    raw = httpx.Client(transport=httpx.MockTransport(handler))
    with TeacherEnvClient(client=raw) as client, pytest.raises(TeacherEnvError) as caught:
        client.step("s", "garbage")
    assert caught.value.kind == "invalid_action"
    assert caught.value.status_code == 422


def test_environment_version_mismatch_is_infrastructure():
    def handler(request):
        return httpx.Response(200, request=request, json={
            "session_id": "s1", "task_id": "t1", "scenario": "single",
            "environment_version": "task-scoped-v2", "observation": "start",
        })

    raw = httpx.Client(transport=httpx.MockTransport(handler))
    with TeacherEnvClient(client=raw) as client, pytest.raises(TeacherEnvError) as caught:
        client.reset("single", "t1")
    assert caught.value.kind == "infrastructure"


@pytest.mark.parametrize("reported_seed", [None, 2, 1])
def test_eval_rng_contract_checked_on_reset_and_mismatch_released(reported_seed):
    from env.evaluation_rng import rng_contract
    calls = []
    def handler(request):
        calls.append(request.url.path)
        body = {"session_id": "s1", "task_id": "t1", "scenario": "single",
                "environment_version": "task-scoped-v3-multisession"}
        if reported_seed is not None:
            body["evaluation_rng"] = rng_contract(reported_seed)
        return httpx.Response(200, request=request, json=body)
    raw = httpx.Client(transport=httpx.MockTransport(handler))
    with TeacherEnvClient(client=raw, expected_evaluation_rng=rng_contract(1)) as client:
        if reported_seed == 1:
            client.reset("single", "t1")
            assert calls == ["/reset"]
        else:
            with pytest.raises(TeacherEnvError, match="RNG contract"):
                client.reset("single", "t1")
            assert calls == ["/reset", "/release"]

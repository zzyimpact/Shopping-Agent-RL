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
            return httpx.Response(200, request=request, json={"session_id": "s1", "observation": "start"})
        return httpx.Response(200, request=request, json={"done": False, "observation": "results"})

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

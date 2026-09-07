"""Teacher API adapter 的 retry/classification 测试；不访问真实 API。"""

from __future__ import annotations

import httpx
import pytest

from rollout.teacher_client import TeacherClient, TeacherClientError


def client_for(handler, **kwargs):
    return TeacherClient(
        api_url="https://relay.invalid/v1/chat/completions", api_key="fixture-key",
        model="relay-model", client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None, jitter=lambda: 0.0, **kwargs,
    )


def test_transient_retry_and_diagnostics():
    calls = []
    retries = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(502, request=request)
        return httpx.Response(
            200, request=request, headers={"x-request-id": "req-1"},
            json={"model": "actual-model", "choices": [{"message": {"content": "你好"}}],
                  "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        )

    with client_for(handler, on_retry=lambda *args: retries.append(args)) as client:
        result = client.generate([{"role": "user", "content": "你好"}])
    assert result.text == "你好"
    assert result.retries == 2
    assert (result.input_tokens, result.output_tokens, result.request_id) == (3, 2, "req-1")
    assert [item[0] for item in retries] == ["HTTP 502", "HTTP 502"]


def test_non_retryable_http_and_secret_redaction():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(401, request=request, text="fixture-provider-body")

    with client_for(handler) as client, pytest.raises(TeacherClientError) as caught:
        client.generate([{"role": "user", "content": "x"}])
    assert calls == 1
    assert caught.value.kind == "config"
    assert caught.value.retries == 0
    assert "fixture-provider-body" not in str(caught.value)
    assert "relay.invalid" not in str(caught.value)


def test_timeout_exhausted_counts_retries():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("fixture timeout", request=request)

    with client_for(handler, max_retries=3) as client, pytest.raises(TeacherClientError) as caught:
        client.generate([{"role": "user", "content": "x"}])
    assert calls == 4
    assert caught.value.kind == "infrastructure"
    assert caught.value.retries == 3
    assert "fixture timeout" not in str(caught.value)


def test_responses_style_parsing():
    def handler(request):
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(
            200, request=request, json={
                "model": "responses-relay",
                "output": [{"content": [{"type": "output_text", "text": "可见输出"}]}],
                "usage": {"input_tokens": 5, "output_tokens": 4},
            },
        )

    with client_for(handler, api_style="responses") as client:
        result = client.generate([{"role": "user", "content": "x"}])
    assert result.text == "可见输出"
    assert result.input_tokens == 5

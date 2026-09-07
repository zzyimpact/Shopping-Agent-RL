"""轻量 teacher relay client。

本模块只负责 API 协议、超时、重试和诊断；不会实现 collection loop，也不会记录
API key 或完整的 provider response。当前支持 OpenAI-compatible Chat Completions
和 Responses 两种请求形态。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import random
import time
from typing import Any, Callable, Mapping, Sequence

import httpx


TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
BACKOFF_SECONDS = (2.0, 5.0, 10.0)


class TeacherClientError(RuntimeError):
    """Teacher request failure with a stable infrastructure/config classification."""

    def __init__(self, message: str, *, kind: str, retryable: bool, status_code: int | None = None,
                 retries: int = 0):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.status_code = status_code
        self.retries = retries


@dataclass(frozen=True)
class TeacherResponse:
    text: str
    model: str | None
    status_code: int
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    retries: int
    api_style: str


def load_env_file(path: str) -> dict[str, str]:
    """读取简单 KEY=VALUE 文件；不会打印值，也不依赖 python-dotenv。"""
    values: dict[str, str] = {}
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def _usage_value(usage: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int):
            return value
    return None


def _content_from_response(data: Mapping[str, Any], style: str) -> str:
    if not isinstance(data, Mapping):
        raise TeacherClientError(
            "teacher provider protocol error", kind="provider_protocol_error", retryable=True
        )
    if style == "chat_completions":
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content") if isinstance(message, Mapping) else None
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    item.get("text", "") for item in content
                    if isinstance(item, Mapping) and isinstance(item.get("text", ""), str)
                )
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output = data.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            for content in item.get("content", []) if isinstance(item, Mapping) else []:
                if isinstance(content, Mapping) and isinstance(content.get("text"), str):
                    chunks.append(content["text"])
        if chunks:
            return "".join(chunks)
    raise TeacherClientError(
        "teacher provider protocol error", kind="provider_protocol_error", retryable=True
    )


class TeacherClient:
    """面向单轮请求的薄 relay adapter。"""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str,
        api_style: str = "chat_completions",
        reasoning_effort: str = "high",
        connect_timeout: float = 15.0,
        read_timeout: float = 300.0,
        max_retries: int = 3,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = lambda: random.uniform(0.0, 0.25),
        on_retry: Callable[[str, int, int, float], None] | None = None,
    ) -> None:
        if api_style not in {"chat_completions", "responses"}:
            raise ValueError("TEACHER_API_STYLE 必须是 chat_completions 或 responses")
        if not api_url or not api_key or not model:
            raise ValueError("TEACHER_API_URL、TEACHER_API_KEY、TEACHER_API_MODEL 均不能为空")
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.api_style = api_style
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self._sleep = sleep
        self._jitter = jitter
        self._on_retry = on_retry
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout)
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "TeacherClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _payload(self, messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        if self.api_style == "responses":
            return {
                "model": self.model,
                "input": list(messages),
                "reasoning": {"effort": self.reasoning_effort},
            }
        return {
            "model": self.model,
            "messages": list(messages),
            "reasoning_effort": self.reasoning_effort,
        }

    def generate(self, messages: Sequence[Mapping[str, str]]) -> TeacherResponse:
        """发送请求并返回 visible output；基础设施失败耗尽后抛出可分类异常。"""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        retries = 0
        started = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(self.api_url, headers=headers, json=self._payload(messages))
            except httpx.TimeoutException as exc:
                if attempt < self.max_retries:
                    retries += 1
                    delay = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)] + self._jitter()
                    if self._on_retry:
                        self._on_retry("timeout", retries, self.max_retries, delay)
                    self._sleep(delay)
                    continue
                raise TeacherClientError(
                    "teacher API timeout after retries", kind="infrastructure", retryable=True, retries=retries
                ) from exc
            except httpx.RequestError as exc:
                if attempt < self.max_retries:
                    retries += 1
                    delay = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)] + self._jitter()
                    if self._on_retry:
                        self._on_retry("connection error", retries, self.max_retries, delay)
                    self._sleep(delay)
                    continue
                raise TeacherClientError(
                    "teacher API unavailable after retries", kind="infrastructure", retryable=True, retries=retries
                ) from exc

            if response.status_code >= 400:
                if response.status_code in TRANSIENT_STATUS and attempt < self.max_retries:
                    retries += 1
                    retry_after = response.headers.get("retry-after")
                    try:
                        delay = float(retry_after) if retry_after else BACKOFF_SECONDS[min(attempt, 2)]
                    except ValueError:
                        try:
                            parsed = parsedate_to_datetime(retry_after)
                            delay = max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
                        except (TypeError, ValueError):
                            delay = BACKOFF_SECONDS[min(attempt, 2)]
                    delay += self._jitter()
                    if self._on_retry:
                        self._on_retry(f"HTTP {response.status_code}", retries, self.max_retries, delay)
                    self._sleep(delay)
                    continue
                kind = "infrastructure" if response.status_code in TRANSIENT_STATUS else "config"
                label = "infrastructure" if kind == "infrastructure" else "request/config"
                raise TeacherClientError(
                    f"teacher API {label} failure (HTTP {response.status_code})",
                    kind=kind, retryable=kind == "infrastructure", status_code=response.status_code, retries=retries
                )

            try:
                data = response.json()
            except (ValueError, json.JSONDecodeError) as exc:
                protocol_error = TeacherClientError(
                    "teacher provider protocol error", kind="provider_protocol_error", retryable=True,
                    retries=retries,
                )
                if attempt < self.max_retries:
                    retries += 1
                    delay = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)] + self._jitter()
                    if self._on_retry:
                        self._on_retry("provider protocol", retries, self.max_retries, delay)
                    self._sleep(delay)
                    continue
                raise TeacherClientError(
                    str(protocol_error), kind="provider_protocol_error", retryable=True, retries=retries
                ) from exc
            try:
                text = _content_from_response(data, self.api_style)
            except TeacherClientError as exc:
                if exc.kind != "provider_protocol_error" or attempt >= self.max_retries:
                    raise TeacherClientError(
                        "teacher provider protocol error", kind="provider_protocol_error", retryable=True,
                        retries=retries,
                    ) from exc
                retries += 1
                delay = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)] + self._jitter()
                if self._on_retry:
                    self._on_retry("provider protocol", retries, self.max_retries, delay)
                self._sleep(delay)
                continue
            if not text.strip():
                raise TeacherClientError(
                    "teacher returned empty visible output", kind="teacher_empty_response",
                    retryable=False, retries=retries,
                )
            usage = data.get("usage") if isinstance(data, Mapping) else {}
            usage = usage if isinstance(usage, Mapping) else {}
            return TeacherResponse(
                text=text,
                model=data.get("model") if isinstance(data.get("model"), str) else None,
                status_code=response.status_code,
                request_id=(
                    response.headers.get("x-request-id")
                    or response.headers.get("request-id")
                    or (data.get("id") if isinstance(data.get("id"), str) else None)
                ),
                input_tokens=_usage_value(usage, "input_tokens", "prompt_tokens"),
                output_tokens=_usage_value(usage, "output_tokens", "completion_tokens"),
                latency_s=time.monotonic() - started,
                retries=retries,
                api_style=self.api_style,
            )
        raise AssertionError("unreachable")

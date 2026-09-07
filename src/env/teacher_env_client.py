"""经 SSH tunnel 访问 remote ShopSimulator teacher service 的轻量客户端。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import time

import httpx


class TeacherEnvError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnvResult:
    payload: dict[str, Any]
    latency_s: float


class TeacherEnvClient:
    def __init__(self, base_url: str = "http://127.0.0.1:5500", *, timeout: float = 60.0,
                 client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "TeacherEnvClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> EnvResult:
        started = time.monotonic()
        try:
            response = self.client.request(method, self.base_url + path, json=payload)
        except httpx.RequestError as exc:
            raise TeacherEnvError("ShopSimulator endpoint unavailable") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("error", "remote error")
            except ValueError:
                detail = "remote error"
            raise TeacherEnvError(f"ShopSimulator HTTP {response.status_code}: {detail}")
        try:
            body = response.json()
        except ValueError as exc:
            raise TeacherEnvError("ShopSimulator returned invalid JSON") from exc
        return EnvResult(body, time.monotonic() - started)

    def health(self) -> EnvResult:
        return self._request("GET", "/health")

    def reset(self, scenario: str, task_id: str) -> EnvResult:
        return self._request("POST", "/reset", {"scenario": scenario, "task_id": task_id})

    def step(self, session_id: str, response: str) -> EnvResult:
        return self._request("POST", "/step", {"session_id": session_id, "response": response})

    def release(self, session_id: str) -> EnvResult:
        return self._request("POST", "/release", {"session_id": session_id})

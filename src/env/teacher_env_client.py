"""经 SSH tunnel 访问 remote ShopSimulator teacher service 的轻量客户端。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import time

import httpx


class TeacherEnvError(RuntimeError):
    def __init__(self, message: str, *, kind: str = "environment", status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


@dataclass(frozen=True)
class EnvResult:
    payload: dict[str, Any]
    latency_s: float


class TeacherEnvClient:
    def __init__(self, base_url: str = "http://127.0.0.1:5500", *, timeout: float = 60.0,
                 client: httpx.Client | None = None,
                 expected_environment_version: str | None = "task-scoped-v3-multisession"):
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)
        self.expected_environment_version = expected_environment_version

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
            raise TeacherEnvError("ShopSimulator endpoint unavailable", kind="infrastructure") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("error", "remote error")
            except ValueError:
                detail = "remote error"
            kind = "invalid_action" if str(detail) in {"invalid_action", "malformed_action"} else "environment"
            raise TeacherEnvError(f"ShopSimulator HTTP {response.status_code}: {detail}", kind=kind,
                                  status_code=response.status_code)
        try:
            body = response.json()
        except ValueError as exc:
            raise TeacherEnvError("ShopSimulator returned invalid JSON", kind="infrastructure") from exc
        if not isinstance(body, dict):
            raise TeacherEnvError("ShopSimulator returned invalid JSON", kind="infrastructure")
        return EnvResult(body, time.monotonic() - started)

    def health(self) -> EnvResult:
        return self._request("GET", "/health")

    def reset(self, scenario: str, task_id: str) -> EnvResult:
        result = self._request("POST", "/reset", {"scenario": scenario, "task_id": task_id})
        if result.payload.get("task_id") != str(task_id) or result.payload.get("scenario") != scenario:
            raise TeacherEnvError("ShopSimulator session identity mismatch after reset", kind="infrastructure")
        if (self.expected_environment_version is not None and
                result.payload.get("environment_version") != self.expected_environment_version):
            raise TeacherEnvError("ShopSimulator environment version mismatch after reset", kind="infrastructure")
        return result

    def step(self, session_id: str, response: str, *, expected_task_id: str | None = None,
             expected_scenario: str | None = None) -> EnvResult:
        result = self._request("POST", "/step", {"session_id": session_id, "response": response})
        if result.payload.get("session_id") != session_id:
            raise TeacherEnvError("ShopSimulator session identity mismatch after step", kind="infrastructure")
        if expected_task_id is not None and result.payload.get("task_id") != str(expected_task_id):
            raise TeacherEnvError("ShopSimulator task identity mismatch after step", kind="infrastructure")
        if expected_scenario is not None and result.payload.get("scenario") != expected_scenario:
            raise TeacherEnvError("ShopSimulator scenario identity mismatch after step", kind="infrastructure")
        if (self.expected_environment_version is not None and
                result.payload.get("environment_version") != self.expected_environment_version):
            raise TeacherEnvError("ShopSimulator environment version mismatch after step", kind="infrastructure")
        return result

    def release(self, session_id: str) -> EnvResult:
        return self._request("POST", "/release", {"session_id": session_id})

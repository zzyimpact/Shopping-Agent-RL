"""Non-paid safety checks for the one-task teacher debug command."""

from __future__ import annotations

import json

import pytest

from scripts.test_teacher_rollout import (
    _available_summary,
    _extract_action_for_display,
    _policy_observation,
    main,
    run_one,
)
from env.teacher_env_client import EnvResult
from rollout.prompt import UPSTREAM_SINGLE_PROMPT, prompt_hash
from rollout.teacher_client import TeacherResponse


def test_policy_observation_never_falls_back_to_raw_observation() -> None:
    assert _policy_observation({"policy_observation": "official"}) == "official"
    assert _policy_observation({"user_message": "wrapped"}) == "wrapped"
    with pytest.raises(ValueError, match="canonical policy observation"):
        _policy_observation({"observation": "raw-only", "instruction": "legacy"})


def test_display_extraction_matches_upstream_shape_without_repairing() -> None:
    assert _extract_action_for_display(
        "Thought: search\nAction: search[枕头]"
    ) == ("search", "枕头", "search[枕头]")
    # A markdown fence is intentionally left malformed; this helper must not
    # broaden the parser just to make a paid smoke appear successful.
    name, argument, raw = _extract_action_for_display(
        "Thought: x\nAction: `search[枕头]`"
    )
    assert name == "`search"
    assert argument == "枕头"
    assert raw == "`search[枕头]`"
    assert _extract_action_for_display("plain text") == (None, None, "plain text")


def test_available_summary_is_compact_and_safe() -> None:
    assert _available_summary({"clickables": ["a", "b"], "has_search_bar": True}) == {
        "clickable_count": 2,
        "has_search_bar": True,
    }
    assert _available_summary({}) == {"clickable_count": "N/A", "has_search_bar": "N/A"}


def test_cli_requires_explicit_task_id_and_never_starts_a_task_list(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--scenario", "single"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "--task-id" in captured.err


def test_run_one_uses_one_explicit_task_and_debug_only_artifact(tmp_path, monkeypatch, capsys) -> None:
    """Exercise the command with fakes; no relay or remote service is used."""

    env_calls = {"reset": 0, "steps": 0, "release": 0}
    client_calls = {"generate": 0}

    class FakeEnv:
        def __init__(self, endpoint, *, timeout):
            assert endpoint == "http://fake"
            assert timeout == 60.0

        def health(self):
            return EnvResult({"status": "ok"}, 0.001)

        def reset(self, scenario, task_id):
            env_calls["reset"] += 1
            assert (scenario, task_id) == ("single", "task-1")
            return EnvResult({
                "session_id": "session-1",
                "policy_observation": "Instruction: buy a cup\n\n搜索功能是否可用: True\n\n可点击的按钮: []",
                "available_actions": {"has_search_bar": True, "clickables": []},
                "policy_context": {
                    "system_prompt": UPSTREAM_SINGLE_PROMPT,
                    "source": "upstream.py",
                    "prompt_hash": prompt_hash(UPSTREAM_SINGLE_PROMPT),
                },
            }, 0.002)

        def step(self, session_id, response):
            env_calls["steps"] += 1
            assert session_id == "session-1"
            assert "Action: search[杯子]" in response
            return EnvResult({
                "action": "search[杯子]",
                "action_valid": True,
                "available_actions": {"has_search_bar": True, "clickables": ["p1"]},
                "policy_observation": "results\n\n搜索功能是否可用: True\n\n可点击的按钮: [\"p1\"]",
                "done": True,
                "reward": 1.0,
                "reward_detail": {"r_succ": 1.0},
            }, 0.003)

        def release(self, session_id):
            env_calls["release"] += 1
            return EnvResult({"released": True}, 0.001)

        def close(self):
            pass

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "fixture-secret"

        def generate(self, messages):
            client_calls["generate"] += 1
            assert messages[-1]["role"] == "user"
            return TeacherResponse(
                "Thought: search\nAction: search[杯子]", "fixture-model", 200,
                "req-1", 3, 2, 0.01, 0, "chat_completions",
            )

        def close(self):
            pass

    monkeypatch.setattr("scripts.test_teacher_rollout.TeacherEnvClient", FakeEnv)
    monkeypatch.setattr("scripts.test_teacher_rollout.TeacherClient", FakeClient)
    env_file = tmp_path / ".env.teacher"
    env_file.write_text(
        "\n".join([
            "TEACHER_API_URL=https://relay.invalid/v1/chat/completions",
            "TEACHER_API_KEY=fixture-secret",
            "TEACHER_API_MODEL=fixture-model",
            "TEACHER_API_STYLE=chat_completions",
            "TEACHER_REASONING_EFFORT=high",
        ]), encoding="utf-8"
    )

    assert run_one(
        scenario="single", task_id="task-1", endpoint="http://fake",
        env_file=env_file, output_root=tmp_path / "debug",
    ) == 0
    assert env_calls == {"reset": 1, "steps": 1, "release": 1}
    assert client_calls == {"generate": 1}
    attempt_files = list((tmp_path / "debug" / "single").glob("*/attempt.json"))
    assert len(attempt_files) == 1
    payload = json.loads(attempt_files[0].read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["task_id"] == "task-1"
    assert "fixture-secret" not in attempt_files[0].read_text(encoding="utf-8")
    assert "ABOUT TO MAKE REAL PAID TEACHER API CALLS" in capsys.readouterr().out

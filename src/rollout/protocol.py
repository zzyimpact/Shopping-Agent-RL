"""Pinned ShopSimulator model-visible observation contract.

The remote teacher service exposes a small amount of diagnostic state (raw
observation and available actions), but the collector must send the same
``instruction`` string that upstream ``single_eval`` sends to its model.  The
functions in this module are intentionally boring and mirror the upstream
formatters byte-for-byte where practical.  Keeping them in one place avoids
having the HTTP service and the local profiler each invent a slightly
different prompt.

This module does *not* parse or repair teacher actions.  Action extraction and
execution remain the upstream environment's responsibility.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def format_available_actions(available_actions: Mapping[str, Any]) -> str:
    """Return upstream ``shop_agent._format_available_actions`` text.

    Upstream copies the clickable list, removes the synthetic ``search``
    entry, and serializes it with ``ensure_ascii=False``.  We deliberately do
    not sort, normalize, or otherwise alter values: ordering is part of the
    environment's visible protocol.
    """

    raw_clickables = available_actions.get("clickables", [])
    if not isinstance(raw_clickables, Sequence) or isinstance(raw_clickables, (str, bytes)):
        raise ValueError("available_actions.clickables must be a sequence")
    clickables = list(raw_clickables)
    if "search" in clickables:
        clickables.remove("search")
    has_search_bar = available_actions.get("has_search_bar", False)
    return (
        f"\n\n搜索功能是否可用: {has_search_bar}"
        f"\n\n可点击的按钮: {json.dumps(clickables, ensure_ascii=False)}"
    )


def _persona_instruction(observation: str, instruction_simple: str | None) -> str:
    """Apply upstream ``single_eval/env.py`` Persona projection.

    ``ShopEnv.interact`` replaces the instruction segment in the state string
    (when the state contains at least two ``[SEP]`` separators) with the
    ``instruction_simple`` value saved at reset.  If the upstream state has no
    such separators it leaves it untouched; reproducing that fallback is
    important for unusual/error observations.
    """

    if not instruction_simple:
        return observation
    parts = observation.split("[SEP]", 2)
    if len(parts) < 3:
        return observation
    return f"{parts[0]}[SEP] {instruction_simple} [SEP]{parts[2]}"


def build_reset_policy_observation(
    instruction: str,
    *,
    available_actions: Mapping[str, Any] | None = None,
    scenario: str = "single",
    instruction_simple: str | None = None,
) -> str:
    """Build the first user message sent by upstream ``Agent.reset``.

    Official reset always advertises a search bar and an empty clickable list;
    callers may pass the observed values for diagnostics, but the default is
    the exact upstream reset contract.  Persona mode uses
    ``instruction_simple`` (the upstream ``ShopEnv.reset`` substitution), not
    the evaluator's full ``instruction_text``.
    """

    if not isinstance(instruction, str):
        raise ValueError("instruction must be text")
    if scenario == "single_persona" and instruction_simple is not None:
        instruction = instruction_simple
    actions = available_actions or {"has_search_bar": True, "clickables": []}
    # single_eval.Agent.reset hard-codes this initial list.  Preserve that
    # behavior unless a caller explicitly asks for a nonstandard diagnostic
    # projection (useful in tests only).
    if available_actions is None:
        actions = {"has_search_bar": True, "clickables": []}
    return instruction + format_available_actions(actions)


def build_step_policy_observation(
    observation: str,
    available_actions: Mapping[str, Any],
    *,
    scenario: str = "single",
    instruction_simple: str | None = None,
) -> str:
    """Build the next user message after one environment interaction."""

    if not isinstance(observation, str):
        observation = str(observation)
    if scenario == "single_persona":
        observation = _persona_instruction(observation, instruction_simple)
    return observation + format_available_actions(available_actions)


def project_persona_instruction(observation: str, instruction_simple: str) -> str:
    """Public testable wrapper for the upstream Persona projection."""

    return _persona_instruction(observation, instruction_simple)


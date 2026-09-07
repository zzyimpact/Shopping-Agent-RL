"""Non-paid regression tests for the upstream model-visible contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rollout.protocol import (
    build_reset_policy_observation,
    build_step_policy_observation,
    format_available_actions,
    project_persona_instruction,
    trace_visible_action,
)


def test_available_actions_formatter_matches_upstream_shape():
    payload = {"has_search_bar": True, "clickables": ["search", "ASIN-1", "红色"]}
    assert format_available_actions(payload) == (
        "\n\n搜索功能是否可用: True\n\n可点击的按钮: "
        + json.dumps(["ASIN-1", "红色"], ensure_ascii=False)
    )


def test_available_actions_formatter_matches_pinned_upstream_function():
    """Compare against the checked-out upstream implementation when present.

    This is intentionally a local, non-network check.  The dependency tree is
    already part of the workspace; importing only ``shop_agent`` avoids
    starting a simulator or loading a catalog.
    """

    upstream_shop_env = Path(__file__).resolve().parents[3] / "ShopSimulator-main" / "shop_env"
    if not upstream_shop_env.exists():
        return
    sys.path.insert(0, str(upstream_shop_env))
    sys.path.insert(0, str(upstream_shop_env / "shop_env"))
    try:
        from shop_agent import _format_available_actions as upstream_format

        payload = {"has_search_bar": False, "clickables": ["search", "A", "蓝色"]}
        assert format_available_actions(payload) == upstream_format(payload)
    finally:
        # Avoid leaking the checkout's module search path into unrelated tests.
        for item in (str(upstream_shop_env), str(upstream_shop_env / "shop_env")):
            while item in sys.path:
                sys.path.remove(item)


def test_reset_policy_observation_matches_single_eval():
    assert build_reset_policy_observation("买一个杯子") == (
        "买一个杯子\n\n搜索功能是否可用: True\n\n可点击的按钮: []"
    )


def test_persona_reset_uses_instruction_simple_not_full_instruction():
    result = build_reset_policy_observation(
        "FULL evaluator instruction", scenario="single_persona",
        instruction_simple="买一个杯子",
    )
    assert result.startswith("买一个杯子")
    assert "FULL evaluator instruction" not in result


def test_step_policy_observation_appends_actions_and_preserves_order():
    result = build_step_policy_observation(
        "page [SEP] prior [SEP] result",
        {"has_search_bar": False, "clickables": ["search", "Buy Now", "red"]},
    )
    assert result == (
        'page [SEP] prior [SEP] result\n\n搜索功能是否可用: False\n\n可点击的按钮: '
        '["Buy Now", "red"]'
    )


def test_persona_step_replaces_only_instruction_segment():
    raw = "old-prefix [SEP] FULL evaluator instruction [SEP] page text"
    projected = project_persona_instruction(raw, "simple instruction")
    assert projected == "old-prefix [SEP] simple instruction [SEP] page text"
    # Upstream leaves malformed/no-separator observations unchanged.
    assert project_persona_instruction("raw page", "simple instruction") == "raw page"


def test_upstream_action_extraction_and_parser_are_not_repaired():
    trace = trace_visible_action("Thought: x\nAction: click[ASIN-1]")
    assert trace["extracted_action"] == "click[ASIN-1]"
    assert trace["action_name"] == "click"
    assert trace["action_argument"] == "ASIN-1"
    assert trace["canonical"] is True
    markdown = trace_visible_action("Thought: x\nAction: `click[ASIN-1]`")
    assert markdown["canonical"] is False
    escaped = trace_visible_action("Thought: x\\nAction: search[杯子]")
    assert escaped["extracted_action"] == "search[杯子]"

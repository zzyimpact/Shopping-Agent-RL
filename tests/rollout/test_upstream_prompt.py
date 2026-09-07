from __future__ import annotations

from pathlib import Path

import pytest

from rollout.upstream_prompt import (
    UpstreamPromptError,
    extract_system_prompt,
    persona_visible_text,
)


def test_extract_literal_block_even_when_other_yaml_is_malformed(tmp_path: Path):
    path = tmp_path / "upstream.yaml"
    path.write_text(
        "agent_config:\n"
        "  source:openai\n"  # historical typo: whole YAML is not valid
        "  system_prompt: |\n"
        "    第一行\n"
        "    Action: click[值]\n"
        "  model_name: {your key}\n",
        encoding="utf-8",
    )
    assert extract_system_prompt(path) == "第一行\nAction: click[值]\n"


def test_extract_rejects_folded_or_ambiguous_blocks(tmp_path: Path):
    folded = tmp_path / "folded.yaml"
    folded.write_text("system_prompt: >\n  folded\n", encoding="utf-8")
    with pytest.raises(UpstreamPromptError, match="folded"):
        extract_system_prompt(folded)
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("system_prompt: |\n  a\nsystem_prompt: |\n  b\n", encoding="utf-8")
    with pytest.raises(UpstreamPromptError, match="exactly one"):
        extract_system_prompt(duplicate)


def test_persona_visible_text_uses_official_phrase_and_strips_reasoning():
    text = persona_visible_text({"偏好": "蓝", "__reasoning__": "private"})
    assert text.startswith("\n用户的个人文档是：")
    assert "private" not in text
    assert '"偏好": "蓝"' in text


def test_persona_visible_text_rejects_evaluator_fields():
    with pytest.raises(UpstreamPromptError, match="evaluator-only"):
        persona_visible_text({"target_asin": "secret-target"})


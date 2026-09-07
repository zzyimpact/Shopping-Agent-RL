"""安全读取 pinned ShopSimulator ``single_eval`` system prompts。

上游 standard YAML 在公开 snapshot 中含有历史性的 YAML typo
(``source:openai``)，因此不能把读取 prompt 绑定到整个 YAML 成功解析。
本模块只解析 ``system_prompt`` 的 literal block，不读取或解释 API key、URL
及其它配置字段；它用于 parity/audit，不会在运行时自动替换项目已冻结的
prompt contract。
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping


class UpstreamPromptError(ValueError):
    """上游 prompt block 缺失、歧义或格式不受支持。"""


_KEY_RE = re.compile(r"^(?P<indent>[ \t]*)system_prompt[ \t]*:[ \t]*(?P<style>[|>][-+0-9]*)?[ \t]*(?:#.*)?$")


def extract_system_prompt(path: str | Path) -> str:
    """Extract a YAML ``system_prompt`` block without loading other fields.

    Literal (``|``) blocks are returned with YAML's clip behavior: trailing
    blank lines are removed and exactly one final newline is retained.  Folded
    (``>``) blocks are rejected deliberately; silently folding a policy prompt
    would make parity claims unsafe.  Tabs in indentation are rejected.
    """

    source = Path(path).read_text(encoding="utf-8")
    lines = source.splitlines()
    matches = [(index, match) for index, line in enumerate(lines)
               if (match := _KEY_RE.match(line))]
    if len(matches) != 1:
        raise UpstreamPromptError(
            f"expected exactly one system_prompt block, found {len(matches)}"
        )
    index, match = matches[0]
    style = match.group("style") or ""
    if style.startswith(">"):
        raise UpstreamPromptError("folded system_prompt (>) is unsupported for parity")
    key_indent = len(match.group("indent"))
    block: list[str] = []
    block_indent: int | None = None
    for line in lines[index + 1:]:
        if "\t" in line[: len(line) - len(line.lstrip(" \t"))]:
            raise UpstreamPromptError("tabs in system_prompt indentation are unsupported")
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if stripped and indent <= key_indent:
            break
        if stripped and block_indent is None:
            block_indent = indent
        block.append(line)
    if block_indent is None:
        raise UpstreamPromptError("system_prompt block is empty")
    if any(line.strip() and len(line) - len(line.lstrip(" ")) < block_indent
           for line in block):
        raise UpstreamPromptError("system_prompt block has inconsistent indentation")
    result_lines = [line[block_indent:] if line.strip() else "" for line in block]
    # YAML literal clip: discard trailing empty physical lines, then one LF.
    while result_lines and result_lines[-1] == "":
        result_lines.pop()
    result = "\n".join(result_lines) + "\n"
    if not result.strip():
        raise UpstreamPromptError("system_prompt block is empty")
    return result


def persona_visible_text(persona: Mapping[str, Any]) -> str:
    """Format the visible Persona suffix used by upstream ``single_eval``.

    ``WebAgentTextEnv.reset`` removes ``__reasoning__`` before returning the
    persona.  Evaluator-only goal/reward fields are rejected instead of being
    silently filtered, so a malformed remote payload cannot leak target data.
    Mapping insertion order is retained, matching ``json.dumps`` upstream.
    """

    if not isinstance(persona, Mapping):
        raise UpstreamPromptError("persona must be a mapping")
    forbidden = {
        "asin", "target_asin", "attribute", "attributes", "options",
        "instruction_options", "pricing", "price", "reward", "reward_detail",
        "goal", "target_product", "target_option", "query",
    }
    leaked = forbidden.intersection(persona.keys())
    if leaked:
        raise UpstreamPromptError("persona contains evaluator-only fields: " + ", ".join(sorted(leaked)))
    visible = {key: value for key, value in persona.items() if key != "__reasoning__"}
    return "\n用户的个人文档是：" + json.dumps(visible, ensure_ascii=False)


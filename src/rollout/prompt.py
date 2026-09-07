"""Upstream ShopSimulator policy prompt and visible conversation assembly.

The profiler does not invent a teacher prompt.  The remote environment returns
the pinned upstream prompt (and its source/hash) during ``reset``; this module
validates that contract and builds a fresh, complete visible message history
for every teacher request.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


UPSTREAM_PROMPT_SOURCE = "shop_env/web_agent_site/envs/web_agent_text_env.py"

# This is the exact PROMPT_TEMPLATE_zh text in the pinned upstream source used
# by the remote service.  It is a test/reference snapshot only; normal runs
# use the text returned by the remote reset policy_context.
UPSTREAM_SINGLE_PROMPT = """你正在进行一次网上购物模拟，目标是从商品库中选购最符合需求的商品。请注意，商品库中存在大量同类商品，你必须通过合理操作，最终购买到最符合要求的目标商品。
购物流程采用多轮对话形式，每一轮我会提供当前页面的观察结果，以及你可执行的操作列表。你需要根据当前状态和可选操作，选择并执行最合适的一步。

操作分为两类：
1. 搜索操作
    格式：search[关键词]
    你可以根据当前需求，自主决定搜索关键词。
    只有在搜索功能可用时，才能使用该操作。
2. 点击操作
    格式：click[值]
    你只能点击当前可用操作列表中的按钮或选项，值必须严格对应操作列表中的内容。

规则说明：
1. 在点击【购买】前，必须至少选择一个商品规格（如颜色、尺码等）。
2. 你的目标是综合所有已知信息，通过搜索、筛选和选择，最终购买到“最符合需求的”商品，而非随意购买任意商品。
3. 如果你给出的操作在当前环境中不可执行，则页面不会有任何变化。
4. 提示：当你点击某一个商品规格时，页面其他信息不会发生变化，但该规格会被选中。请据此判断后续操作。

输出格式
在每一步，你必须按照下面格式输出：
Thought: 简要说明你在当前状态下的思考过程和操作依据。
Action: 用规定格式输出你选择的操作。
"""

UPSTREAM_PERSONA_PROMPT = """你正在进行一次网上购物模拟，目标是从商品库中选购最符合需求的商品。
我会提供他们的目标商品（例如“一双鞋”）以及个人文档（例如偏好、预算、使用场景等）请注意，商品库中存在大量同类商品，你必须通过合理操作，，综合分析用户需求和当前页面信息，最终购买到最符合用户要求的商品。
购物流程采用多轮对话形式，每一轮我会提供当前页面的观察结果，以及你可执行的操作列表。你需要根据当前状态和可选操作，选择并执行最合适的一步。

操作分为两类：
1. 搜索操作
    格式：search[关键词]
    你可以根据当前需求，自主决定搜索关键词。
    只有在搜索功能可用时，才能使用该操作。
2. 点击操作
    格式：click[值]
    你只能点击当前可用操作列表中的按钮或选项，值必须严格对应操作列表中的内容。

规则说明：
1. 在点击【购买】前，必须至少选择一个商品规格（如颜色、尺码等）。
2. 你的目标是综合所有已知信息，通过搜索、筛选和选择，最终购买到“最符合需求的”商品，而非随意购买任意商品。
3. 如果你给出的操作在当前环境中不可执行，则页面不会有任何变化。
4. 提示：当你点击某一个商品规格时，页面其他信息不会发生变化，但该规格会被选中。请据此判断后续操作。

输出格式
在每一步，你必须按照下面格式输出：
Thought: 简要说明你在当前状态下的思考过程和操作依据。
Action: 用规定格式输出你选择的操作。
"""


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PolicyContext:
    scenario: str
    system_prompt: str
    source: str
    source_hash: str
    persona: Mapping[str, Any] | None = None


def policy_context_from_reset(payload: Mapping[str, Any], scenario: str) -> PolicyContext:
    """Read the explicit remote policy contract and reject silent fallbacks."""
    context = payload.get("policy_context")
    if not isinstance(context, Mapping):
        raise ValueError("remote reset missing policy_context; refusing to invent a prompt")
    system_prompt = context.get("system_prompt")
    source = context.get("source")
    source_hash = context.get("prompt_hash")
    if not all(isinstance(v, str) and v for v in (system_prompt, source, source_hash)):
        raise ValueError("remote policy_context has incomplete prompt provenance")
    if prompt_hash(system_prompt) != source_hash:
        raise ValueError("remote policy_context prompt hash mismatch")
    persona = context.get("user_persona")
    if scenario == "single_persona" and not isinstance(persona, Mapping):
        raise ValueError("persona reset missing user_persona")
    if scenario == "single" and persona is not None:
        raise ValueError("single reset unexpectedly exposed persona")
    return PolicyContext(scenario, system_prompt, source, source_hash,
                         dict(persona) if isinstance(persona, Mapping) else None)


def _persona_text(persona: Mapping[str, Any]) -> str:
    # Match upstream's visible persona projection and remove its private
    # reasoning field.  Never include evaluator target/reward fields here.
    clean = {k: v for k, v in persona.items() if k != "__reasoning__"}
    forbidden = {"asin", "target_asin", "attribute", "attributes", "options",
                 "instruction_options", "pricing", "price", "reward", "goal",
                 "target_product", "target_option"}
    if forbidden.intersection(clean):
        raise ValueError("persona contains evaluator-only goal fields")
    # Match ``single_eval/agent.py`` exactly.  Preserve mapping insertion order
    # rather than sorting keys because the serialized visible conversation is
    # part of the policy contract.
    return "\n用户的个人文档是：" + json.dumps(clean, ensure_ascii=False)


def build_initial_messages(context: PolicyContext, observation: str) -> list[dict[str, str]]:
    """Build the first visible request without hidden/server-side history."""
    system = context.system_prompt
    if context.persona is not None:
        system += _persona_text(context.persona)
    return [{"role": "system", "content": system},
            {"role": "user", "content": observation}]


def append_turn(messages: Sequence[Mapping[str, str]], assistant_text: str,
                observation: str) -> list[dict[str, str]]:
    """Return a copied full history for the next request."""
    result = [dict(message) for message in messages]
    result.extend([{"role": "assistant", "content": assistant_text},
                   {"role": "user", "content": observation}])
    return result


def assert_no_evaluator_leakage(messages: Sequence[Mapping[str, str]]) -> None:
    forbidden = ("target_asin", "instruction_options", "reward_detail", "evaluator_only")
    joined = json.dumps(list(messages), ensure_ascii=False).lower()
    for token in forbidden:
        if token.lower() in joined:
            raise AssertionError(f"evaluator-only field leaked into policy messages: {token}")

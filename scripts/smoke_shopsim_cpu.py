#!/usr/bin/env python3
"""使用真实 upstream TextEnv 运行 gold-aware CPU smoke episode。

该脚本只验证 environment/action/reward 链路，不生成或写入训练数据。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def run_episode(env, task_index: int, query: str, asin: str, option_hint: str) -> dict:
    from shop_agent import _extract_action_from_response

    def model_action(action: str):
        # 使用 upstream 的 response extraction，再进入同一 parse_action/step 链。
        response = f"Thought: smoke validation\nAction: {action}"
        return env.step(_extract_action_from_response(response))

    obs, _ = env.reset(idx=task_index)
    assert obs and env.get_available_actions()["has_search_bar"]
    model_action(f"search[{query}]")
    clickable = env.get_available_actions()["clickables"]
    asin_key = asin.lower()
    assert asin_key in clickable, f"search result missing target {asin_key}"
    model_action(f"click[{asin_key}]")
    clickable = env.get_available_actions()["clickables"]
    option = next((x for x in clickable if option_hint in x), None)
    if option is not None:
        model_action(f"click[{option}]")
    status = model_action("click[Buy Now]")[1]
    assert status.get("done") is True
    assert status.get("purchase", {}).get("asin") == asin
    assert status.get("reward_detail", {}).get("query_match") is False
    return {"task_index": task_index, "asin": asin, "query": query, "status": status}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.environ["SHOPSIM_SEARCH_ROOT"] = args.search_root
    os.environ["JAVA_TOOL_OPTIONS"] = os.environ.get(
        "JAVA_TOOL_OPTIONS", "-Xmx64m -XX:MaxDirectMemorySize=64m"
    )
    # 让脚本从项目目录直接执行时也能找到未复制的 upstream checkout。
    upstream_root = os.environ.get("UPSTREAM_ROOT", "/root/ShopSimulator")
    upstream_shop_env = str(Path(upstream_root) / "shop_env")
    upstream_shop_package = str(Path(upstream_shop_env) / "shop_env")
    for path in (upstream_shop_env, upstream_shop_package):
        if path not in sys.path:
            sys.path.insert(0, path)
    import gym
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    examples = [
        (0, "金丝胡桃木 屏风", "834368861472", "金丝胡桃木"),
        (1, "商用 大容量 冷冻", "920921857638", "BD/BC-719DKEM"),
    ]
    results = []
    for persona in (False, True):
        env = WebAgentTextEnv(observation_mode="text", file_path=args.catalog, if_persona=persona)
        for index, query, asin, option in examples:
            results.append({"persona": persona, **run_episode(env, index, query, asin, option)})
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""经 local SSH tunnel 执行固定 TRAIN task environment smoke。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from env.teacher_env_client import TeacherEnvClient, TeacherEnvError


def summary(text: str, limit: int = 360) -> str:
    compact = " ".join(text.split())
    return compact[:limit] + ("..." if len(compact) > limit else "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:5500")
    parser.add_argument("--scenario", choices=("single", "single_persona", "all"), default="all")
    parser.add_argument("--task-id", default="834368861472")
    parser.add_argument("--query", default="金丝胡桃木 屏风")
    parser.add_argument("--option-hint", default="金丝胡桃木")
    args = parser.parse_args()
    scenarios = ("single", "single_persona") if args.scenario == "all" else (args.scenario,)
    try:
        with TeacherEnvClient(args.endpoint) as client:
            health = client.health()
            print(f"Health: {health.payload['status']} ({health.latency_s:.3f}s)")
            for scenario in scenarios:
                reset = client.reset(scenario, args.task_id)
                session = reset.payload["session_id"]
                print(f"[{scenario}] Reset: {args.task_id} ({reset.latency_s:.3f}s)")
                print("Observation:", summary(reset.payload["observation"]))
                search = client.step(session, f"Thought: 检查真实搜索链路\nAction: search[{args.query}]")
                print(f"[{scenario}] Search round trip: {search.latency_s:.3f}s")
                print("Search observation:", summary(search.payload["observation"]))
                clickables = search.payload["available_actions"]["clickables"]
                target = args.task_id.lower()
                if target not in clickables:
                    raise TeacherEnvError(f"{scenario}: target product not present in real search result page")
                product = client.step(session, f"Thought: 打开目标商品\nAction: click[{target}]")
                print(f"[{scenario}] Product click: {product.latency_s:.3f}s")
                product_clickables = product.payload["available_actions"]["clickables"]
                generic = {"buy now", "description", "features", "reviews", "attributes", "back to search", "next >", "< prev"}
                option = next((x for x in product_clickables if args.option_hint.lower() in x.lower()), None)
                if option is None:
                    option = next((x for x in product_clickables if x not in generic and x != target), None)
                if option is None:
                    raise TeacherEnvError(f"{scenario}: no selectable option/SKU in product observation")
                selected = client.step(session, f"Thought: 选择商品规格\nAction: click[{option}]")
                print(f"[{scenario}] Option click '{option}': {selected.latency_s:.3f}s")
                final = client.step(session, "Thought: 完成购买\nAction: click[Buy Now]")
                print(f"[{scenario}] Buy Now: {final.latency_s:.3f}s")
                if not final.payload.get("done"):
                    raise TeacherEnvError(f"{scenario}: scripted purchase did not terminate")
                if final.payload.get("reward_detail", {}).get("query_match") is not False:
                    raise TeacherEnvError(f"{scenario}: expected missing-query query_match=False")
                print(f"[{scenario}] terminal reward={final.payload.get('reward')} query_match=False")
                client.release(session)
    except TeacherEnvError as exc:
        print(f"Environment smoke failed: {exc}", file=sys.stderr)
        return 2
    print("Infrastructure smoke only: PASS（未调用 teacher API，未生成 teacher data）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

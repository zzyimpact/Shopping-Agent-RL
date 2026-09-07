#!/usr/bin/env python3
"""Remote task-scoped ShopSimulator service。

每次 reset 只 materialize 当前 task goal，但商品加载与 Lucene retrieval 始终使用
完整 Catalog-Fine universe。服务仅绑定 127.0.0.1，供 SSH tunnel 使用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from flask import Flask, jsonify, request

app = Flask(__name__)
active = {"env": None, "session_id": None, "task_id": None, "scenario": None}
settings = {}


def manifest_ids(path: Path) -> set[str]:
    return {str(item["task_id"]) for item in json.loads(path.read_text(encoding="utf-8"))["tasks"]}


def close_active() -> None:
    if active["env"] is not None:
        active["env"].close()
    active.update({"env": None, "session_id": None, "task_id": None, "scenario": None})


@app.get("/health")
def health():
    return jsonify({
        "status": "ok", "mode": "task_scoped",
        "catalog": str(settings["catalog"]), "search_root": str(settings["search_root"]),
        "source_fingerprint": settings["source_fingerprint"],
        "runtime_loaded": settings.get("server") is not None,
    })


@app.post("/reset")
def reset():
    body = request.get_json(silent=True) or {}
    scenario, task_id = body.get("scenario"), str(body.get("task_id", ""))
    if scenario not in {"single", "single_persona"}:
        return jsonify({"error": "invalid scenario"}), 400
    if task_id not in settings["train_ids"][scenario]:
        return jsonify({"error": "task_id is not in frozen TRAIN manifest"}), 400
    try:
        close_active()
        import web_agent_site.envs.web_agent_text_env as env_module
        original_get_goals = env_module.get_goals

        def current_task_goal(products, prices, if_persona=False):
            selected = [product for product in products if str(product.get("asin")) == task_id]
            if len(selected) != 1:
                raise RuntimeError(f"task product not unique: {task_id}")
            return original_get_goals(selected, {task_id: prices[task_id]}, if_persona=if_persona)

        if settings.get("server") is None:
            env_module.get_goals = current_task_goal
            try:
                env = env_module.WebAgentTextEnv(
                    observation_mode="text", file_path=str(settings["catalog"]),
                    if_persona=scenario == "single_persona",
                )
            finally:
                env_module.get_goals = original_get_goals
            settings["server"] = env.server
        else:
            server = settings["server"]
            product = server.product_item_dict.get(task_id)
            if product is None:
                return jsonify({"error": "task product absent from Catalog-Fine"}), 500
            server.goals = original_get_goals(
                [product], {task_id: server.product_prices[task_id]},
                if_persona=scenario == "single_persona",
            )
            server.user_sessions.pop(0, None)
            server.user_sessions.pop("0", None)
            env = env_module.WebAgentTextEnv(
                observation_mode="text", server=server,
                if_persona=scenario == "single_persona",
            )
        observation, _ = env.reset(idx=0)
        session_id = uuid.uuid4().hex
        active.update({"env": env, "session_id": session_id, "task_id": task_id, "scenario": scenario})
        system_prompt = str(env.prompt_template)
        policy_context = {
            "system_prompt": system_prompt,
            "source": "shop_env/web_agent_site/envs/web_agent_text_env.py",
            "prompt_hash": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        }
        if scenario == "single_persona":
            persona = getattr(env, "user_persona", None)
            if not isinstance(persona, dict):
                persona = env.server.goals[0].get("user_persona")
            policy_context["user_persona"] = dict(persona or {})
        return jsonify({
            "session_id": session_id, "task_id": task_id, "scenario": scenario,
            "observation": observation, "available_actions": env.get_available_actions(),
            "instruction": env.instruction_text, "policy_context": policy_context,
            "environment_version": "task-scoped-v1", "reward_deviation_version": "query-match-false-v1",
        })
    except Exception as exc:
        close_active()
        app.logger.exception("reset failed")
        return jsonify({"error": f"reset failed: {type(exc).__name__}"}), 500


@app.post("/step")
def step():
    body = request.get_json(silent=True) or {}
    if body.get("session_id") != active["session_id"] or active["env"] is None:
        return jsonify({"error": "unknown or expired session"}), 409
    response = body.get("response")
    if not isinstance(response, str):
        return jsonify({"error": "response must be text"}), 400
    try:
        from shop_agent import _extract_action_from_response
        from web_agent_site.engine.engine import parse_action
        action = _extract_action_from_response(response.replace("\\n", "\n"))
        action_name, action_arg = parse_action(action)
        normalized_name = str(action_name).strip().lower()
        normalized_arg = str(action_arg or "").strip().lower()
        if normalized_name not in {"search", "click"} or not normalized_arg:
            return jsonify({"error": "malformed_action"}), 422
        observation, status, _ = active["env"].step(action)
        result = {
            "session_id": active["session_id"], "action": action, "observation": observation,
            "available_actions": active["env"].get_available_actions(),
            "action_valid": normalized_name == "search" or (normalized_arg != "search" and normalized_arg in active["env"].text_to_clickable),
            **status,
        }
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("step failed")
        return jsonify({"error": f"step failed: {type(exc).__name__}"}), 500


@app.post("/release")
def release():
    body = request.get_json(silent=True) or {}
    if body.get("session_id") == active["session_id"]:
        close_active()
    return jsonify({"released": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="/root/data/shopsim/catalog-fine-runtime.json", type=Path)
    parser.add_argument("--search-root", default="/root/data/shopsim/search_engine", type=Path)
    parser.add_argument("--manifests", default="/root/data/shopsim/manifests", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5100, type=int)
    parser.add_argument("--source-fingerprint", default="unknown")
    args = parser.parse_args()
    upstream_root = Path(os.environ.get("UPSTREAM_ROOT", "/root/ShopSimulator"))
    sys.path[:0] = [str(upstream_root / "shop_env"), str(upstream_root / "shop_env/shop_env")]
    os.environ["SHOPSIM_SEARCH_ROOT"] = str(args.search_root)
    os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Xmx96m -XX:MaxDirectMemorySize=96m")
    settings.update({
        "catalog": args.catalog, "search_root": args.search_root,
        "source_fingerprint": args.source_fingerprint,
        "train_ids": {
            "single": manifest_ids(args.manifests / "train_single.json"),
            "single_persona": manifest_ids(args.manifests / "train_single_persona.json"),
        },
    })
    app.run(host=args.host, port=args.port, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()

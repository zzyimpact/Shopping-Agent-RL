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
import threading
import uuid

from flask import Flask, jsonify, request

# This file is copied to the remote CPU host by teacher_env_up.sh.  Importing
# the small helper is safe there because it has no simulator dependencies;
# keep the upstream parser itself as the execution source of truth below.
try:
    from rollout.protocol import (
        POLICY_OBSERVATION_VERSION,
        PROFILER_PROTOCOL_VERSION,
        build_reset_policy_observation,
        build_step_policy_observation,
    )
except ModuleNotFoundError:  # remote service launched outside project venv
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from rollout.protocol import (  # type: ignore[no-redef]
        POLICY_OBSERVATION_VERSION,
        PROFILER_PROTOCOL_VERSION,
        build_reset_policy_observation,
        build_step_policy_observation,
    )

app = Flask(__name__)
# ``active`` remains as a compatibility shim for the old endpoint unit tests;
# production requests use the isolated ``sessions`` table below.
active = {"env": None, "session_id": None, "task_id": None, "scenario": None}
settings = {}
sessions: dict[str, dict[str, object]] = {}
environment_lock = threading.RLock()
MAX_SESSIONS = 32
ENVIRONMENT_VERSION = "task-scoped-v3-multisession"


def _extract_yaml_system_prompt(path: Path) -> str:
    """Extract only a YAML ``system_prompt: |`` block.

    The pinned standard config has legacy malformed scalar keys (for example
    ``source:openai``), so parsing the whole file with PyYAML is deliberately
    avoided.  This parser is limited to the known block-scalar shape and
    rejects missing/ambiguous blocks instead of inventing a prompt.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    candidates = [i for i, line in enumerate(lines)
                  if line.strip().startswith("system_prompt:") and "|" in line]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one system_prompt block in {path}")
    start = candidates[0]
    key_indent = len(lines[start]) - len(lines[start].lstrip())
    body: list[str] = []
    for line in lines[start + 1:]:
        nonempty = bool(line.strip())
        indent = len(line) - len(line.lstrip())
        if nonempty and indent <= key_indent:
            break
        body.append(line)
    nonempty_indents = [len(line) - len(line.lstrip()) for line in body if line.strip()]
    if not nonempty_indents:
        raise RuntimeError(f"empty system_prompt block in {path}")
    strip_indent = min(nonempty_indents)
    prompt = "\n".join(
        line[strip_indent:] if line.strip() else "" for line in body
    ).rstrip(" \t\n") + "\n"
    if not prompt.strip():
        raise RuntimeError(f"empty system_prompt block in {path}")
    return prompt


def _system_prompt_for(scenario: str) -> tuple[str, str]:
    upstream_root = Path(os.environ.get("UPSTREAM_ROOT", "/root/ShopSimulator"))
    config_name = "persona" if scenario == "single_persona" else "standard"
    path = upstream_root / "single_eval/configs" / config_name / "qwen3_235b.yaml"
    return _extract_yaml_system_prompt(path), str(path)


def _safe_persona(value: Any) -> dict[str, Any]:
    """Keep only policy-visible persona fields; never expose evaluator data."""

    if not isinstance(value, dict):
        return {}
    forbidden = {
        "__reasoning__", "asin", "target_asin", "attribute", "attributes",
        "options", "instruction_options", "pricing", "price", "reward",
        "goal", "target_product", "target_option", "query", "category",
    }
    return {str(key): item for key, item in value.items() if str(key) not in forbidden}


def manifest_ids(path: Path, scenario: str, split: str) -> set[str]:
    tasks = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    if not tasks or any(item.get("scenario") != scenario or item.get("official_split") != split
                        or not item.get("task_id") for item in tasks):
        raise ValueError(f"invalid official split manifest: {path}")
    ids = {str(item["task_id"]) for item in tasks}
    if len(ids) != len(tasks):
        raise ValueError(f"duplicate task IDs: {path}")
    return ids


def load_admission(manifests: Path) -> dict[str, dict[str, set[str]]]:
    # P2 full official pools, never the sampled eval_128 or SFT manifests.
    pools = {split: {scenario: manifest_ids(manifests / f"{split}_{scenario}.json", scenario, split)
                     for scenario in ("single", "single_persona")} for split in ("train", "test")}
    for scenario in ("single", "single_persona"):
        if pools["train"][scenario] & pools["test"][scenario]:
            raise ValueError(f"official train/test overlap: {scenario}")
    return pools


def _memory_metrics() -> dict[str, int | None]:
    rss = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError):
        pass
    cgroup = None
    for path in (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            cgroup = int(path.read_text().strip())
            break
        except (OSError, ValueError):
            continue
    return {"process_rss_bytes": rss, "cgroup_memory_current_bytes": cgroup}


def _goal_slots(goal: object, slot: int) -> list[object | None]:
    goals: list[object | None] = [None] * int(settings.get("max_sessions", MAX_SESSIONS))
    goals[slot] = goal
    return goals


def _bind_server_slot(session: dict[str, object]) -> None:
    server = settings.get("server")
    if server is None:
        raise RuntimeError("shared ShopSimulator server is not initialized")
    server.goals = _goal_slots(session["goal"], int(session["slot"]))


def _close_session(session_id: str, session: dict[str, object]) -> None:
    env = session.get("env")
    try:
        server = settings.get("server")
        slot = int(session["slot"])
        if server is not None:
            sessions_map = getattr(server, "user_sessions", {})
            sessions_map.pop(slot, None)
            sessions_map.pop(str(slot), None)
    finally:
        if env is not None:
            env.close()
        sessions.pop(session_id, None)


def close_active() -> None:
    """Close the legacy shim and all sessions during shutdown/error cleanup."""
    with environment_lock:
        for session_id, session in list(sessions.items()):
            _close_session(session_id, session)
        if active["env"] is not None:
            active["env"].close()
        active.update({"env": None, "session_id": None, "task_id": None, "scenario": None})


@app.get("/health")
def health():
    with environment_lock:
        metrics = _memory_metrics()
        return jsonify({
        "status": "ok", "mode": "task_scoped_multisession",
        "shared_runtime": True,
        "task_split": settings.get("task_split", "train"),
        "catalog": str(settings["catalog"]), "search_root": str(settings["search_root"]),
        "source_fingerprint": settings["source_fingerprint"],
        "runtime_loaded": settings.get("server") is not None,
        "environment_version": ENVIRONMENT_VERSION,
        "policy_observation_version": POLICY_OBSERVATION_VERSION,
        "profiler_protocol_version": PROFILER_PROTOCOL_VERSION,
        "active_sessions": len(sessions), "max_sessions": int(settings.get("max_sessions", MAX_SESSIONS)),
        **metrics,
        })


@app.post("/reset")
def reset():
    body = request.get_json(silent=True) or {}
    scenario, task_id = body.get("scenario"), str(body.get("task_id", ""))
    if scenario not in {"single", "single_persona"}:
        return jsonify({"error": "invalid scenario"}), 400
    task_split = settings.get("task_split", "train")
    if task_id not in settings[f"{task_split}_ids"][scenario]:
        return jsonify({"error": f"task_id is not in frozen {task_split.upper()} manifest"}), 400
    with environment_lock:
        if len(sessions) >= int(settings.get("max_sessions", MAX_SESSIONS)):
            return jsonify({"error": "session capacity exhausted"}), 503
        slot = next((i for i in range(int(settings.get("max_sessions", MAX_SESSIONS)))
                     if all(int(item["slot"]) != i for item in sessions.values())), None)
        if slot is None:
            return jsonify({"error": "session capacity exhausted"}), 503
        try:
            import web_agent_site.envs.web_agent_text_env as env_module
            original_get_goals = settings.get("original_get_goals") or env_module.get_goals

            def current_task_goal(products, prices, if_persona=False):
                selected = [product for product in products if str(product.get("asin")) == task_id]
                if len(selected) != 1:
                    raise RuntimeError(f"task product not unique: {task_id}")
                return original_get_goals(selected, {task_id: prices[task_id]}, if_persona=if_persona)

            server = settings.get("server")
            if server is None:
                env_module.get_goals = current_task_goal
                try:
                    env = env_module.WebAgentTextEnv(
                        observation_mode="text", file_path=str(settings["catalog"]),
                        if_persona=scenario == "single_persona",
                    )
                finally:
                    env_module.get_goals = original_get_goals
                server = env.server
                settings["server"] = server
                settings["original_get_goals"] = original_get_goals
                goal = server.goals[0]
            else:
                product = server.product_item_dict.get(task_id)
                if product is None:
                    return jsonify({"error": "task product absent from Catalog-Fine"}), 500
                goal = original_get_goals(
                    [product], {task_id: server.product_prices[task_id]},
                    if_persona=scenario == "single_persona",
                )[0]
                server.goals = _goal_slots(goal, int(slot))
                server.user_sessions.pop(slot, None)
                server.user_sessions.pop(str(slot), None)
                env = env_module.WebAgentTextEnv(
                    observation_mode="text", server=server,
                    if_persona=scenario == "single_persona",
                )
            session_id = uuid.uuid4().hex
            session = {"env": env, "session_id": session_id, "task_id": task_id,
                       "scenario": scenario, "slot": int(slot), "goal": goal}
            sessions[session_id] = session
            _bind_server_slot(session)
            observation, _ = env.reset(idx=int(slot))
            # single_eval supplies the policy prompt from its scenario YAML.
            # Do not silently substitute the environment template.
            system_prompt, prompt_source = _system_prompt_for(scenario)
            instruction_simple = str(getattr(env, "instruction_simple", ""))
            policy_instruction = (
                instruction_simple if scenario == "single_persona" and instruction_simple
                else str(getattr(env, "instruction_text", ""))
            )
            available_actions = env.get_available_actions()
            policy_observation = build_reset_policy_observation(
                policy_instruction, scenario=scenario,
                instruction_simple=instruction_simple or None,
            )
            policy_context = {
                "system_prompt": system_prompt,
                "source": prompt_source,
                "prompt_hash": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
                "policy_observation_version": POLICY_OBSERVATION_VERSION,
                "profiler_protocol_version": PROFILER_PROTOCOL_VERSION,
            }
            if scenario == "single_persona":
                persona = getattr(env, "user_persona", None)
                if not isinstance(persona, dict):
                    persona = (env.server.goals[int(slot)] or {}).get("user_persona")
                policy_context["user_persona"] = _safe_persona(persona)
            return jsonify({
                "session_id": session_id, "task_id": task_id, "scenario": scenario,
                "internal_slot": int(slot),
                # ``observation`` remains raw diagnostic state for compatibility;
                # only ``policy_observation`` is intended for the teacher.
                "observation": observation, "raw_observation": observation,
                "policy_observation": policy_observation,
                "available_actions": available_actions,
                "instruction": getattr(env, "instruction_text", ""),
                "instruction_simple": instruction_simple,
                "policy_context": policy_context,
                "environment_version": ENVIRONMENT_VERSION,
                "policy_observation_version": POLICY_OBSERVATION_VERSION,
                "profiler_protocol_version": PROFILER_PROTOCOL_VERSION,
                "reward_deviation_version": "query-match-false-v1",
            })
        except Exception as exc:
            if 'session_id' in locals() and session_id in sessions:
                _close_session(session_id, sessions[session_id])
            app.logger.exception("reset failed")
            return jsonify({"error": f"reset failed: {type(exc).__name__}"}), 500


@app.post("/step")
def step():
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id")
    response = body.get("response")
    if not isinstance(response, str):
        return jsonify({"error": "response must be text"}), 400
    with environment_lock:
        # Resolve the session while holding the same lock as release/reset.
        # A lookup performed before the lock could otherwise let a step run on
        # an env that another request has already released.
        session = sessions.get(str(session_id)) if isinstance(session_id, str) else None
        # Compatibility path for the old endpoint unit tests that inject
        # ``active``; production requests always use ``sessions``.
        if session is None and body.get("session_id") == active.get("session_id") and active.get("env") is not None:
            session = {"env": active["env"], "session_id": active["session_id"],
                       "task_id": active.get("task_id"), "scenario": active.get("scenario", "single"),
                       "slot": 0, "goal": getattr(active["env"], "goal", {})}
        if session is None or session.get("env") is None:
            return jsonify({"error": "unknown or expired session"}), 409
        try:
            from shop_agent import _extract_action_from_response
            from web_agent_site.engine.engine import parse_action
            env = session["env"]
            if session_id in sessions:
                _bind_server_slot(session)
            action = _extract_action_from_response(response.replace("\\n", "\n"))
            action_name, action_arg = parse_action(action)
            available_before = env.get_available_actions()
            clickable_before = set(getattr(env, "text_to_clickable", {}) or {})
            action_name = str(action_name)
            action_arg_text = str(action_arg) if action_arg is not None else ""
            normalized_arg = action_arg_text.lower()
            if action_name not in {"search", "click"} or not normalized_arg:
                return jsonify({"error": "malformed_action"}), 422
            action_valid = ((action_name == "search" and bool(action_arg)) or
                            (action_name == "click" and normalized_arg != "search" and normalized_arg in clickable_before))
            observation, status, _ = env.step(action)
            available_after = env.get_available_actions()
            instruction_simple = str(getattr(env, "instruction_simple", ""))
            policy_observation = build_step_policy_observation(
                str(observation), available_after, scenario=str(session["scenario"]),
                instruction_simple=instruction_simple or None,
            )
            return jsonify({
                "session_id": session_id, "task_id": session.get("task_id"),
                "scenario": session.get("scenario"), "internal_slot": session.get("slot"),
                "action": action, "observation": observation, "raw_observation": observation,
                "policy_observation": policy_observation, "available_actions": available_after,
                "pre_step_available_actions": available_before, "post_step_available_actions": available_after,
                "action_valid": bool(action_valid), "policy_observation_version": POLICY_OBSERVATION_VERSION,
                "profiler_protocol_version": PROFILER_PROTOCOL_VERSION,
                "environment_version": ENVIRONMENT_VERSION, **status,
            })
        except Exception as exc:
            app.logger.exception("step failed")
            return jsonify({"error": f"step failed: {type(exc).__name__}"}), 500


@app.post("/release")
def release():
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id")
    with environment_lock:
        if isinstance(session_id, str) and session_id in sessions:
            _close_session(session_id, sessions[session_id])
        elif session_id == active.get("session_id"):
            active.update({"env": None, "session_id": None, "task_id": None, "scenario": None})
    return jsonify({"released": True, "session_id": session_id})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="/root/data/shopsim/catalog-fine-runtime.json", type=Path)
    parser.add_argument("--search-root", default="/root/data/shopsim/search_engine", type=Path)
    parser.add_argument("--manifests", default="/root/data/shopsim/manifests", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5100, type=int)
    parser.add_argument("--task-split", choices=("train", "test"), default="train")
    parser.add_argument("--source-fingerprint", default="unknown")
    parser.add_argument("--max-sessions", default=MAX_SESSIONS, type=int)
    args = parser.parse_args()
    pools = load_admission(args.manifests)
    upstream_root = Path(os.environ.get("UPSTREAM_ROOT", "/root/ShopSimulator"))
    sys.path[:0] = [str(upstream_root / "shop_env"), str(upstream_root / "shop_env/shop_env")]
    os.environ["SHOPSIM_SEARCH_ROOT"] = str(args.search_root)
    os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Xmx96m -XX:MaxDirectMemorySize=96m")
    settings.update({
        "catalog": args.catalog, "search_root": args.search_root,
        "source_fingerprint": args.source_fingerprint,
        "max_sessions": max(1, min(int(args.max_sessions), MAX_SESSIONS)),
        "task_split": args.task_split,
        "train_ids": pools["train"], "test_ids": pools["test"],
    })
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()

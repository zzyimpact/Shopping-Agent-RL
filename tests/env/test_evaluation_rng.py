"""No-model restart tests using the deployed upstream random functions on tiny data.

Set SHOPSIM_RNG_SOURCE to web_agent_site (or its source-only copy). Only selected
function ASTs are compiled: no upstream imports, catalog, Lucene, NLP or weights.
"""

import ast
from collections import defaultdict
from copy import deepcopy
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import sys
import traceback
import types

import pytest

from env.evaluation_rng import local_rng, rng_contract, upstream_rng_scope


def _functions(path, names, module):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)


def _fixture_service(source, base_seed, split):
    for name in ("web_agent_site", "web_agent_site.engine", "web_agent_site.envs"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    engine = types.ModuleType("web_agent_site.engine.engine")
    engine.random, engine.SEARCH_RETURN_N = random, 150
    _functions(Path(source) / "engine/engine.py",
               ["generate_product_prices", "get_top_n_product_from_keywords"], engine)
    engine.parse_action = lambda action: (action.split("[", 1)[0], action.split("[", 1)[1][:-1])
    goal = types.ModuleType("web_agent_site.engine.goal")
    goal.random, goal.math, goal.defaultdict = random, math, defaultdict
    _functions(Path(source) / "engine/goal.py", ["get_price_range_above", "get_goals",
               "get_existed_goals", "get_reward", "get_option_reward"], goal)
    # Non-random NLP/category/attribute logic is out of scope; r_price and the
    # upstream reward formula execute unchanged on the generated goal/price.
    goal.get_type_reward = lambda *args: dict(r_type=1, query_match=True, category_match=True, title_score=1)
    goal.get_attribute_reward = lambda *args: (1, 1)
    env_module = types.ModuleType("web_agent_site.envs.web_agent_text_env")
    env_module.random, env_module.get_goals = random, goal.get_goals
    for module in (engine, goal, env_module):
        sys.modules[module.__name__] = module
    agent = types.ModuleType("shop_agent")
    agent._extract_action_from_response = lambda response: response.split("Action: ", 1)[-1]
    sys.modules[agent.__name__] = agent
    products = [dict(asin=f"t{i}", pricing=[100.0 + i, 300.0 + i], category="category",
                     query="query", title=f"product{i}", user_persona={"style": "simple"},
                     instruction_attributes=["attribute"], instructions=[dict(
                         instruction=f"buy product{i}", instruction_simple=f"buy product{i}",
                         instruction_options={})]) for i in range(160)]

    class FixtureEnv:
        def __init__(self, *, server=None, **kwargs):
            if server is None:
                catalog = deepcopy(products)
                prices = engine.generate_product_prices(catalog)
                server = types.SimpleNamespace(
                    product_item_dict={p["asin"]: p for p in catalog}, product_prices=prices,
                    goals=env_module.get_goals(catalog, prices, if_persona=kwargs.get("if_persona", False)),
                    user_sessions={})
            self.server = server
            self.text_to_clickable = {"buy now": True}

        def reset(self, idx):
            self.idx = idx
            self.goal = self.server.goals[idx]
            self.instruction_text = self.goal["instruction_text"]
            self.instruction_simple = self.goal["instruction_simple"]
            self.user_persona = self.goal["user_persona"]
            self.server.user_sessions[idx] = {"results": []}
            return self.instruction_text, {}

        def get_available_actions(self):
            return {"has_search_bar": True, "clickables": ["Buy Now"]}

        def step(self, action):
            if action.startswith("search["):
                results = engine.get_top_n_product_from_keywords(
                    ["<r>"], None, list(self.server.product_item_dict.values()), self.server.product_item_dict)
                self.server.user_sessions[self.idx]["results"] = [p["asin"] for p in results]
                visible = [(p["asin"], self.server.product_prices[p["asin"]]) for p in results[:3]]
                return json.dumps(visible), {"done": False, "reward": 0.0}, None
            selected = self.server.user_sessions[self.idx]["results"]
            asin = selected[0] if selected else self.goal["asin"]
            reward, detail = goal.get_reward(deepcopy(self.server.product_item_dict[asin]), self.goal,
                                            self.server.product_prices[asin], {}, verbose=True)
            return "done", {"done": True, "reward": reward, "reward_detail": detail}, None

        def close(self):
            pass

    env_module.WebAgentTextEnv = FixtureEnv
    service = importlib.import_module("scripts.remote_teacher_env_service")
    service.settings = {"task_split": split, "formal_eval_seed": base_seed,
                        "catalog": "fixture", "search_root": "fixture", "source_fingerprint": "fixture",
                        "train_ids": dict.fromkeys(("single", "single_persona"), {"train-task"}),
                        "test_ids": dict.fromkeys(("single", "single_persona"), {"t0", "t1", "t2"})}
    service.sessions = {}
    service._system_prompt_for = lambda scenario: ("Shop", "fixture")
    return service


def _serve(connection, source, base_seed, split):
    try:
        service = _fixture_service(source, base_seed, split)
        client = service.app.test_client()
        while True:
            message = connection.recv()
            if message is None:
                break
            method, path, body = message
            before = random.getstate()
            response = client.open(path, method=method, json=body)
            payload = response.get_json()
            session_id = payload.get("session_id")
            session = service.sessions.get(session_id)
            state = None
            if session:
                server = service.settings["server"]
                state = {"goal": deepcopy(session["goal"]), "prices": dict(server.product_prices),
                         "page": deepcopy(server.user_sessions[session["slot"]])}
            connection.send((response.status_code, payload, state, before == random.getstate()))
        service.close_active()
    except BaseException:
        connection.send(traceback.format_exc())
    finally:
        connection.close()


class ProcessService:
    def __init__(self, source, seed=1):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child, str(source), seed, "test"))
        self.process.start()
        child.close()

    def request(self, method, path, body=None):
        self.connection.send((method, path, body))
        assert self.connection.poll(20), "fixture service timeout"
        result = self.connection.recv()
        assert not isinstance(result, str), result
        assert result[3], "TEST request changed process-global random state"
        return result[:3]

    def close(self):
        if self.process.is_alive():
            self.connection.send(None)
        self.process.join(5)
        if self.process.is_alive():
            self.process.terminate()  # Only this test-owned child, never a deployed service.
            self.process.join(5)
            pytest.fail("fixture service did not exit")
        self.connection.close()
        assert self.process.exitcode == 0


@pytest.fixture
def source():
    root = os.environ.get("SHOPSIM_RNG_SOURCE")
    if not root:
        pytest.skip("source-only upstream fixture requires SHOPSIM_RNG_SOURCE")
    path = Path(root)
    for name in ("engine/engine.py", "engine/goal.py"):
        assert (path / name).is_file()
    return path


def test_local_rng_scope_nesting_threads_and_train_fallback():
    from concurrent.futures import ThreadPoolExecutor
    module = types.SimpleNamespace(random=random)
    global_state = random.getstate()
    try:
        with upstream_rng_scope(local_rng(1, "runtime"), [module]):
            first = module.random.random()
            with upstream_rng_scope(local_rng(1, "goal", "single", "t0"), [module]):
                goal = module.random.random()
            second = module.random.random()
        expected = local_rng(1, "runtime")
        assert (first, second) == (expected.random(), expected.random())
        assert goal != first
        assert random.getstate() == global_state
        def draw(task):
            with upstream_rng_scope(local_rng(1, "goal", "single", task), [module]):
                return [module.random.random() for _ in range(3)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(draw, ["t0", "t1"] * 4)) == [draw("t0"), draw("t1")] * 4
        assert draw("t0") != draw("t1")
        # No TEST context: exactly the original global stream, not a new TRAIN seed.
        reference = random.Random()
        reference.setstate(global_state)
        assert module.random.random() == reference.random()
        assert random.getstate() == reference.getstate()
    finally:
        random.setstate(global_state)


def test_runtime_goal_session_restart_and_order_independence(source):
    def reset(service, task, scenario="single"):
        code, body, state = service.request("POST", "/reset", {"scenario": scenario, "task_id": task})
        assert code == 200, body
        assert body["evaluation_rng"] == rng_contract(1)
        return body["session_id"], state

    def search(service, session):
        code, body, state = service.request("POST", "/step", {
            "session_id": session, "response": "Thought: search\nAction: search[<r>]"})
        assert code == 200, body
        return body["policy_observation"], state

    service = ProcessService(source)
    try:
        assert service.request("GET", "/health")[1]["evaluation_rng"] == rng_contract(1)
        a, original = reset(service, "t0")
        b, other = reset(service, "t1")
        duplicate, repeated = reset(service, "t0")
        assert repeated == original
        assert original["goal"]["price_upper"] != other["goal"]["price_upper"]
        first = search(service, a)
        search(service, b)
        assert search(service, duplicate) == first
        second = search(service, a)
        reset(service, "t2")
        assert search(service, duplicate) == second
        _, persona = reset(service, "t0", "single_persona")
    finally:
        service.close()
    service = ProcessService(source)
    try:
        # Different first task exercises the cold/warm runtime goal paths.
        reset(service, "t2")
        a, restarted = reset(service, "t0")
        assert restarted == original
        assert search(service, a) == first
        _, new_persona = reset(service, "t0", "single_persona")
        assert new_persona == persona
        # Compare every generated price against upstream uniform with the exact stream.
        rng = local_rng(1, "runtime")
        assert restarted["prices"] == {f"t{i}": rng.uniform(100.0 + i, 300.0 + i) for i in range(160)}
    finally:
        service.close()
    service = ProcessService(source, seed=2)
    try:
        code, _, state = service.request("POST", "/reset", {"scenario": "single", "task_id": "t0"})
        assert code == 200
        assert state["prices"] != original["prices"]
        assert state["goal"]["price_upper"] != original["goal"]["price_upper"]
    finally:
        service.close()


def test_evaluator_interrupt_service_restart_resume_equivalence(source, tmp_path):
    import httpx
    from env.teacher_env_client import TeacherEnvClient
    from training.eval import evaluate_policy

    class SamplingPolicy:
        last_usage = {"input_tokens": 10, "generated_tokens": 5}

        def generate(self, messages):
            assert "evaluation_rng" not in repr(messages)
            action = "search[<r>]" if len(messages) == 2 else "click[Buy Now]"
            return f"Thought: draw={random.random()}\nAction: {action}"

    def run(service, output, events, *, resume=False, interrupt=False):
        def handler(request):
            body = json.loads(request.content) if request.content else None
            code, payload, state = service.request(request.method, request.url.path, body)
            if state is not None:
                events.append((request.url.path, payload["task_id"], payload["policy_observation"],
                               payload.get("reward_detail"), state))
            if interrupt and request.url.path == "/step" and payload["task_id"] == "t1":
                raise KeyboardInterrupt()
            return httpx.Response(code, json=payload, request=request)
        return evaluate_policy(policy=SamplingPolicy(), scenario="single", task_ids=["t0", "t1", "t2"],
            env_factory=lambda: TeacherEnvClient(client=httpx.Client(transport=httpx.MockTransport(handler)),
                                                expected_evaluation_rng=rng_contract(1)),
            output_dir=output, resume=resume, base_seed=1)

    full, interrupted, resumed = [], [], []
    service = ProcessService(source)
    try:
        result = run(service, tmp_path / "full", full)
    finally:
        service.close()
    service = ProcessService(source)
    try:
        with pytest.raises(KeyboardInterrupt):
            run(service, tmp_path / "resume", interrupted, interrupt=True)
        assert service.request("GET", "/health")[1]["active_sessions"] == 0
    finally:
        service.close()
    service = ProcessService(source)
    try:
        restarted = run(service, tmp_path / "resume", resumed, resume=True)
    finally:
        service.close()
    assert resumed == [event for event in full if event[1] in ("t1", "t2")]
    assert restarted["metrics"] == result["metrics"]
    def rows(path, name):
        return [json.loads(line) for line in (tmp_path / path / name).read_text().splitlines()]
    for key in ("task_id", "manifest_index", "episode_seed", "actions", "reward_metrics"):
        assert [r[key] for r in rows("full", "episodes.jsonl")] == [r[key] for r in rows("resume", "episodes.jsonl")]
    for task in ("t1", "t2"):
        original = [r["visible_response"] for r in rows("full", "responses.jsonl") if r["task_id"] == task]
        retry = [r["visible_response"] for r in rows("resume", "responses.jsonl") if r["task_id"] == task][-2:]
        assert original == retry

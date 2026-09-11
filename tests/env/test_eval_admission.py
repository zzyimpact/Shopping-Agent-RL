"""CPU-only split admission and session regressions; no upstream runtime imports."""

import importlib
import json
import sys
import types

import pytest


@pytest.fixture
def service(monkeypatch, tmp_path):
    pytest.importorskip("flask")
    module = importlib.import_module("scripts.remote_teacher_env_service")
    pools = {}
    for split in ("train", "test"):
        pools[split] = {}
        for scenario in ("single", "single_persona"):
            task_id = f"{split}-{scenario}"
            pools[split][scenario] = {task_id}
            (tmp_path / f"{split}_{scenario}.json").write_text(json.dumps({"tasks": [
                {"task_id": task_id, "official_split": split, "scenario": scenario}]}))
    monkeypatch.setattr(module, "sessions", {})
    monkeypatch.setattr(module, "active", {"env": None, "session_id": None})
    products = {task_id: {"asin": task_id} for pool in pools.values()
                for ids in pool.values() for task_id in ids}
    server = types.SimpleNamespace(product_item_dict=products, product_prices=dict.fromkeys(products, 1),
                                   goals=[], user_sessions={})
    def goals(products, prices, if_persona=False):
        return [{"asin": products[0]["asin"], "user_persona": {"style": "simple"}}]
    class FakeEnv:
        def __init__(self, *, server, **kwargs):
            self.server = server
            self.instruction_text = "shop"
            self.instruction_simple = "simple shop"
            self.text_to_clickable = {"buy now": True}
        def reset(self, idx):
            self.idx = idx
            self.server.user_sessions[idx] = {"page": "initial"}
            return "initial", {}
        def get_available_actions(self):
            return {"has_search_bar": True, "clickables": ["Buy Now"]}
        def step(self, action):
            assert self.server.goals[self.idx]["asin"] in products
            self.server.user_sessions[self.idx]["page"] = "done"
            return "done", {"done": True, "reward": 0.75}, None
        def close(self):
            pass
    fake = types.ModuleType("web_agent_site.envs.web_agent_text_env")
    fake.get_goals, fake.WebAgentTextEnv = goals, FakeEnv
    for name in ("web_agent_site", "web_agent_site.envs"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    from tests.env.test_remote_teacher_env_service import _install_parser_modules
    _install_parser_modules(monkeypatch)
    monkeypatch.setattr(module, "_system_prompt_for", lambda scenario: ("shop prompt", "fixture"))
    monkeypatch.setattr(module, "settings", {"catalog": "fixture", "search_root": "fixture",
                        "source_fingerprint": "fixture", "server": server, "original_get_goals": goals,
                        "train_ids": pools["train"], "test_ids": pools["test"]})
    return module, tmp_path


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("scenario", ["single", "single_persona"])
def test_split_admission_isolation_and_reward(service, split, scenario):
    module, _ = service
    module.settings["task_split"] = split
    client = module.app.test_client()
    assert client.get("/health").json["task_split"] == split
    other = "test" if split == "train" else "train"
    wrong_scenario = "single_persona" if scenario == "single" else "single"
    for task_id in (f"{other}-{scenario}", "unknown", f"{split}-{wrong_scenario}"):
        response = client.post("/reset", json={"scenario": scenario, "task_id": task_id,
                                               "split": split, "task_split": other})
        assert response.status_code == 400
        assert "manifest" in response.json["error"]
    assert client.post("/reset", json={"scenario": "arbitrary", "task_id": "unknown"}).status_code == 400
    ids = []
    for _ in range(2):
        response = client.post("/reset", json={"scenario": scenario, "task_id": f"{split}-{scenario}"})
        assert response.status_code == 200
        ids.append(response.json["session_id"])
    assert ids[0] != ids[1]
    assert module.sessions[ids[0]]["slot"] != module.sessions[ids[1]]["slot"]
    result = client.post("/step", json={"session_id": ids[0], "response": "Thought: buy\nAction: click[Buy Now]"})
    assert result.status_code == 200 and result.json["reward"] == 0.75 and result.json["done"]
    assert result.json["action_valid"] is True
    second_slot = module.sessions[ids[1]]["slot"]
    assert module.settings["server"].user_sessions[second_slot]["page"] == "initial"
    for session in ids:
        assert client.post("/release", json={"session_id": session}).json["released"]
    assert client.get("/health").json["active_sessions"] == 0


def test_default_train_and_cli_rejects_union(service, monkeypatch):
    module, path = service
    monkeypatch.setattr(sys, "argv", ["service", "--manifests", str(path)])
    monkeypatch.setattr(module.app, "run", lambda **kwargs: None)
    module.main()
    assert module.settings["task_split"] == "train"
    monkeypatch.setattr(sys, "argv", ["service", "--task-split", "both"])
    with pytest.raises(SystemExit):
        module.main()


def test_split_manifest_integrity(service):
    module, path = service
    assert module.load_admission(path)["test"]["single"] == {"test-single"}
    target = path / "test_single.json"
    target.write_text(json.dumps({"tasks": [{"task_id": "train-single", "scenario": "single", "official_split": "test"}]}))
    with pytest.raises(ValueError, match="overlap"):
        module.load_admission(path)
    target.write_text(json.dumps({"tasks": [{"task_id": "test-single", "scenario": "single_persona", "official_split": "test"}]}))
    with pytest.raises(ValueError, match="invalid official"):
        module.load_admission(path)


def test_lifecycle_command_and_pid_ownership(tmp_path, monkeypatch):
    from scripts import eval_env
    assert eval_env.service_command()[-6:] == ["--port", "5200", "--task-split", "test", "--manifests", str(eval_env.MANIFESTS)]
    saved = {"pid": 42, "starttime": "123", "argv": eval_env.service_command()}
    path = tmp_path / "pid"
    path.write_text(json.dumps(saved))
    monkeypatch.setattr(eval_env, "PIDFILE", path)
    monkeypatch.setattr(eval_env, "process_identity", lambda pid: saved)
    assert eval_env.owned_identity() == saved
    monkeypatch.setattr(eval_env, "process_identity", lambda pid: {**saved, "starttime": "456"})
    with pytest.raises(RuntimeError, match="ownership mismatch"):
        eval_env.owned_identity()


def test_lifecycle_rejects_missing_pidfd_before_start(monkeypatch):
    from scripts import eval_env
    monkeypatch.delattr(eval_env.os, "pidfd_open", raising=False)
    with pytest.raises(RuntimeError, match="pidfd required"):
        eval_env.require_pidfd()

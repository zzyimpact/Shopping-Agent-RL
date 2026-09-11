"""Non-paid regression tests for the remote service action contract."""

from __future__ import annotations

import importlib
import sys
import types

import pytest


@pytest.fixture()
def service_module(monkeypatch):
    # ``scripts`` is a namespace package in this repository, so importing the
    # Flask app does not start its server.  Keep the fixture isolated because
    # the module-level ``active`` session is intentionally tiny.
    # Flask is intentionally a remote-only dependency; local controller
    # installations need not carry it.  Run these endpoint-level tests when
    # the remote runtime dependency is available, otherwise leave the local
    # lightweight test suite green.
    pytest.importorskip("flask")
    service = importlib.import_module("scripts.remote_teacher_env_service")
    service.active.update({"env": None, "session_id": None, "task_id": None, "scenario": None})
    return service


class _MutatingEnv:
    def __init__(self, *, initial_clickables):
        # Upstream get_available_actions lowercases button text keys.
        self.text_to_clickable = {name.lower(): object() for name in initial_clickables}
        self.calls = []

    def get_available_actions(self):
        self.calls.append("available")
        return {
            "has_search_bar": True,
            "clickables": list(self.text_to_clickable),
        }

    def step(self, action):
        self.calls.append(("step", action))
        # Simulate the important state transition: clicking a product leaves
        # the search-result page, so the product is absent post-step.
        self.text_to_clickable = {"Back to Search": object()}
        return "page [SEP] product", {"done": False, "reward": 0.0}, None


def _install_parser_modules(monkeypatch):
    shop_agent = types.ModuleType("shop_agent")
    shop_agent._extract_action_from_response = lambda response: response.split("Action: ", 1)[-1]
    engine = types.ModuleType("web_agent_site.engine.engine")
    engine.parse_action = lambda action: (action.split("[", 1)[0], action.split("[", 1)[1][:-1])
    # Preserve any existing package modules while injecting only the names
    # imported by the endpoint.
    monkeypatch.setitem(sys.modules, "shop_agent", shop_agent)
    monkeypatch.setitem(sys.modules, "web_agent_site.engine.engine", engine)


def test_action_valid_is_evaluated_against_pre_step_clickables(service_module, monkeypatch):
    _install_parser_modules(monkeypatch)
    env = _MutatingEnv(initial_clickables=["asin-1"])
    service_module.active.update({"env": env, "session_id": "s1", "scenario": "single", "task_id": "t"})
    with service_module.app.test_client() as client:
        response = client.post("/step", json={
            "session_id": "s1", "response": "Thought: click\nAction: click[asin-1]",
        })
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["action_valid"] is True
    assert payload["available_actions"]["clickables"] == ["Back to Search"]
    # The endpoint must inspect the legal set before executing the mutating
    # step, not merely trust the post-step set.
    assert env.calls[0] == "available"
    assert env.calls[1] == ("step", "click[asin-1]")


def test_illegal_click_is_false_and_does_not_become_legal_after_step(service_module, monkeypatch):
    _install_parser_modules(monkeypatch)
    env = _MutatingEnv(initial_clickables=["other"])
    service_module.active.update({"env": env, "session_id": "s1", "scenario": "single", "task_id": "t"})
    with service_module.app.test_client() as client:
        response = client.post("/step", json={
            "session_id": "s1", "response": "Thought: click\nAction: click[asin-1]",
        })
    assert response.status_code == 200
    assert response.get_json()["action_valid"] is False


def test_search_is_valid_when_search_bar_is_available(service_module, monkeypatch):
    _install_parser_modules(monkeypatch)
    env = _MutatingEnv(initial_clickables=[])
    service_module.active.update({"env": env, "session_id": "s1", "scenario": "single", "task_id": "t"})
    with service_module.app.test_client() as client:
        response = client.post("/step", json={
            "session_id": "s1", "response": "Thought: search\nAction: search[keyword]",
        })
    assert response.status_code == 200
    assert response.get_json()["action_valid"] is True


@pytest.mark.parametrize("clickable", ["asin-1", "red", "Buy Now"])
def test_valid_product_option_and_buy_now_are_pre_step_valid(service_module, monkeypatch, clickable):
    _install_parser_modules(monkeypatch)
    env = _MutatingEnv(initial_clickables=[clickable])
    service_module.active.update({"env": env, "session_id": "s1", "scenario": "single", "task_id": "t"})
    with service_module.app.test_client() as client:
        response = client.post("/step", json={
            "session_id": "s1", "response": f"Thought: click\nAction: click[{clickable}]",
        })
    assert response.status_code == 200
    assert response.get_json()["action_valid"] is True

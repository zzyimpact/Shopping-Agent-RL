from rollout.diversity import behavior_fingerprint, exact_duplicate, similarity_features
from rollout.prompt import (PolicyContext, UPSTREAM_PERSONA_PROMPT, UPSTREAM_SINGLE_PROMPT,
                             append_turn, assert_no_evaluator_leakage, build_initial_messages,
                             prompt_hash)


def test_upstream_prompt_snapshot_hashes_are_stable():
    assert prompt_hash(UPSTREAM_SINGLE_PROMPT) == "4be63238fe2be77c9dcce1479be92f59d6c52073a264c83a37a401c4e63640b1"
    assert prompt_hash(UPSTREAM_PERSONA_PROMPT) == "cf739cc859b92ce881f1c9d7b1b26c95f9f3e78f0c766956ee3fa80545998963"


def test_persona_injection_and_no_evaluator_leakage():
    context = PolicyContext("single_persona", UPSTREAM_PERSONA_PROMPT, "upstream.py",
                            prompt_hash(UPSTREAM_PERSONA_PROMPT), {"预算": "100", "偏好": "蓝色"})
    messages = build_initial_messages(context, "页面观察")
    assert messages[0]["role"] == "system"
    assert "用户的个人文档是：" in messages[0]["content"]
    assert_no_evaluator_leakage(messages)


def test_full_visible_history_does_not_drop_turns():
    context = PolicyContext("single", UPSTREAM_SINGLE_PROMPT, "upstream.py",
                            prompt_hash(UPSTREAM_SINGLE_PROMPT))
    messages = build_initial_messages(context, "obs-0")
    messages = append_turn(messages, "Thought: x\nAction: search[枕头]", "obs-1")
    messages = append_turn(messages, "Thought: y\nAction: click[p1]", "obs-2")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[-1]["content"] == "obs-2"


def test_behavior_fingerprint_ignores_thought_wording_and_detects_duplicates():
    left = behavior_fingerprint(["Thought: a\nAction: search[枕头]", "Thought: b\nAction: click[ABC12345]"])
    right = behavior_fingerprint(["Thought: completely different\nAction: search[ 枕头 ]", "Action: click[abc12345]"])
    assert exact_duplicate(left, right)
    assert similarity_features(left, right)["provisional_near_duplicate"]

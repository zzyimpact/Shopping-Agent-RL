import hashlib

from scripts.freeze_sft_dataset import direct_target_search, percentile


def test_target_id_exploitation_gate_catches_search_and_persona_id_reasoning():
    base = {"evaluator_only": {"goal": {"asin": "123456"}}, "actions": [], "messages": []}
    assert direct_target_search({**base, "actions": ["search[123456]"]})
    assert direct_target_search({**base, "messages": [
        {"role": "assistant", "content": "商品编号123456与用户ID一致"}]})
    assert not direct_target_search({**base, "actions": ["click[123456]"]})


def test_freeze_stats_are_deterministic_primitives():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert hashlib.sha256(b"train").hexdigest() == hashlib.sha256(b"train").hexdigest()

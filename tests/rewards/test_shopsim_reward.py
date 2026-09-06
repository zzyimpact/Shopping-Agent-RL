"""upstream scorer wrapper 的确定性测试。"""

from __future__ import annotations

import pytest

from rewards.shopsim_reward import score_episode


def product(*, category="家居›枕头›乳胶枕", title="天然乳胶枕", attrs=None):
    return {
        "asin": "p1",
        "query": "枕头",
        "category": category,
        "title": title,
        "Title": title,
        "Attributes": attrs or ["天然", "护颈"],
        "BulletPoints": ["天然护颈乳胶枕"],
        "Description": "天然护颈乳胶枕",
    }


def goal(*, category="家居›枕头›乳胶枕", attrs=None, options=None, price_upper=100.0):
    return {
        "asin": "p1",
        "query": "枕头",
        "category": category,
        "name": "天然乳胶枕",
        "attributes": attrs or ["天然", "护颈"],
        "goal_options": options or ["白色"],
        "price_upper": price_upper,
    }


def test_complete_success_and_endpoints():
    result = score_episode(product(), goal(), price=80.0, options={"颜色": "白色"}, finished=True)
    assert result["r_succ"] == 1.0
    assert result["r_strict"] == 1.0
    assert result["r_finish"] == 1.0
    assert result["r_alpha"] == result["r_strict"]
    loose = score_episode(product(), goal(), price=80.0, options={"颜色": "白色"}, finished=True, alpha=0.0)
    assert loose["r_alpha"] == loose["r_loose"]


def test_category_mismatch():
    mismatched = product(category="服饰›女鞋›单鞋", title="跑鞋")
    mismatched["query"] = "鞋"
    result = score_episode(mismatched, goal(), price=80.0, options={"颜色": "白色"}, finished=True)
    assert result["r_category"] == 0.5
    assert result["r_strict"] < 1.0


def test_attribute_mismatch():
    mismatched = product(attrs=["塑料"])
    mismatched["Title"] = mismatched["title"] = "塑料收纳盒"
    mismatched["BulletPoints"] = ["收纳用品"]
    mismatched["Description"] = "收纳用品"
    result = score_episode(mismatched, goal(), price=80.0, options={"颜色": "白色"}, finished=True)
    assert result["r_attribute"] < 1.0
    assert result["r_succ"] == 0.0


def test_option_mismatch():
    result = score_episode(product(), goal(), price=80.0, options={"颜色": "黑色"}, finished=True)
    assert result["r_option"] == 0.0
    assert result["r_strict"] == 0.0


def test_price_violation():
    result = score_episode(product(), goal(price_upper=50.0), price=80.0, options={"颜色": "白色"}, finished=True)
    assert result["r_price"] == 0.0
    assert result["r_strict"] == 0.0


@pytest.mark.parametrize("finished", [False])
def test_no_purchase_or_unfinished(finished):
    result = score_episode(product(), goal(), price=80.0, options={"颜色": "白色"}, finished=finished)
    assert all(value == 0.0 for value in result.values())


def test_alpha_validation():
    with pytest.raises(ValueError):
        score_episode(product(), goal(), price=80.0, options={"颜色": "白色"}, finished=True, alpha=1.1)


def test_component_and_aggregate_parity_with_upstream():
    from web_agent_site.engine.goal import get_reward

    p, g = product(), goal()
    upstream_loose, detail = get_reward(p.copy(), g.copy(), price=80.0, options={"颜色": "白色"}, verbose=True)
    result = score_episode(p, g, price=80.0, options={"颜色": "白色"}, finished=True, alpha=0.0)
    assert result["r_loose"] == upstream_loose
    assert result["r_category"] == detail["r_type"]
    assert result["r_attribute"] == detail["r_att"]
    assert result["r_option"] == detail["r_option"]
    assert result["r_price"] == detail["r_price"]


def test_missing_query_forces_query_match_false_and_keeps_category_fallback():
    from web_agent_site.engine.goal import get_type_reward

    purchased = product()
    target = goal()
    purchased["query"] = target["query"] = None
    purchased["query_available"] = target["query_available"] = False
    detail = get_type_reward(purchased, target)
    assert detail["query_match"] is False
    assert detail["category_match"] is True
    assert detail["r_type"] == 1.0


def test_empty_query_trap_never_matches():
    from web_agent_site.engine.goal import get_type_reward

    purchased = product()
    target = goal()
    purchased["query"] = target["query"] = ""
    detail = get_type_reward(purchased, target)
    assert detail["query_match"] is False


@pytest.mark.parametrize("left,right,expected", [("枕头", "枕头", True), ("枕头", "鞋", False)])
def test_existing_query_uses_upstream_compatibility_branch(left, right, expected):
    from web_agent_site.engine.goal import get_type_reward

    purchased = product()
    target = goal()
    purchased["query"] = left
    target["query"] = right
    purchased["query_available"] = target["query_available"] = True
    detail = get_type_reward(purchased, target)
    assert detail["query_match"] is expected

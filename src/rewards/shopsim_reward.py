"""ShopSimulator 官方 scorer 的最薄项目包装。

匹配、fuzzy threshold 和 loose 分数全部委托给 upstream ``goal.py``；本模块
只统一指标命名，并按项目/论文定义组合 strict 与 alpha endpoint。
"""

from __future__ import annotations

from typing import Any, Mapping


def _upstream():
    from web_agent_site.engine.goal import get_reward

    return get_reward


def score_episode(
    purchased_product: Mapping[str, Any],
    goal: Mapping[str, Any],
    *,
    price: float,
    options: Mapping[str, Any] | None,
    finished: bool,
    alpha: float = 1.0,
) -> dict[str, float]:
    """返回一个 terminal episode 的 8 个统一指标及 alpha reward。

    ``finished=False`` 表示 timeout、invalid action 或未购买；此时不调用
    upstream purchase scorer，所有 terminal metrics 均为 0。
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须位于 [0, 1]")
    if not finished:
        zeros = {
            "r_loose": 0.0,
            "r_strict": 0.0,
            "r_succ": 0.0,
            "r_finish": 0.0,
            "r_category": 0.0,
            "r_attribute": 0.0,
            "r_option": 0.0,
            "r_price": 0.0,
        }
        zeros["r_alpha"] = 0.0
        return zeros

    _, detail = _upstream()(
        dict(purchased_product),
        dict(goal),
        price=price,
        options=dict(options or {}),
        verbose=True,
    )
    category = float(detail.get("r_type", 0.0))
    attribute = float(detail.get("r_att", 0.0))
    option = float(detail.get("r_option", 0.0))
    price_ok = float(detail.get("r_price", 0.0))
    strict = category * attribute * option * price_ok
    loose = float(
        _upstream()(
            dict(purchased_product),
            dict(goal),
            price=price,
            options=dict(options or {}),
        )
    )
    success = float(strict == 1.0)
    result = {
        "r_loose": loose,
        "r_strict": strict,
        "r_succ": success,
        "r_finish": 1.0,
        "r_category": category,
        "r_attribute": attribute,
        "r_option": option,
        "r_price": price_ok,
    }
    result["r_alpha"] = alpha * strict + (1.0 - alpha) * loose
    return result

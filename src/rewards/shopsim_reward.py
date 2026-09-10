"""ShopSimulator 官方 scorer 的最薄项目包装。

匹配、fuzzy threshold 和 loose 分数全部委托给 upstream ``goal.py``；本模块
只统一指标命名，并按项目/论文定义组合 strict 与 alpha endpoint。
"""

from __future__ import annotations

from typing import Any, Mapping


METRIC_KEYS = (
    "r_loose", "r_strict", "r_succ", "r_finish", "r_category",
    "r_attribute", "r_option", "r_price",
)


def combine_reward(metrics: Mapping[str, Any], *, alpha: float) -> float:
    """Combine already-computed strict/loose metrics without rescoring an episode."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须位于 [0, 1]")
    return alpha * float(metrics.get("r_strict", 0.0) or 0.0) + (1.0 - alpha) * float(
        metrics.get("r_loose", 0.0) or 0.0
    )


def metrics_from_environment(payload: Mapping[str, Any], *, alpha: float = 1.0) -> dict[str, float]:
    """Normalize the remote upstream scorer's terminal payload (profiler semantics).

    No new scoring: loose is returned by ShopSimulator; strict is the same
    product of the four upstream components used by ``score_episode``.
    """
    if not (payload.get("done") or payload.get("over")):
        return {key: 0.0 for key in (*METRIC_KEYS, "r_alpha")}
    detail = payload.get("reward_detail") or {}
    category = float(detail.get("r_category", detail.get("r_type", 0.0)) or 0.0)
    attribute = float(detail.get("r_attribute", detail.get("r_att", 0.0)) or 0.0)
    option = float(detail.get("r_option", 0.0) or 0.0)
    price = float(detail.get("r_price", 0.0) or 0.0)
    strict = category * attribute * option * price
    metrics = dict(zip(METRIC_KEYS, (
        float(payload.get("reward", 0.0) or 0.0), strict, float(strict == 1.0),
        1.0, category, attribute, option, price,
    )))
    metrics["r_alpha"] = combine_reward(metrics, alpha=alpha)
    return metrics


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

"""P3b formal collection policy 的最小 loader、hash 与纯解释辅助。

本模块刻意不包含网络、环境或完整 collector scheduler。P3c 通过这里验证 frozen
policy identity，并复用少量无副作用的边界规则。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from rollout.diversity import normalize_action


DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "teacher"
    / "formal_collection_p3b_v1_1.yaml"
)


class CollectionPolicyError(ValueError):
    """Policy 缺字段、相互矛盾或 hash 不匹配。"""


@dataclass(frozen=True)
class PostSuccessState:
    accepted_demos: int
    attempts_used: int
    remaining_attempts: int


def _at(policy: Mapping[str, Any], *path: str) -> Any:
    current: Any = policy
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise CollectionPolicyError("policy 缺少字段: " + ".".join(path))
        current = current[key]
    return current


def canonical_policy_hash(policy: Mapping[str, Any]) -> str:
    """对排除 ``identity.policy_hash`` 后的 sorted compact JSON 求 SHA-256。"""
    if not isinstance(policy, Mapping):
        raise CollectionPolicyError("policy root 必须是 mapping")
    value = deepcopy(dict(policy))
    identity = value.get("identity")
    if not isinstance(identity, dict):
        raise CollectionPolicyError("policy 缺少 identity mapping")
    identity.pop("policy_hash", None)
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.append(str(key).strip().lower())
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def _walk_strings(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, Mapping):
        for child in value.values():
            values.extend(_walk_strings(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_walk_strings(child))
    elif isinstance(value, str):
        values.append(value)
    return values


def validate_collection_policy(policy: Mapping[str, Any]) -> None:
    """验证受支持的 frozen P3b policy invariants；不执行 collection。"""
    version = _at(policy, "identity", "policy_version")
    if version not in {"p3b-v1", "p3b-v1.1"}:
        raise CollectionPolicyError(f"不支持的 policy version: {version}")
    api_style = "responses" if version == "p3b-v1" else "chat_completions"
    expected = {
        ("identity", "policy_status"): "frozen",
        ("identity", "seed"): 1,
        ("teacher", "model"): "gpt-5.6-sol",
        ("teacher", "api_style"): api_style,
        ("teacher", "reasoning_effort"): "high",
        ("teacher", "request_semantics"): "p3a-compatible",
        ("concurrency", "workers_default"): 8,
        ("concurrency", "workers_immutable_per_run"): True,
        ("targets", "primary_tasks_per_scenario"): 3000,
        ("targets", "accepted_trajectories_total"): 12000,
        ("attempt_budgets", "first_success_genuine_attempts_max"): 2,
        ("attempt_budgets", "second_demo_attempts_max"): 2,
        ("attempt_budgets", "post_first_success_attempts_max"): 3,
        ("passes", "A", "reserve", "cross_stratum_fallback"): False,
        ("passes", "B", "total_attempts_max"): 2,
        ("passes", "C", "opens_new_coverage_or_reserve_search"): False,
        ("quality", "no_progress_loop", "disposition"): "hard_reject",
        ("diversity", "exact_behavioral_duplicate", "disposition"): "hard_reject",
        ("diversity", "provisional_near_duplicate", "disposition"): "diagnostic_only",
        ("infrastructure", "exhausted_action"): "global_graceful_stop",
    }
    for path, wanted in expected.items():
        actual = _at(policy, *path)
        if actual != wanted:
            raise CollectionPolicyError(
                f"policy invariant {'.'.join(path)}={actual!r}, expected {wanted!r}"
            )

    if _at(policy, "scope", "pass_order") != ["A", "B", "C"]:
        raise CollectionPolicyError("pass order 必须固定为 A/B/C")
    if set(_at(policy, "scope", "scenarios")) != {"single", "single_persona"}:
        raise CollectionPolicyError("scenario 必须是 single 与 single_persona")
    if _at(policy, "protocol", "environment_version") != "task-scoped-v3-multisession":
        raise CollectionPolicyError("formal environment 必须使用 multi-session v3")
    max_actions = policy.get("protocol", {}).get("max_action_steps", 30)
    if max_actions != 30:
        raise CollectionPolicyError("max action steps 必须固定为 30")

    workers = _at(policy, "concurrency", "workers_default")
    if not _at(policy, "concurrency", "workers_min") <= workers <= _at(
        policy, "concurrency", "workers_max_guard"
    ):
        raise CollectionPolicyError("default workers 超出 guard")

    accepted = _at(policy, "targets", "accepted_demos_per_task")
    if [accepted.get(key) for key in ("minimum", "target", "maximum")] != [1, 2, 3]:
        raise CollectionPolicyError("accepted demos/task 必须固定为 1/2/3")
    hard_target = _at(policy, "targets", "accepted_trajectories_per_scenario")
    if hard_target != {"target": 6000, "hard": True}:
        raise CollectionPolicyError("每 scenario hard accepted target 必须为 6000")
    if _at(policy, "passes", "A", "genuine_attempts_per_task_max") != _at(
        policy, "attempt_budgets", "first_success_genuine_attempts_max"
    ):
        raise CollectionPolicyError("Pass A attempt cap 与全局 budget 不一致")
    if _at(policy, "passes", "B", "total_attempts_max") > _at(
        policy, "attempt_budgets", "post_first_success_attempts_max"
    ):
        raise CollectionPolicyError("Pass B 不能超过 post-first-success budget")

    if _at(policy, "quality", "no_progress_loop", "consecutive_identical_normalized_actions") != 3:
        raise CollectionPolicyError("no-progress threshold 必须为 3")
    if _at(policy, "quality", "no_progress_loop", "comparison") != "exact":
        raise CollectionPolicyError("no-progress observation comparison 必须是 exact")

    retryable = set(_at(policy, "infrastructure", "retryable_failure_classes"))
    required_retryable = {
        "connection_error", "timeout", "HTTP_408", "HTTP_429", "HTTP_500",
        "HTTP_502", "HTTP_503", "HTTP_504", "provider_protocol_error",
    }
    if retryable != required_retryable:
        raise CollectionPolicyError("infrastructure retryable classes 不完整或有未冻结项")

    forbidden_keys = {
        "api_key", "teacher_api_key", "authorization", "api_url",
        "teacher_api_url", "base_url", "endpoint_url",
    }
    found = forbidden_keys.intersection(_walk_keys(policy))
    if found:
        raise CollectionPolicyError("policy 不得包含 secret/URL 配置字段: " + ", ".join(sorted(found)))
    if any(value.lower().startswith(("http://", "https://")) for value in _walk_strings(policy)):
        raise CollectionPolicyError("policy 不得包含 API/HTTP URL value")

    stored_hash = _at(policy, "identity", "policy_hash")
    computed_hash = canonical_policy_hash(policy)
    if stored_hash != computed_hash:
        raise CollectionPolicyError(
            f"policy hash 不匹配: stored={stored_hash}, computed={computed_hash}"
        )


def load_collection_policy(path: str | Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CollectionPolicyError("policy YAML root 必须是 mapping")
    validate_collection_policy(value)
    return value


def next_coverage_action(
    *, genuine_attempts: int, first_success: bool, same_stratum_reserve_available: bool,
    policy: Mapping[str, Any],
) -> str:
    """解释 Pass A 当前 coverage slot 的下一步，不选择具体 task。"""
    if genuine_attempts < 0:
        raise ValueError("genuine_attempts 不能为负")
    if first_success:
        return "covered"
    cap = int(_at(policy, "attempt_budgets", "first_success_genuine_attempts_max"))
    if genuine_attempts < cap:
        return "retry_current_task"
    return "same_stratum_reserve" if same_stratum_reserve_available else "reserve_exhausted"


def interpret_post_success_attempts(
    outcomes: Sequence[str], policy: Mapping[str, Any], *, initial_accepted: int = 1,
) -> PostSuccessState:
    """解释 bounded B/C candidate outcomes；不负责调度 task 或 pass。"""
    maximum = int(_at(policy, "targets", "accepted_demos_per_task", "maximum"))
    budget = int(_at(policy, "attempt_budgets", "post_first_success_attempts_max"))
    accepted = initial_accepted
    used = 0
    for outcome in outcomes:
        if used >= budget:
            break
        if outcome not in {"accepted", "failure", "exact_duplicate", "hygiene_reject"}:
            raise ValueError(f"未知 post-success outcome: {outcome}")
        used += 1
        if outcome == "accepted" and accepted < maximum:
            accepted += 1
    return PostSuccessState(accepted, used, max(0, budget - used))


def collection_terminal_status(
    *, accepted_trajectories: int, eligible_capacity: int, policy: Mapping[str, Any],
) -> str:
    target = int(_at(policy, "targets", "accepted_trajectories_per_scenario", "target"))
    if accepted_trajectories >= target:
        return "complete"
    if eligible_capacity <= 0:
        return str(_at(policy, "targets", "bounded_capacity_exhausted_status"))
    return "in_progress"


def has_deterministic_no_progress_loop(
    actions: Sequence[Any], policy_visible_observations: Sequence[str], *, threshold: int = 3,
) -> bool:
    """检测 exact no-progress loop。

    observations 可以是每个 action 的 pre-action observation（长度等于 actions），也可以是
    rollout 常见的 initial + 每步 post-action 形式（长度等于 actions + 1）。
    """
    if threshold < 2:
        raise ValueError("threshold 必须至少为 2")
    if len(policy_visible_observations) == len(actions) + 1:
        pre_action = policy_visible_observations[:-1]
    elif len(policy_visible_observations) == len(actions):
        pre_action = policy_visible_observations
    else:
        raise ValueError("policy-visible observations 必须与 actions 对齐")
    normalized = [normalize_action(action) for action in actions]
    for end in range(threshold - 1, len(normalized)):
        start = end - threshold + 1
        if len(set(normalized[start:end + 1])) == 1 and len(set(pre_action[start:end + 1])) == 1:
            return True
    return False

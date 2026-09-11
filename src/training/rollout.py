"""One local-policy episode; shared by evaluation and future online GRPO."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping
import warnings

from env.teacher_env_client import TeacherEnvError
from rewards.shopsim_reward import METRIC_KEYS, metrics_from_environment
from rollout.prompt import (
    append_turn, assert_no_evaluator_leakage, build_initial_messages, policy_context_from_reset,
)
from rollout.protocol import trace_visible_action
from training.policy import GenerationConfig
from training.token_trace import TokenTrace


@dataclass
class RolloutResult:
    task_id: str
    scenario: str
    status: str = "max_steps"
    messages: list[dict[str, str]] = field(default_factory=list)
    visible_responses: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    reward_metrics: dict[str, float] = field(default_factory=dict)
    steps: int = 0
    invalid_action_count: int = 0
    malformed_action_count: int = 0
    generation_count: int = 0
    input_tokens: int | None = None
    generated_tokens: int | None = None
    generation_time_s: float = 0.0
    environment_wait_s: float = 0.0
    wall_time_s: float = 0.0
    token_trace: TokenTrace | None = None
    context_limit_at_step: int | None = None
    final_input_tokens: int | None = None
    remaining_context_tokens: int | None = None

    @property
    def reward(self) -> float:
        return self.reward_metrics["r_alpha"]


def _payload(result: Any) -> Mapping[str, Any]:
    return result.payload if hasattr(result, "payload") else result


def _observation(payload: Mapping[str, Any]) -> str:
    # Same explicit policy-visible fields used by the teacher profiler.
    for key in ("policy_observation", "user_message"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("environment missing policy_observation/user_message")


class AgentRollout:
    """Fresh environment session per trajectory, with no teacher storage/runtime."""

    def __init__(self, *, policy: Any, env_factory: Callable[[], Any], scenario: str,
                 reward_alpha: float = 1.0, max_action_steps: int = 30) -> None:
        if scenario not in {"single", "single_persona"}:
            raise ValueError("unsupported scenario")
        if not 0.0 <= reward_alpha <= 1.0:
            raise ValueError("reward_alpha must be in [0, 1]")
        if max_action_steps < 1:
            raise ValueError("max_action_steps must be positive")
        self.policy, self.env_factory, self.scenario = policy, env_factory, scenario
        self.reward_alpha, self.max_action_steps = reward_alpha, max_action_steps

    def run(self, task_id: str, *, sampling: GenerationConfig | None = None) -> RolloutResult:
        started = time.monotonic()
        episode = RolloutResult(str(task_id), self.scenario)
        if sampling is not None:
            if not sampling.do_sample:
                raise ValueError("GRPO sampling must be stochastic")
            episode.token_trace = TokenTrace()
        episode.reward_metrics = {key: 0.0 for key in (*METRIC_KEYS, "r_alpha")}
        env = self.env_factory()
        session_id = None

        def env_call(name: str, *args: Any, **kwargs: Any) -> Any:
            tick = time.monotonic()
            try:
                return getattr(env, name)(*args, **kwargs)
            finally:
                episode.environment_wait_s += time.monotonic() - tick

        try:
            reset = _payload(env_call("reset", self.scenario, str(task_id)))
            session_id = reset.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("environment reset missing session_id")
            context = policy_context_from_reset(reset, self.scenario)
            episode.messages = build_initial_messages(context, _observation(reset))
            assert_no_evaluator_leakage(episode.messages)
            for _ in range(self.max_action_steps):
                tick = time.monotonic()
                if episode.token_trace is None:
                    response = self.policy.generate(episode.messages)
                else:
                    sample = episode.token_trace.sample_turn(self.policy, episode.messages, sampling)
                    if sample is None:
                        episode.status = "context_limit"
                        break
                    response = sample.text
                episode.generation_time_s += time.monotonic() - tick
                # The generate path exposes hard-window outcomes; GRPO's separate
                # token-trace/sample path keeps its existing semantics.
                generation = (getattr(self.policy, "last_generation", {})
                              if episode.token_trace is None else {})
                episode.generation_count += int(generation.get("model_called", True))
                usage = getattr(self.policy, "last_usage", {})
                for name in ("input_tokens", "generated_tokens"):
                    if usage.get(name) is not None:
                        setattr(episode, name, (getattr(episode, name) or 0) + int(usage[name]))
                if not isinstance(response, str):
                    raise TypeError("policy.generate must return assistant text")
                if generation.get("context_limit"):
                    episode.status = "context_limit"
                    episode.context_limit_at_step = episode.steps + 1  # 1-based attempted action turn.
                    episode.final_input_tokens = generation["input_tokens_before_generation"]
                    episode.remaining_context_tokens = generation["remaining_context_tokens"]
                    if response:
                        episode.visible_responses.append(response)
                        episode.messages.append({"role": "assistant", "content": response})
                    break  # Never parse or send a context-truncated action to ShopEnv.
                episode.visible_responses.append(response)
                trace = trace_visible_action(response)
                if not trace["canonical"]:
                    episode.messages.append({"role": "assistant", "content": response})
                    episode.status = "malformed_action"
                    episode.malformed_action_count += 1
                    break
                try:
                    payload = _payload(env_call("step",
                        session_id, response, expected_task_id=str(task_id),
                        expected_scenario=self.scenario,
                    ))
                except TeacherEnvError as exc:
                    if exc.kind != "invalid_action":
                        raise  # Infrastructure failures are not scored as policy failures.
                    episode.messages.append({"role": "assistant", "content": response})
                    episode.status = "invalid_action"
                    episode.invalid_action_count += 1
                    break
                episode.actions.append(str(payload.get("action", trace["extracted_action"])))
                episode.steps += 1
                episode.messages = append_turn(episode.messages, response, _observation(payload))
                assert_no_evaluator_leakage(episode.messages)
                if payload.get("action_valid") is False:
                    episode.status = "invalid_action"
                    episode.invalid_action_count += 1
                    break
                if payload.get("done") or payload.get("over"):
                    episode.reward_metrics = metrics_from_environment(payload, alpha=self.reward_alpha)
                    episode.status = ("success" if episode.reward_metrics["r_succ"] == 1.0
                                      else "terminal_unsuccessful")
                    break
            return episode
        finally:
            try:
                if session_id:
                    try:
                        env_call("release", session_id)
                    except Exception:
                        warnings.warn("ShopEnv session release failed", RuntimeWarning)
            finally:
                env.close()
                episode.wall_time_s = time.monotonic() - started

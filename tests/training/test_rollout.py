import pytest

from env.teacher_env_client import EnvResult, TeacherEnvError
from rewards.shopsim_reward import combine_reward
from rollout.prompt import UPSTREAM_SINGLE_PROMPT, prompt_hash
from training.rollout import AgentRollout


class FakeEnv:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.released, self.responses = [], []
        self.closed = False

    def reset(self, scenario, task_id):
        self.scenario, self.task_id = scenario, task_id
        context = {"system_prompt": UPSTREAM_SINGLE_PROMPT, "source": "fixture",
                   "prompt_hash": prompt_hash(UPSTREAM_SINGLE_PROMPT)}
        if scenario == "single_persona":
            context["user_persona"] = {"preferences": "blue", "__reasoning__": "PRIVATE"}
        return EnvResult({"session_id": "session-1", "policy_observation": "Instruction: buy shoes",
                          "policy_context": context, "evaluator_only": {"goal": "PRIVATE"}}, 0.0)

    def step(self, session_id, response, *, expected_task_id, expected_scenario):
        assert session_id == "session-1"
        assert (expected_task_id, expected_scenario) == (self.task_id, self.scenario)
        self.responses.append(response)
        step = next(self.steps)
        if isinstance(step, Exception):
            raise step
        return EnvResult(step, 0.0)

    def release(self, session_id):
        self.released.append(session_id)

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.seen = []
        self.last_usage = {"input_tokens": 10, "generated_tokens": 5}

    def generate(self, messages):
        self.seen.append([dict(message) for message in messages])
        return next(self.responses)


def step_payload(observation, *, done=False, success=False, valid=True):
    return {"policy_observation": observation, "action_valid": valid, "done": done,
            "reward": float(success),
            "reward_detail": {"r_type": float(success), "r_att": float(success),
                              "r_option": float(success), "r_price": float(success)},
            "goal": {"asin": "PRIVATE"}}


def test_multistep_history_parser_terminal_cleanup():
    env = FakeEnv([step_payload("page 2"), step_payload("done", done=True, success=True)])
    responses = ["Thought: search\nAction: search[shoes]", "Thought: buy\nAction: click[Buy]"]
    policy = FakePolicy(responses)
    result = AgentRollout(policy=policy, env_factory=lambda: env, scenario="single").run("task-1")
    assert result.status == "success" and result.steps == 2 and result.reward == 1.0
    assert result.actions == ["search[shoes]", "click[Buy]"]
    assert env.responses == responses
    assert policy.seen[1][-2:] == [{"role": "assistant", "content": responses[0]},
                                 {"role": "user", "content": "page 2"}]
    assert "PRIVATE" not in str(result.messages)
    assert (result.input_tokens, result.generated_tokens, result.generation_count) == (20, 10, 2)
    assert env.released == ["session-1"] and env.closed


@pytest.mark.parametrize("response", ["not an action", "", "Thought: x\nAction: click[]"])
def test_malformed_response_no_environment_step(response):
    env = FakeEnv([])
    result = AgentRollout(policy=FakePolicy([response]), env_factory=lambda: env, scenario="single").run("t")
    assert result.status == "malformed_action" and result.malformed_action_count == 1
    assert not env.responses and result.steps == 0 and result.reward == 0
    assert result.messages[-1] == {"role": "assistant", "content": response}
    assert env.closed


@pytest.mark.parametrize("payload", [step_payload("same", valid=False), TeacherEnvError("bad action", kind="invalid_action")])
def test_invalid_action_stops_with_zero_metrics(payload):
    env = FakeEnv([payload])
    result = AgentRollout(policy=FakePolicy(["Thought: x\nAction: click[Nope]"]),
                          env_factory=lambda: env, scenario="single").run("t")
    assert result.status == "invalid_action" and result.invalid_action_count == 1
    assert all(value == 0 for value in result.reward_metrics.values())
    assert env.closed and env.released


def test_max_steps_stops_without_extra_generation():
    env = FakeEnv([step_payload("same")] * 2)
    result = AgentRollout(policy=FakePolicy(["Thought: x\nAction: search[x]"] * 2),
                          env_factory=lambda: env, scenario="single", max_action_steps=2).run("t")
    assert result.status == "max_steps" and result.steps == 2 and result.generation_count == 2
    assert result.reward_metrics["r_finish"] == 0


def test_persona_and_partial_terminal_reward_over_flag():
    terminal = step_payload("done")
    terminal.update(over=True, reward=0.8, reward_detail={"r_type": 1, "r_att": 0.5, "r_option": 1, "r_price": 1})
    env, policy = FakeEnv([terminal]), FakePolicy(["Thought: buy\nAction: click[Buy]"])
    result = AgentRollout(policy=policy, env_factory=lambda: env, scenario="single_persona", reward_alpha=0.5).run("t")
    assert "blue" in policy.seen[0][0]["content"] and "PRIVATE" not in str(policy.seen)
    assert result.status == "terminal_unsuccessful"
    assert result.reward == pytest.approx(0.65)
    assert result.reward_metrics["r_finish"] == 1
    assert combine_reward(result.reward_metrics, alpha=1) == 0.5
    assert combine_reward(result.reward_metrics, alpha=0) == 0.8


def test_infrastructure_failure_propagates_and_releases():
    env = FakeEnv([TeacherEnvError("unavailable", kind="infrastructure")])
    with pytest.raises(TeacherEnvError):
        AgentRollout(policy=FakePolicy(["Thought: x\nAction: search[x]"]),
                     env_factory=lambda: env, scenario="single").run("t")
    assert env.released == ["session-1"] and env.closed

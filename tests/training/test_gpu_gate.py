"""Pure local tests of the real-observation audit, with no model or environment calls."""
from copy import deepcopy
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

audit = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/gpu_preflight.py"))["check_real_trace"]


class Policy:
    tokenizer = SimpleNamespace(eos_token_id=9)

    def prompt_token_ids(self, messages, sampling):
        assert messages == [{"role": "user", "content": "visible instruction"}]
        return [10, 11]

    def observation_token_ids(self, observation, *, previous_token, sampling):
        assert observation == "actual returned page" and previous_token == 9
        return [20, 21, 22]


def fixture():
    return dict(policy=Policy(), sampling=None, prompt_ids=[10, 11],
        completion_ids=[1, 9, 20, 21, 22, 2, 9],
        logprobs=[-.1, -.2, 0., 0., 0., -.3, -.4], mask=[1, 1, 0, 0, 0, 1, 1],
        events=[{"path": "/reset", "messages": [{"role": "user", "content": "visible instruction"}]},
                {"path": "/step", "observation": "actual returned page", "done": False},
                {"path": "/step", "observation": "unused final page", "done": True}])


def test_real_trace_audit_uses_exact_observation_and_omits_final_page():
    result = audit(**fixture())
    assert result["inserted_spans"] == [{"start": 2, "end": 5, "previous_sampled_eos": True}]
    assert result["initial_prompt_exact"] and result["no_trailing_external_tokens"]


@pytest.mark.parametrize("failure", ["prompt_metadata", "wrong_observation", "trailing_env", "no_second_turn", "invalid_mask"])
def test_real_trace_audit_refuses_false_pass(failure):
    data = deepcopy(fixture())
    if failure == "prompt_metadata": data["prompt_ids"].append(999)
    elif failure == "wrong_observation": data["completion_ids"][2] = 999
    elif failure == "trailing_env": data["mask"][-1] = 0
    elif failure == "invalid_mask": data["mask"][2] = 2
    else: data["mask"] = [1] * 7
    with pytest.raises(AssertionError):
        audit(**data)

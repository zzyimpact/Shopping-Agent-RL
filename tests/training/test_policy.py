import pytest
from types import SimpleNamespace

from training.policy import GenerationConfig, QwenPolicy


class Tensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, index):
        return self.rows[index]


class Encoding(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    def __init__(self):
        self.encoding = Encoding(input_ids=Tensor([[10, 11, 12]]), attention_mask=Tensor([[1, 1, 1]]))

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return self.encoding

    def decode(self, tokens, **kwargs):
        assert tokens == [80, 81]  # Never decode the prompt as the assistant response.
        return "Thought: choose\nAction: search[shoes]"


class FakeModel:
    device = "test-device"

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return Tensor([[10, 11, 12, 80, 81]])


def test_injected_policy_template_device_and_completion_only_decode():
    model, tokenizer = FakeModel(), FakeTokenizer()
    policy = QwenPolicy(model=model, tokenizer=tokenizer)
    messages = [{"role": "user", "content": "state"}]
    assert policy.generate(messages).endswith("Action: search[shoes]")
    assert tokenizer.messages == messages
    assert tokenizer.kwargs["add_generation_prompt"] is True
    assert tokenizer.kwargs["enable_thinking"] is False
    assert tokenizer.encoding.device == model.device
    assert model.kwargs["attention_mask"] is tokenizer.encoding["attention_mask"]
    assert "temperature" not in model.kwargs  # Greedy evaluator default.
    assert policy.last_usage == {"input_tokens": 3, "generated_tokens": 2}
    assert policy.last_generation["token_ids"] == [80, 81]
    assert policy.last_generation["raw_text"].startswith("Thought:")
    assert model.kwargs["use_model_defaults"] is False


def test_greedy_neutralizes_artifact_sampling_defaults_without_changing_eos():
    model = FakeModel()
    model.generation_config = SimpleNamespace(do_sample=True, temperature=0.6, top_p=0.95,
                                             top_k=20, eos_token_id=[81, 82], pad_token_id=82)
    policy = QwenPolicy(model=model, tokenizer=FakeTokenizer(), adapter_path="sft-checkpoint")
    policy.generate([{"role": "user", "content": "state"}])
    cfg = model.kwargs["generation_config"]
    assert not cfg.do_sample and (cfg.temperature, cfg.top_p, cfg.top_k) == (1, 1, 50)
    assert cfg.eos_token_id == [81, 82] and cfg.pad_token_id == 82
    assert model.generation_config.do_sample and model.generation_config.temperature == 0.6
    assert not {"temperature", "top_p", "top_k"} & model.kwargs.keys()
    assert policy.last_generation["ended_with_eos"]
    assert policy.tokenizer.kwargs["enable_thinking"] is False


def test_project_native_thinking_cannot_be_reenabled():
    assert GenerationConfig(chat_template_kwargs={}).chat_template_kwargs == {"enable_thinking": False}
    with pytest.raises(ValueError, match="enable_thinking=False"):
        GenerationConfig(chat_template_kwargs={"enable_thinking": True})


def test_mode_b_explicit_sampling_kwargs_and_no_grpo_leakage():
    from training.runtime import load_config, PROJECT_ROOT
    cfg = load_config(PROJECT_ROOT / "configs/training/eval_mode_b_diagnostic.yaml")
    assert cfg["diagnostic_only"] and not cfg["merge_into_formal_results"]
    model, tok = FakeModel(), FakeTokenizer()
    policy = QwenPolicy(model=model, tokenizer=tok, generation=GenerationConfig(**cfg["generation"]))
    policy.generate([{"role": "user", "content": "state"}])
    assert {k:model.kwargs[k] for k in ("temperature", "top_p", "top_k", "min_p")} == {
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
    assert model.kwargs["use_model_defaults"] is False and tok.kwargs["enable_thinking"] is False
    assert model.kwargs["max_new_tokens"] == 512
    assert cfg["max_action_steps"] == 30 and cfg["seed"] == 1
    with pytest.raises(ValueError, match="evaluation-only"):
        policy.sample([10], sampling=policy.generation)


@pytest.mark.parametrize("kwargs", [{"top_k": -1}, {"top_k": 0.5}, {"min_p": 1.1}])
def test_invalid_sampling_filter_config(kwargs):
    with pytest.raises(ValueError):
        GenerationConfig(**kwargs)


def test_context_limit_refuses_silent_history_truncation():
    policy = QwenPolicy(model=FakeModel(), tokenizer=FakeTokenizer(),
                        generation=GenerationConfig(max_context_tokens=4, max_new_tokens=2))
    with pytest.raises(ValueError, match="no silent truncation"):
        policy.generate([{"role": "user", "content": "state"}])

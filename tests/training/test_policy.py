import pytest

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
    assert tokenizer.encoding.device == model.device
    assert model.kwargs["attention_mask"] is tokenizer.encoding["attention_mask"]
    assert "temperature" not in model.kwargs  # Greedy evaluator default.
    assert policy.last_usage == {"input_tokens": 3, "generated_tokens": 2}


def test_context_limit_refuses_silent_history_truncation():
    policy = QwenPolicy(model=FakeModel(), tokenizer=FakeTokenizer(),
                        generation=GenerationConfig(max_context_tokens=4, max_new_tokens=2))
    with pytest.raises(ValueError, match="no silent truncation"):
        policy.generate([{"role": "user", "content": "state"}])

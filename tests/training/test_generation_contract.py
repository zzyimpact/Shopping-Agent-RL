"""Real tokenizer/config tests only: no model construction or forward calls."""

import os

import pytest

from training.policy import GenerationConfig, QwenPolicy
from training.grpo import load_grpo_config
from training.runtime import load_config


def test_eval_and_grpo_resolved_contract():
    evaluation = load_config()["generation"]
    sampling = load_grpo_config()["sampling"]
    assert evaluation["chat_template_kwargs"] == sampling["chat_template_kwargs"] == {"enable_thinking": False}
    assert not evaluation["do_sample"] and evaluation["max_new_tokens"] == 512
    assert sampling["do_sample"] and sampling["temperature"] == sampling["top_p"] == 1


@pytest.mark.skipif(not os.environ.get("QWEN_TOKENIZER_PATH"), reason="tokenizer-only opt-in")
def test_real_qwen_non_thinking_and_greedy_config():
    from transformers import AutoTokenizer, GenerationConfig as HFConfig
    from training.sft_data import tokenize_with_assistant_mask
    tokenizer = AutoTokenizer.from_pretrained(os.environ["QWEN_TOKENIZER_PATH"], local_files_only=True)
    messages = [{"role": "system", "content": "Shop"}, {"role": "user", "content": "page"}]
    closed = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).endswith("<|im_start|>assistant\n")
    for adapter in (None, "sft-checkpoint"):
        policy = QwenPolicy(model=object(), tokenizer=tokenizer, adapter_path=adapter)
        for config in (policy.generation, GenerationConfig(**load_grpo_config()["sampling"])):
            ids = policy.prompt_token_ids(messages, config)
            assert tokenizer.decode(ids, skip_special_tokens=False).endswith(closed)
            suffix = policy.observation_token_ids("next", previous_token=tokenizer.eos_token_id, sampling=config)
            assert tokenizer.decode(suffix, skip_special_tokens=False).endswith(closed)
    target = "Thought: select\nAction: search[shoes]"
    encoded = tokenize_with_assistant_mask(tokenizer, messages + [
        {"role": "assistant", "content": target}, {"role": "user", "content": "terminal"}])
    trained = tokenizer.decode([x for x in encoded["labels"] if x != -100], skip_special_tokens=False)
    assert trained == target + "<|im_end|>"

    class NoForwardModel:
        device = "cpu"
        generation_config = HFConfig.from_pretrained(os.environ["QWEN_TOKENIZER_PATH"], local_files_only=True)
        def generate(self, *, input_ids, generation_config, **kwargs):
            assert not kwargs["do_sample"] and kwargs["use_model_defaults"] is False
            assert not {"temperature", "top_p", "top_k"} & kwargs.keys()
            generation_config.validate(strict=True)
            assert generation_config.eos_token_id == self.generation_config.eos_token_id
            self.seen = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            # Construct output IDs only; no model/forward/optimizer of any size.
            import torch
            return torch.cat((input_ids, torch.tensor([[tokenizer.eos_token_id]])), dim=1)
    model = NoForwardModel()
    policy = QwenPolicy(model=model, tokenizer=tokenizer)
    policy.generate(messages)
    assert model.seen.endswith(closed)
    assert policy.last_generation["token_ids"] == [tokenizer.eos_token_id]
    assert policy.last_generation["ended_with_eos"]

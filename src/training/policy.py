"""Qwen3 policy runtime 的最薄封装。

模块级不导入 Transformers/PEFT；teacher collection 和纯逻辑测试可以在没有
训练依赖的环境中继续运行。正式构造模型时才执行 lazy imports。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

CONTEXT_BOUNDARY_STRATEGY = "remaining_context_budget_v1"


@dataclass(frozen=True)
class GenerationConfig:
    """项目级 generation 参数；训练实验可在外层 config 覆盖。"""

    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    do_sample: bool = False
    max_context_tokens: int = 32768
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"enable_thinking": False})

    def __post_init__(self) -> None:
        if self.chat_template_kwargs.get("enable_thinking", False) is not False:
            raise ValueError("project v1 requires enable_thinking=False")
        object.__setattr__(self, "chat_template_kwargs", {**self.chat_template_kwargs, "enable_thinking": False})
        if self.max_new_tokens < 1 or self.max_context_tokens <= self.max_new_tokens:
            raise ValueError("invalid generation/context token budget")
        if self.do_sample and (self.temperature <= 0 or not 0 < self.top_p <= 1):
            raise ValueError("sampling requires temperature > 0 and 0 < top_p <= 1")
        if self.top_k is not None and (not isinstance(self.top_k, int) or self.top_k < 0):
            raise ValueError("top_k must be a nonnegative integer")
        if self.min_p is not None and not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be in [0, 1]")


class QwenPolicy:
    """暴露统一 ``generate(messages) -> str`` 的 Qwen3/PEFT policy。"""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        adapter_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        generation: GenerationConfig | None = None,
        model: Any = None,
        tokenizer: Any = None,
        device_map: str | Mapping[str, Any] | None = "auto",
        torch_dtype: Any = "bfloat16",
        trust_remote_code: bool = False,
    ) -> None:
        self.model_path = str(model_path) if model_path is not None else None
        self.tokenizer_path = str(tokenizer_path or model_path)
        self.adapter_path = str(adapter_path) if adapter_path is not None else None
        self.generation = generation or GenerationConfig()
        self.last_usage: dict[str, int] = {}
        self.last_generation: dict[str, Any] = {}
        self.model = model
        self.tokenizer = tokenizer
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        if (model is None) != (tokenizer is None):
            raise ValueError("model 与 tokenizer 必须同时提供，或都留空")
        if model is None and not self.model_path:
            raise ValueError("未提供 model_path")
        if model is None:
            self._load_runtime()

    def _load_runtime(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on training env
            raise RuntimeError(
                "QwenPolicy 需要独立 training environment 的 torch/transformers"
            ) from exc

        dtype = self.torch_dtype
        if dtype == "bfloat16":
            dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=self.trust_remote_code, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
        )
        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover - depends on training env
                raise RuntimeError("加载 adapter 需要独立 training environment 的 peft") from exc
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path, local_files_only=True)
        self.model.eval()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _device(self) -> Any:
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return None

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        """用 Qwen chat template 生成一条完整可解析的 Thought/Action response。"""
        if not self.is_loaded:
            raise RuntimeError("QwenPolicy runtime 尚未加载")
        self.last_usage = {}
        self.last_generation = {}
        encoded = self.tokenizer.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            **dict(self.generation.chat_template_kwargs),
        )
        if hasattr(encoded, "to"):
            device = self._device()
            if device is not None:
                encoded = encoded.to(device)
        if isinstance(encoded, Mapping):
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        else:
            input_ids = encoded
            attention_mask = None
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.generation.max_new_tokens,
            "do_sample": self.generation.do_sample,
            "use_model_defaults": False,
        }
        if self.generation.do_sample:
            kwargs.update(temperature=self.generation.temperature, top_p=self.generation.top_p)
            for name in ("top_k", "min_p"):
                if getattr(self.generation, name) is not None:
                    kwargs[name] = getattr(self.generation, name)
        elif getattr(self.model, "generation_config", None) is not None:
            # Preserve artifact EOS/stopping, but neutralize inherited sampling-only
            # settings on a copy. HF validates these even when greedy ignores them.
            generation = deepcopy(self.model.generation_config)
            generation.do_sample = False
            generation.temperature, generation.top_p, generation.top_k = 1.0, 1.0, 50
            kwargs["generation_config"] = generation
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        prompt_length = int(input_ids.shape[-1])
        remaining = self.generation.max_context_tokens - prompt_length
        effective = max(0, min(self.generation.max_new_tokens, remaining))
        self.last_generation = {
            "input_tokens_before_generation": prompt_length,
            "configured_max_new_tokens": self.generation.max_new_tokens,
            "effective_max_new_tokens": effective,
            "remaining_context_tokens": remaining,
            "context_budget_reduced": effective < self.generation.max_new_tokens,
            "generated_tokens": 0, "eos_reached": False,
            "context_window_cap": False, "normal_generation_cap": False,
            "context_limit": remaining <= 0, "model_called": remaining > 0,
            "token_ids": [], "raw_text": "", "ended_with_eos": False,
            "enable_thinking": False,
        }
        if remaining <= 0:
            self.last_usage = {"input_tokens": prompt_length, "generated_tokens": 0}
            return ""
        kwargs["max_new_tokens"] = effective
        output = self.model.generate(input_ids=input_ids, **kwargs)
        generated = output[0][prompt_length:]
        self.last_usage = {"input_tokens": prompt_length, "generated_tokens": len(generated)}
        token_ids = generated.tolist() if hasattr(generated, "tolist") else list(generated)
        eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id",
                      getattr(self.tokenizer, "eos_token_id", None))
        eos_ids = eos if isinstance(eos, list) else [eos] if eos is not None else []
        ended_with_eos = bool(token_ids and token_ids[-1] in eos_ids)
        context_cap = effective < self.generation.max_new_tokens and len(token_ids) >= effective and not ended_with_eos
        self.last_generation.update(
            token_ids=token_ids, raw_text=str(self.tokenizer.decode(token_ids, skip_special_tokens=False)),
            eos_token_ids=eos_ids, ended_with_eos=ended_with_eos, eos_reached=ended_with_eos,
            generated_tokens=len(token_ids), context_window_cap=context_cap, context_limit=context_cap,
            normal_generation_cap=(effective == self.generation.max_new_tokens
                                   and len(token_ids) >= effective and not ended_with_eos),
        )
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=True)).strip()

    def prompt_token_ids(self, messages, sampling: GenerationConfig) -> list[int]:
        return list(self.tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True,
            **dict(sampling.chat_template_kwargs),
        ))

    def observation_token_ids(self, observation: str, *, previous_token: int,
                              sampling: GenerationConfig) -> list[int]:
        """Only tokenize inserted text; never render/re-tokenize generated history.

        Qwen's native template renders a user observation and the next assistant
        prefix. A sampled im_end already closes the assistant; when generation
        hit its token limit the missing close is inserted and owned by the env.
        """
        eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        boundary = "\n" if previous_token == eos else "<|im_end|>\n"
        suffix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": observation}], tokenize=False,
            add_generation_prompt=True, **dict(sampling.chat_template_kwargs),
        )
        return list(self.tokenizer(boundary + suffix, add_special_tokens=False)["input_ids"])

    def sample(self, input_ids: Sequence[int], *, sampling: GenerationConfig) -> PolicySample:
        """Stochastic Transformers sampling on the exact accumulated token stream."""
        if not sampling.do_sample:
            raise ValueError("GRPO sampling must be stochastic")
        if sampling.top_k not in (None, 0) or sampling.min_p not in (None, 0):
            raise ValueError("GRPO retains top_k=0 and disabled min_p; Mode B is evaluation-only")
        remaining = sampling.max_context_tokens - len(input_ids)
        if remaining < 1:
            raise ValueError("no remaining sampling context")
        import torch
        from transformers import GenerationConfig as HFGenerationConfig

        ids = torch.tensor([list(input_ids)], dtype=torch.long, device=self._device())
        # A fresh config + use_model_defaults=False avoids HF replacing explicit
        # global defaults (temperature/top_p=1) with Qwen artifact settings.
        generation = HFGenerationConfig(
            do_sample=True, temperature=sampling.temperature, top_p=sampling.top_p, top_k=0,
            max_new_tokens=min(sampling.max_new_tokens, remaining), num_beams=1,
            repetition_penalty=1.0, eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id, use_cache=True,
            return_dict_in_generate=True, output_scores=True,
        )
        with torch.inference_mode():
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         generation_config=generation, use_model_defaults=False)
            generated = output.sequences[0, len(input_ids):].tolist()
            # Includes temperature/top-p processing and the sampled EOS. No text round-trip.
            scores = self.model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True,
            )[0].float().cpu().tolist()
        self.last_usage = {"input_tokens": len(input_ids), "generated_tokens": len(generated)}
        return PolicySample(self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
                            generated, scores)


@dataclass(frozen=True)
class PolicySample:
    """Generated IDs and normalized probabilities from the sampling backend itself."""

    text: str
    token_ids: list[int]
    logprobs: list[float]

    def __post_init__(self) -> None:
        import math

        if not self.token_ids or len(self.token_ids) != len(self.logprobs):
            raise ValueError("sample token IDs/logprobs must be non-empty and aligned")
        if any(not math.isfinite(value) or value > 1e-5 for value in self.logprobs):
            raise ValueError("sampling logprobs must be finite normalized log probabilities")

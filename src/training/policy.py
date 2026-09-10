"""Qwen3 policy runtime 的最薄封装。

模块级不导入 Transformers/PEFT；teacher collection 和纯逻辑测试可以在没有
训练依赖的环境中继续运行。正式构造模型时才执行 lazy imports。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GenerationConfig:
    """项目级 generation 参数；训练实验可在外层 config 覆盖。"""

    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False
    max_context_tokens: int = 32768
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1 or self.max_context_tokens <= self.max_new_tokens:
            raise ValueError("invalid generation/context token budget")
        if self.do_sample and (self.temperature <= 0 or not 0 < self.top_p <= 1):
            raise ValueError("sampling requires temperature > 0 and 0 < top_p <= 1")


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
        }
        if self.generation.do_sample:
            kwargs.update(temperature=self.generation.temperature, top_p=self.generation.top_p)
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        prompt_length = int(input_ids.shape[-1])
        if prompt_length + self.generation.max_new_tokens > self.generation.max_context_tokens:
            raise ValueError("visible history plus generation budget exceeds max_context_tokens; no silent truncation")
        output = self.model.generate(input_ids=input_ids, **kwargs)
        generated = output[0][prompt_length:]
        self.last_usage = {"input_tokens": prompt_length, "generated_tokens": len(generated)}
        return str(self.tokenizer.decode(generated, skip_special_tokens=True)).strip()

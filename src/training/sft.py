"""TRL 1.12.0 + PEFT SFT construction, with explicit assistant-only labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.runtime import append_metrics


@dataclass(frozen=True)
class SFTConfigSpec:
    output_dir: Path
    num_train_epochs: float = 4.0
    learning_rate: float = 1e-5
    effective_batch_size: int = 32
    per_device_train_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    max_length: int = 32768
    lora_r: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] | None = None
    gradient_checkpointing: bool = True
    save_steps: int = 100
    logging_steps: int = 10
    seed: int = 1

    def validate(self) -> None:
        required = (self.per_device_train_batch_size, self.gradient_accumulation_steps,
                    self.lora_r, self.lora_alpha)
        if any(value is None or value < 1 for value in required) or not self.target_modules:
            raise ValueError("explicit microbatch, accumulation, LoRA rank/alpha/target_modules required (GPU-PREFLIGHT)")
        if self.per_device_train_batch_size * self.gradient_accumulation_steps != self.effective_batch_size:
            raise ValueError("single-GPU microbatch * accumulation must equal effective_batch_size")
        if min(self.num_train_epochs, self.learning_rate, self.max_length, self.save_steps, self.logging_steps) <= 0:
            raise ValueError("SFT epochs/LR/context/cadence must be positive")


def _training_imports():
    try:
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
        from transformers import TrainerCallback
    except ImportError as exc:
        raise RuntimeError("SFT requires the independent training environment (TRL/PEFT/Transformers)") from exc
    return SFTConfig, SFTTrainer, LoraConfig, TrainerCallback


def build_sft_trainer(*, model: Any, tokenizer: Any, train_dataset: Any,
                      config: SFTConfigSpec, use_cpu: bool = False) -> Any:
    """Caller supplies pretokenized input_ids/labels from sft_data.py."""
    config.validate()
    if not len(train_dataset):
        raise ValueError("SFT requires a non-empty dataset")
    # Do not let TRL's collator silently fall back to training on every input token.
    for row in train_dataset:
        if "labels" not in row or len(row["input_ids"]) != len(row["labels"]):
            raise ValueError("SFT requires explicit, aligned assistant-only labels")
        if len(row["input_ids"]) > config.max_length:
            raise ValueError("SFT row exceeds max_length")
    SFTConfig, SFTTrainer, LoraConfig, TrainerCallback = _training_imports()

    class MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero:
                append_metrics(config.output_dir / "metrics.jsonl",
                               {"event": "sft", "step": state.global_step, **(logs or {})})

    args = SFTConfig(
        output_dir=str(config.output_dir / "checkpoints"),
        num_train_epochs=config.num_train_epochs, learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        use_cpu=use_cpu, bf16=not use_cpu, fp16=False, seed=config.seed, data_seed=config.seed,
        dataloader_pin_memory=not use_cpu,
        gradient_checkpointing=config.gradient_checkpointing,
        # Labels already encode assistant-only loss. Do not invoke a second template/mask.
        assistant_only_loss=False, completion_only_loss=False,
        dataset_kwargs={"skip_prepare_dataset": True}, max_length=None,
        packing=False, padding_free=False, report_to="none",
        save_strategy="steps", save_steps=config.save_steps, logging_steps=config.logging_steps,
        save_only_model=False, optim="adamw_torch", warmup_ratio=0.0, max_grad_norm=1.0,
    )
    adapter = LoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules), task_type="CAUSAL_LM",
    )
    model.config.use_cache = False
    return SFTTrainer(model=model, args=args, train_dataset=train_dataset,
                      processing_class=tokenizer, peft_config=adapter,
                      callbacks=[MetricsCallback()])


def train_sft(trainer: Any, *, resume_from_checkpoint: str | Path | None = None) -> Any:
    """HF owns adapter/optimizer/scheduler/step/RNG checkpoint restoration."""
    result = trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
    trainer.save_model(str(Path(trainer.args.output_dir) / "final"))
    return result

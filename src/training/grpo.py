"""Pinned TRL 1.12.0 glue; GRPO optimization remains entirely in TRL."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any

from rewards.shopsim_reward import METRIC_KEYS
from training.policy import GenerationConfig, QwenPolicy
from training.rollout import AgentRollout
from training.runtime import PROJECT_ROOT, append_metrics, load_config, sha256_file


@dataclass(frozen=True)
class GRPOSpec:
    loss_type: str = "grpo"
    num_generations: int = 8
    beta: float = 0.0
    scale_rewards: str = "group"
    learning_rate: float = 1e-6
    task_groups_per_update: int = 32
    max_steps: int = 200
    per_device_train_batch_size: int | None = None
    lora_r: int | None = None
    lora_alpha: int | None = None
    target_modules: list[str] | None = None
    save_steps: int = 10
    logging_steps: int = 1

    def validate(self, init: str) -> None:
        if (self.loss_type, self.beta, self.scale_rewards) != ("grpo", 0.0, "group"):
            raise ValueError("Round C requires loss_type=grpo, beta=0, scale_rewards=group")
        if self.num_generations < 2 or min(self.task_groups_per_update, self.max_steps) < 1:
            raise ValueError("G >= 2 and positive groups/updates required")
        if not self.per_device_train_batch_size or self.per_device_train_batch_size < 1:
            raise ValueError("explicit microbatch required after GPU-PREFLIGHT")
        if self.generation_batch_size % self.per_device_train_batch_size:
            raise ValueError("G * task_groups_per_update must be divisible by microbatch")
        if min(self.learning_rate, self.save_steps, self.logging_steps) <= 0:
            raise ValueError("positive LR/checkpoint/logging cadence required")
        if init not in {"base", "sft_adapter"}:
            raise ValueError("init must be base or sft_adapter")
        if init == "base" and (not self.lora_r or self.lora_r < 1 or not self.lora_alpha
                               or self.lora_alpha < 1 or not self.target_modules):
            raise ValueError("Direct GRPO requires explicit fresh LoRA settings (GPU-PREFLIGHT)")

    @property
    def generation_batch_size(self) -> int:
        return self.num_generations * self.task_groups_per_update

    @property
    def accumulation_steps(self) -> int:
        return self.generation_batch_size // self.per_device_train_batch_size


def load_grpo_config(override=None) -> dict[str, Any]:
    import yaml

    config = load_config()
    config.pop("sft")
    config.pop("generation")
    for path in [PROJECT_ROOT / "configs/training/grpo.yaml", *([Path(override)] if override else [])]:
        for key, value in yaml.safe_load(path.read_text()).items():
            config[key] = ({**config[key], **value} if isinstance(value, dict)
                           and isinstance(config.get(key), dict) else value)
    config["sampling"] = asdict(GenerationConfig(**config["sampling"]))
    return config


def task_schedule(task_ids: list[str], *, seed: int, groups: int, updates: int) -> list[str]:
    """Fixed seeded schedule of task identities, NOT pregenerated trajectories.

    TRL RepeatSampler(shuffle=False) has no private RNG cursor to restore. HF
    skips consumed batches on resume; schedule hash/order is part of run identity.
    """
    if not task_ids or groups < 1 or updates < 1:
        raise ValueError("nonempty tasks and positive groups/updates required")
    rng, schedule = random.Random(seed), []
    while len(schedule) < groups * updates:
        epoch = list(task_ids)
        rng.shuffle(epoch)
        schedule.extend(epoch)
    return schedule[:groups * updates]


def task_dataset_rows(schedule: list[str], scenario: str) -> list[dict[str, Any]]:
    # Empty prompt is TRL transport only. The real prompt comes from reset inside AgentRollout.
    return [{"prompt": [{"role": "user", "content": ""}], "task_id": task,
             "scenario": scenario, "schedule_index": index} for index, task in enumerate(schedule)]


def group_metadata(inputs, *, scenario: str, group_size: int) -> tuple[dict, ...]:
    if not inputs or len(inputs) % group_size:
        raise ValueError("TRL batch must contain complete G groups on the single process")
    metadata = tuple({key: row[key] for key in ("task_id", "scenario", "schedule_index")} for row in inputs)
    for start in range(0, len(metadata), group_size):
        group = metadata[start:start + group_size]
        if any(row != group[0] or row["scenario"] != scenario for row in group):
            raise ValueError("TRL batch is not contiguous independent replicas of the same scheduled task")
    return metadata


@contextmanager
def metadata_bridge(trainer, inputs, *, scenario: str, group_size: int):
    if getattr(trainer, "_shop_batch", None) is not None:
        raise RuntimeError("nested rollout metadata bridge")
    trainer._shop_batch = group_metadata(inputs, scenario=scenario, group_size=group_size)
    try:
        yield
    finally:
        trainer._shop_batch = None


def rollout_reward(*, rollout_reward, **kwargs):
    """Pure trajectory scalar passthrough; never call an environment or scorer."""
    return list(rollout_reward)


def rollout_metrics(episodes, group_size: int) -> dict[str, Any]:
    rewards = [episode.reward for episode in episodes]
    group_stds = [statistics.stdev(rewards[i:i + group_size]) for i in range(0, len(rewards), group_size)]
    count = len(episodes)
    return {
        "reward_mean": statistics.mean(rewards), "reward_std": statistics.stdev(rewards),
        **{key: statistics.mean(ep.reward_metrics[key] for ep in episodes) for key in METRIC_KEYS},
        "group_reward_std": statistics.mean(group_stds),
        "zero_variance_group_fraction": sum(value == 0 for value in group_stds) / len(group_stds),
        "episode_steps": statistics.mean(ep.steps for ep in episodes),
        "invalid_rate": sum(ep.status == "invalid_action" for ep in episodes) / count,
        "malformed_rate": sum(ep.status == "malformed_action" for ep in episodes) / count,
        "context_limit_rate": sum(ep.status == "context_limit" for ep in episodes) / count,
        "generated_policy_tokens": sum(sum(ep.token_trace.env_mask) for ep in episodes),
        "rollout_wall_s": sum(ep.wall_time_s for ep in episodes),
    }


def collect_online_batch(prompts, trainer, *, policy, env_factory, scenario: str,
                         sampling: GenerationConfig, spec: GRPOSpec, reward_alpha: float,
                         max_action_steps: int) -> dict[str, list]:
    metadata = trainer._shop_batch
    if metadata is None or len(prompts) != len(metadata):
        raise ValueError("rollout_func must run inside the metadata bridge")
    runner = AgentRollout(policy=policy, env_factory=env_factory, scenario=scenario,
                          reward_alpha=reward_alpha, max_action_steps=max_action_steps)
    # Inputs are ALREADY repeated by TRL's RepeatSampler. Exactly one fresh session per row.
    episodes = [runner.run(row["task_id"], sampling=sampling) for row in metadata]
    metrics = rollout_metrics(episodes, spec.num_generations)
    trainer._shop_rollout_seconds = getattr(trainer, "_shop_rollout_seconds", 0.0) + metrics["rollout_wall_s"]
    trainer._shop_last_metrics = {"event": "rollout", "update_index": trainer.state.global_step,
                                  "task_ids": [row["task_id"] for row in metadata],
                                  "schedule_indices": [row["schedule_index"] for row in metadata], **metrics}
    return {
        "prompt_ids": [ep.token_trace.prompt_ids for ep in episodes],
        "completion_ids": [ep.token_trace.completion_ids for ep in episodes],
        "logprobs": [ep.token_trace.logprobs for ep in episodes],
        "env_mask": [ep.token_trace.env_mask for ep in episodes],
        "rollout_reward": [ep.reward for ep in episodes],
        "rollout_metrics": [ep.reward_metrics for ep in episodes],
        "rollout_status": [ep.status for ep in episodes],
    }


def shop_trainer_class(base_class):
    """One private method override; no copied generation/loss/sampler implementation."""
    class ShopGRPOTrainer(base_class):
        def _generate_and_score_completions(self, inputs):
            with metadata_bridge(self, inputs, scenario=self.shop_scenario, group_size=self.num_generations):
                # Parent merges extra reward fields into rows, so do not mutate the Dataset batch.
                return super()._generate_and_score_completions([dict(row) for row in inputs])

    return ShopGRPOTrainer


def validate_lineage(config: dict) -> dict[str, Any]:
    if config["init"] == "base":
        if config.get("adapter_path") or config.get("sft_run_dir"):
            raise ValueError("Direct GRPO uses base + fresh LoRA, without SFT adapter")
        return {"init": "base"}
    if not config.get("sft_run_dir") or not config.get("adapter_path"):
        raise ValueError("SFT-init requires sft_run_dir and adapter_path")
    root, adapter = Path(config["sft_run_dir"]).resolve(), Path(config["adapter_path"]).resolve()
    manifest = root / "run_manifest.json"
    identity = json.loads(manifest.read_text())["identity"]
    source = identity["config"]
    if source["mode"] != "sft" or source["scenario"] != config["scenario"]:
        raise ValueError("SFT adapter scenario/mode mismatch")
    if Path(identity["inputs"]["model_path"]).resolve() != Path(config["model_path"]).resolve():
        raise ValueError("SFT-init base model path mismatch")
    tokenizer_path = config.get("tokenizer_path") or config["model_path"]
    if Path(identity["inputs"]["tokenizer_path"]).resolve() != Path(tokenizer_path).resolve():
        raise ValueError("SFT-init tokenizer path mismatch")
    if adapter.parent != root / "checkpoints" or not (adapter / "adapter_config.json").is_file():
        raise ValueError("adapter must belong to the declared SFT run")
    return {"init": "sft_adapter", "sft_run_manifest_sha256": sha256_file(manifest),
            "adapter_path": str(adapter), "adapter_config_sha256": sha256_file(adapter / "adapter_config.json")}


def initialize_grpo_model(model_path, *, init: str, adapter_path=None):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, local_files_only=True)
    if init == "sft_adapter":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True, local_files_only=True)
    return model


def build_grpo_trainer(*, model, tokenizer, dataset, config, env_factory, use_cpu: bool = False):
    from importlib.metadata import version
    if version("trl") != "1.12.0":
        raise RuntimeError("custom rollout glue requires pinned trl==1.12.0")
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig
    from transformers import TrainerCallback

    spec = GRPOSpec(**config["grpo"])
    spec.validate(config["init"])
    sampling = GenerationConfig(**config["sampling"])
    if not sampling.do_sample:
        raise ValueError("GRPO requires stochastic sampling")
    output = Path(config["output_dir"])
    args = GRPOConfig(
        output_dir=str(output / "checkpoints"), loss_type=spec.loss_type,
        num_generations=spec.num_generations, beta=spec.beta, scale_rewards=spec.scale_rewards,
        learning_rate=spec.learning_rate, max_steps=spec.max_steps,
        per_device_train_batch_size=spec.per_device_train_batch_size,
        gradient_accumulation_steps=spec.accumulation_steps, steps_per_generation=spec.accumulation_steps,
        num_iterations=1, shuffle_dataset=False, remove_unused_columns=False,
        seed=config["seed"], data_seed=config["seed"], dataloader_num_workers=0, ignore_data_skip=False,
        temperature=sampling.temperature, top_p=sampling.top_p, top_k=0,
        max_completion_length=sampling.max_context_tokens, mask_truncated_completions=False,
        use_vllm=False, use_liger_kernel=False, disable_dropout=True,
        use_cpu=use_cpu, bf16=not use_cpu, fp16=False, gradient_checkpointing=True,
        dataloader_pin_memory=not use_cpu,
        report_to="none", log_completions=False, save_only_model=False, eval_strategy="no",
        save_strategy="steps", save_steps=spec.save_steps, logging_steps=spec.logging_steps,
        optim="adamw_torch", warmup_ratio=0.0, epsilon=0.2, max_grad_norm=1.0,
    )
    peft = None
    if config["init"] == "base":
        peft = LoraConfig(r=spec.lora_r, lora_alpha=spec.lora_alpha, target_modules=spec.target_modules,
                          lora_dropout=0.0, task_type="CAUSAL_LM")

    def rollout(prompts, trainer):
        # Always unwrap the CURRENT trainer policy, including after checkpoint restore/update.
        live_model = trainer.accelerator.unwrap_model(trainer.model_wrapped)
        was_training = live_model.training
        live_model.eval()
        try:
            policy = QwenPolicy(model=live_model, tokenizer=tokenizer, generation=sampling)
            result = collect_online_batch(prompts, trainer, policy=policy, env_factory=env_factory,
                                           scenario=config["scenario"], sampling=sampling, spec=spec,
                                           reward_alpha=config["reward_alpha"], max_action_steps=config["max_action_steps"])
            append_metrics(output / "metrics.jsonl", trainer._shop_last_metrics)
            return result
        finally:
            live_model.train(was_training)

    class MetricsCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()
            trainer._shop_rollout_seconds = 0.0

        def on_step_end(self, args, state, control, **kwargs):
            metrics = {
                "event": "update", "step": state.global_step,
                "update_wall_s": max(0.0, time.monotonic() - self.started - trainer._shop_rollout_seconds),
            }
            if args.device.type == "cuda":
                import torch
                metrics["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            append_metrics(output / "metrics.jsonl", metrics)

        def on_log(self, args, state, control, logs=None, **kwargs):
            append_metrics(output / "metrics.jsonl", {"event": "trl", "step": state.global_step, **(logs or {})})

    Trainer = shop_trainer_class(GRPOTrainer)
    trainer = Trainer(model=model, processing_class=tokenizer, args=args, train_dataset=dataset,
                      reward_funcs=rollout_reward, rollout_func=rollout, peft_config=peft,
                      callbacks=[MetricsCallback()])
    trainer.shop_scenario = config["scenario"]
    if trainer.accelerator.num_processes != 1:
        raise ValueError("Round C correctness path requires one process/GPU")
    return trainer


def train_grpo(trainer, *, resume_from_checkpoint=None):
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(Path(trainer.args.output_dir) / "final"))
    return result

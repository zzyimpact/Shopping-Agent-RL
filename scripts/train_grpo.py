#!/usr/bin/env python3
"""Online ShopSimulator GRPO; run only in the independent training environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from env.teacher_env_client import TeacherEnvClient
from training.eval import load_task_ids
from training.grpo import (
    GRPOSpec, build_grpo_trainer, initialize_grpo_model, load_grpo_config,
    task_dataset_rows, task_schedule, train_grpo, validate_lineage,
)
from training.policy import GenerationConfig
from training.runtime import json_hash, model_metadata, prepare_run, sha256_file


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="small YAML override of configs/training/grpo.yaml")
    parser.add_argument("--scenario", choices=("single", "single_persona"))
    parser.add_argument("--init", choices=("base", "sft_adapter"))
    for name in ("model-path", "tokenizer-path", "adapter-path", "sft-run-dir",
                 "train-task-manifest", "output-dir", "endpoint"):
        parser.add_argument("--" + name)
    for name in ("seed", "num-generations", "task-groups-per-update", "max-steps",
                 "per-device-train-batch-size", "lora-r", "lora-alpha",
                 "save-steps", "logging-steps", "max-new-tokens", "max-context-tokens"):
        parser.add_argument("--" + name, type=int)
    for name in ("learning-rate", "temperature", "top-p"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--target-modules", nargs="+")
    reward = parser.add_mutually_exclusive_group()
    reward.add_argument("--reward", choices=("strict", "loose"))
    reward.add_argument("--alpha", type=float)
    parser.add_argument("--resume-from-checkpoint")
    args = vars(parser.parse_args(argv))
    config = load_grpo_config(args.pop("config"))
    resume = args.pop("resume_from_checkpoint")
    reward_name, alpha = args.pop("reward"), args.pop("alpha")
    for section in ("grpo", "sampling"):
        for key in config[section]:
            value = args.pop(key, None)
            if value is not None:
                config[section][key] = value
    config.update({key: value for key, value in args.items() if value is not None})
    config["reward_alpha"] = (alpha if alpha is not None else
                              {"strict": 1.0, "loose": 0.0}[reward_name] if reward_name else
                              config["reward_alpha"])
    config["mode"] = "grpo"
    for key in ("scenario", "model_path", "train_task_manifest", "output_dir"):
        if not config.get(key):
            parser.error(f"{key} required via CLI or config")
    if config["scenario"] not in {"single", "single_persona"} or not 0 <= config["reward_alpha"] <= 1:
        parser.error("invalid scenario or reward_alpha")
    spec = GRPOSpec(**config["grpo"])
    spec.validate(config["init"])
    sampling = GenerationConfig(**config["sampling"])
    if not sampling.do_sample:
        parser.error("GRPO requires stochastic sampling")
    return config, spec, resume


def main(argv=None) -> int:
    config, spec, resume = parse_config(argv)
    tasks = load_task_ids(config["train_task_manifest"], scenario=config["scenario"], split="train")
    schedule = task_schedule(tasks, seed=config["seed"], groups=spec.task_groups_per_update,
                             updates=spec.max_steps)
    tokenizer_path = config.get("tokenizer_path") or config["model_path"]
    inputs = {"train_task_manifest_sha256": sha256_file(config["train_task_manifest"]),
              "task_schedule_sha256": json_hash(schedule), "scheduled_groups": len(schedule),
              "token_contract": "qwen-append-only-sampled-ids-v1", "trl_contract": "1.12.0",
              "lineage": validate_lineage(config), **model_metadata(config["model_path"], tokenizer_path)}
    root = prepare_run(config["output_dir"], config=config, inputs=inputs, resume_from_checkpoint=resume)
    try:
        from importlib.metadata import version
        from datasets import Dataset
        from transformers import AutoTokenizer, set_seed
        import torch
    except ImportError as exc:
        raise RuntimeError("train_grpo requires the independent training environment; do not install into teacher env") from exc
    if version("trl") != "1.12.0":
        raise RuntimeError("custom rollout glue requires pinned trl==1.12.0")
    if torch.cuda.device_count() != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("select one training GPU with CUDA_VISIBLE_DEVICES for GRPO group semantics")
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)
    dataset = Dataset.from_list(task_dataset_rows(schedule, config["scenario"]))
    model = initialize_grpo_model(config["model_path"], init=config["init"], adapter_path=config.get("adapter_path"))
    trainer = build_grpo_trainer(model=model, tokenizer=tokenizer, dataset=dataset, config=config,
                                 env_factory=lambda: TeacherEnvClient(config["endpoint"]))
    train_grpo(trainer, resume_from_checkpoint=resume)
    tokenizer.save_pretrained(str(root / "checkpoints" / "final"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

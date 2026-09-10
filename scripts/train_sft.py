#!/usr/bin/env python3
"""Train LoRA from a selected accepted manifest; requires the training environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.eval import load_task_ids
from training.sft import SFTConfigSpec, build_sft_trainer, train_sft
from training.sft_data import load_selected_examples, tokenize_with_assistant_mask
from training.runtime import json_hash, load_config, model_metadata, prepare_run, sha256_file


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="small YAML override of configs/training/defaults.yaml")
    parser.add_argument("--scenario", choices=("single", "single_persona"))
    for name in ("model-path", "tokenizer-path", "selection-manifest", "accepted-root",
                 "train-task-manifest", "output-dir"):
        parser.add_argument("--" + name)
    parser.add_argument("--epochs", type=float, dest="num_train_epochs")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--target-modules", nargs="+")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume-from-checkpoint")
    args = vars(parser.parse_args(argv))
    config = load_config(args.pop("config"))
    resume = args.pop("resume_from_checkpoint")
    for key in list(config["sft"]):
        value = args.pop(key, None)
        if value is not None:
            config["sft"][key] = value
    config.update({key: value for key, value in args.items() if value is not None})
    config["mode"] = "sft"
    config.pop("generation")
    for key in ("scenario", "model_path", "selection_manifest", "accepted_root", "train_task_manifest", "output_dir"):
        if not config.get(key):
            parser.error(f"{key} required via CLI or config")
    if config["scenario"] not in {"single", "single_persona"}:
        parser.error("unsupported scenario")
    if config.get("adapter_path"):
        parser.error("Round B SFT starts from base; resume uses --resume-from-checkpoint")
    spec = SFTConfigSpec(output_dir=Path(config["output_dir"]).resolve(), seed=config["seed"], **config["sft"])
    spec.validate()  # Refuse an accidental full-parameter job before importing/loading any model.
    return config, spec, resume


def main(argv=None) -> int:
    config, spec, resume = parse_config(argv)
    allowed = set(load_task_ids(config["train_task_manifest"], scenario=config["scenario"], split="train"))
    examples = load_selected_examples(config["selection_manifest"], accepted_root=config["accepted_root"],
                                      scenario=config["scenario"])
    if any(example.task_id not in allowed for example in examples):
        raise ValueError("selected accepted artifact is not in this scenario's TRAIN split")
    tokenizer_path = config["tokenizer_path"] or config["model_path"]
    inputs = {"selection_manifest_sha256": sha256_file(config["selection_manifest"]),
              "train_task_manifest_sha256": sha256_file(config["train_task_manifest"]),
              "selected_count": len(examples), "sft_projection": "visible-messages-qwen-offsets-assistant-eos-v2",
              "selected_content_sha256": json_hash([(e.accepted_id, e.task_id, e.source_sha256) for e in examples]),
              **model_metadata(config["model_path"], tokenizer_path)}
    root = prepare_run(config["output_dir"], config=config, inputs=inputs, resume_from_checkpoint=resume)
    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        import torch
    except ImportError as exc:
        raise RuntimeError("train_sft requires the independent training environment; do not install into teacher env") from exc

    # This first path intentionally uses one GPU. No device_map='auto' training sharding.
    if torch.cuda.device_count() != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("select one training GPU with CUDA_VISIBLE_DEVICES for effective batch semantics")
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)
    dataset = Dataset.from_list([tokenize_with_assistant_mask(tokenizer, example.messages, max_length=spec.max_length)
                                 for example in examples])
    model = AutoModelForCausalLM.from_pretrained(config["model_path"], torch_dtype=torch.bfloat16,
                                               local_files_only=True)
    trainer = build_sft_trainer(model=model, tokenizer=tokenizer, train_dataset=dataset, config=spec)
    train_sft(trainer, resume_from_checkpoint=resume)
    tokenizer.save_pretrained(str(root / "checkpoints" / "final"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

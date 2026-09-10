#!/usr/bin/env python3
"""Evaluate Base or PEFT policy using a fixed test task manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from env.teacher_env_client import TeacherEnvClient
from training.eval import evaluate_policy, load_task_ids
from training.policy import GenerationConfig, QwenPolicy
from training.runtime import load_config, model_metadata, prepare_run, sha256_file, append_metrics


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="small YAML override of configs/training/defaults.yaml")
    parser.add_argument("--scenario", choices=("single", "single_persona"))
    parser.add_argument("--model-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--adapter", dest="adapter_path")
    parser.add_argument("--manifest", dest="task_manifest")
    parser.add_argument("--endpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    reward = parser.add_mutually_exclusive_group()
    reward.add_argument("--reward", choices=("strict", "loose"))
    reward.add_argument("--alpha", type=float)
    args = vars(parser.parse_args(argv))
    config = load_config(args.pop("config"))
    reward_name, alpha = args.pop("reward"), args.pop("alpha")
    for key in ("max_new_tokens", "do_sample", "temperature", "top_p"):
        value = args.pop(key)
        if value is not None:
            config["generation"][key] = value
    config.update({key: value for key, value in args.items() if value is not None})
    config["reward_alpha"] = (alpha if alpha is not None else
                              {"strict": 1.0, "loose": 0.0}[reward_name] if reward_name else
                              config.get("reward_alpha", 1.0))
    config["mode"] = "eval"
    config.pop("sft")
    config.setdefault("endpoint", "http://127.0.0.1:5500")
    for key in ("scenario", "model_path", "task_manifest", "output_dir"):
        if not config.get(key):
            parser.error(f"{key} required via CLI or config")
    if config["scenario"] not in {"single", "single_persona"} or not 0 <= config["reward_alpha"] <= 1:
        parser.error("invalid scenario or reward_alpha")
    return config


def main(argv=None) -> int:
    config = parse_config(argv)
    task_ids = load_task_ids(config["task_manifest"], scenario=config["scenario"], split="test")
    tokenizer_path = config["tokenizer_path"] or config["model_path"]
    inputs = {"task_manifest_sha256": sha256_file(config["task_manifest"]),
              **model_metadata(config["model_path"], tokenizer_path)}
    root = prepare_run(config["output_dir"], config=config, inputs=inputs)
    from transformers import set_seed

    set_seed(config["seed"])
    policy = QwenPolicy(config["model_path"], tokenizer_path=tokenizer_path,
                        adapter_path=config["adapter_path"], generation=GenerationConfig(**config["generation"]))
    summary = evaluate_policy(policy=policy, scenario=config["scenario"], task_ids=task_ids,
                              env_factory=lambda: TeacherEnvClient(config["endpoint"]),
                              reward_alpha=config["reward_alpha"],
                              max_action_steps=config["max_action_steps"], output_dir=root / "eval")
    append_metrics(root / "metrics.jsonl", {"event": "eval", **summary})
    print(f"scenario={config['scenario']} episodes={summary['episodes']} metrics={summary['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate Base or PEFT policy using a fixed test task manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from env.teacher_env_client import TeacherEnvClient
from training.eval import completed_rows, evaluate_policy, load_task_ids
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
    parser.add_argument("--resume", action="store_true",
                        help="resume an existing compatible eval directory from episodes.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="validate files/config only; no model, GPU, or environment calls")
    parser.add_argument("--fixed-128", action="store_true", help="enforce P2 frozen subset hash and Base evaluation defaults")
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
    GenerationConfig(**config["generation"])
    return config


FIXED_IDS_HASH = {
    "single": "2bddabe94e2367fff581c771b2bee9d602906e5aa07e83be51a8c69624bd9155",
    "single_persona": "68db6d538a6c06efe0e181957499bef470be73b01ae77ad2eb48174864ba3410",
}


def validate_inputs(config):
    tasks = load_task_ids(config["task_manifest"], scenario=config["scenario"], split="test")
    ids_hash = hashlib.sha256("\n".join(tasks).encode()).hexdigest()
    if config["fixed_128"]:
        if len(tasks) != 128 or ids_hash != FIXED_IDS_HASH[config["scenario"]]:
            raise ValueError("fixed-128 manifest differs from frozen P2 task IDs/order")
        expected = dict(GenerationConfig().__dict__)
        actual = {**expected, **config["generation"]}
        if (config["seed"] != 1 or config["max_action_steps"] != 30 or config["adapter_path"]
                or actual != expected):
            raise ValueError("fixed-128 Base requires seed=1, 30 steps, default greedy/512/native template, no adapter")
    tokenizer = config["tokenizer_path"] or config["model_path"]
    for root, name in ((config["model_path"], "config.json"),
                       (config["model_path"], "model.safetensors.index.json"),
                       (tokenizer, "tokenizer_config.json")):
        if not (Path(root) / name).is_file():
            raise ValueError(f"missing local model/tokenizer metadata: {Path(root) / name}")
    inputs = {"task_manifest_sha256": sha256_file(config["task_manifest"]), "task_ids_sha256": ids_hash,
              "task_count": len(tasks), **model_metadata(config["model_path"], tokenizer)}
    return tasks, inputs


def prepare_eval_run(config, inputs, *, resume):
    root = Path(config["output_dir"]).resolve()
    if not resume:
        root = prepare_run(root, config=config, inputs=inputs)
        (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        return root
    previous = json.loads((root / "run_manifest.json").read_text())
    if previous["identity"] != {"config": config, "inputs": inputs}:
        raise ValueError("evaluation resume config/manifest/model identity mismatch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    if previous["git_commit"] != commit:
        raise ValueError("evaluation resume requires the original project commit")
    if config["generation"]["do_sample"]:
        raise ValueError("evaluation resume currently supports greedy decoding only")
    append_metrics(root / "metrics.jsonl", {"event": "eval_resume", "git_commit": commit})
    return root


def validate_eval_health(health):
    if health.get("environment_version") != "task-scoped-v3-multisession":
        raise ValueError("incompatible environment protocol")
    if health.get("status") != "ok" or health.get("task_split") != "test":
        raise ValueError("evaluation requires a healthy explicit TEST-only endpoint; use port 5200")


def main(argv=None) -> int:
    config = parse_config(argv)
    resume, dry_run = config.pop("resume"), config.pop("dry_run")
    task_ids, inputs = validate_inputs(config)
    if dry_run:
        print(json.dumps({"event": "dry_run", "config": config, "inputs": inputs,
                          "model_loaded": False, "environment_checked": False}, ensure_ascii=False, indent=2))
        return 0
    # Validate existing results before loading any weights. No environment is replayed.
    if resume:
        completed_rows(Path(config["output_dir"]) / "eval/episodes.jsonl", task_ids, config["scenario"])
    with TeacherEnvClient(config["endpoint"]) as env:
        health = env.health().payload
    validate_eval_health(health)
    print(json.dumps({"event": "environment_health", "health": health}), flush=True)
    import torch
    from transformers import set_seed
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("formal evaluator requires one visible CUDA GPU; no CPU fallback")
    root = prepare_eval_run(config, inputs, resume=resume)
    set_seed(config["seed"])
    print(json.dumps({"event": "model_loading", "path": config["model_path"], "run": str(root)}), flush=True)
    policy = QwenPolicy(config["model_path"], tokenizer_path=config["tokenizer_path"] or config["model_path"],
                       adapter_path=config["adapter_path"], generation=GenerationConfig(**config["generation"]),
                       device_map={"": 0})
    summary = evaluate_policy(policy=policy, scenario=config["scenario"], task_ids=task_ids,
                              env_factory=lambda: TeacherEnvClient(config["endpoint"]),
                              reward_alpha=config["reward_alpha"],
                              max_action_steps=config["max_action_steps"], output_dir=root / "eval", resume=resume)
    append_metrics(root / "metrics.jsonl", {"event": "eval", **summary})
    print(json.dumps({"event": "summary", **summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

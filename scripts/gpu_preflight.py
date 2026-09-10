#!/usr/bin/env python3
"""D2/P5 bounded GPU contract checks. TEST ONLY; never starts a service or formal run.

Run stages in fresh processes to release model/Trainer memory between checks.
Uses the existing policy, AgentRollout, SFT/GRPO factories and HF checkpoint paths.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Reuse the bounded FakeEnv fixture; no remote service substitute.
sys.path.insert(0, str(ROOT / "src"))


def memory():
    import torch
    torch.cuda.synchronize()
    return {"allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30}


def parameter_hash(model):
    """Hash tiny trainable adapters only; never hash all 8B base weights."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert "lora_" in name, name
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_model(path):
    import torch
    from transformers import AutoModelForCausalLM
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map={"": 0})
    assert next(model.parameters()).device.type == "cuda"
    assert next(model.parameters()).dtype == torch.bfloat16
    print(json.dumps({"event": "model_loaded", "seconds": time.monotonic() - started, **memory()}), flush=True)
    return model


def runtime_gate():
    import platform
    import torch
    expected = {"transformers": "4.57.6", "trl": "1.12.0", "peft": "0.19.1",
                "accelerate": "1.15.0", "datasets": "4.8.5", "tokenizers": "0.22.2"}
    assert torch.__version__.split("+")[0] == "2.8.0"
    assert all(version(name) == value for name, value in expected.items())
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    initial = memory()
    x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
    assert torch.isfinite(x @ x).all() and torch.cuda.is_bf16_supported()
    prop = torch.cuda.get_device_properties(0)
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "gpu": prop.name, "capability": torch.cuda.get_device_capability(0),
            "vram_gib": prop.total_memory / 2**30, "bf16": True, "initial_memory": initial, **expected}


def base_stage(args, tokenizer):
    import torch
    from peft import LoraConfig, get_peft_model
    from training.policy import GenerationConfig, QwenPolicy
    from rollout.prompt import UPSTREAM_SINGLE_PROMPT
    started = time.monotonic()
    model = load_model(args.model_path).eval()
    load_s = time.monotonic() - started
    report = {"load_s": load_s, "class": type(model).__name__, "dtype": str(model.dtype),
              "device": str(model.device), "params": sum(p.numel() for p in model.parameters()),
              "attention_backend": model.config._attn_implementation, "base_memory": memory()}
    with torch.no_grad():
        output = model(**tokenizer("Hello", return_tensors="pt").to("cuda"))
        assert torch.isfinite(output.logits).all()
    del output
    sampling = GenerationConfig(do_sample=True, max_new_tokens=128, max_context_tokens=4096,
                                 chat_template_kwargs={"enable_thinking": False})
    policy = QwenPolicy(model=model, tokenizer=tokenizer)
    prompt = policy.prompt_token_ids([{"role": "system", "content": UPSTREAM_SINGLE_PROMPT},
                                      {"role": "user", "content": "Instruction: 买一双蓝色运动鞋。\n搜索功能是否可用: True\n可点击的按钮: []"}], sampling)
    original = model.generate
    captured = []
    effective = []
    original_prepare = model._prepare_generation_config
    def prepare(*a, **kw):
        config, rest = original_prepare(*a, **kw)
        effective.append({key: getattr(config, key) for key in
                          ("temperature", "top_p", "top_k", "repetition_penalty")})
        return config, rest
    model._prepare_generation_config = prepare
    def generate(**kwargs):
        result = original(**kwargs)
        captured.append(result)
        return result
    model.generate = generate
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    sample = policy.sample(prompt, sampling=sampling)
    assert effective == [{"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}]
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    result = captured.pop()
    assert sample.token_ids == result.sequences[0, len(prompt):].tolist()
    expected = model.compute_transition_scores(result.sequences, result.scores, normalize_logits=True)[0].float()
    assert torch.allclose(torch.tensor(sample.logprobs, device="cuda"), expected)
    assert torch.isfinite(expected).all()
    report["generation"] = {"prompt_tokens": len(prompt), "sample": asdict(sample),
        "effective_sampling": effective[0],
        "eos": sample.token_ids[-1] == tokenizer.eos_token_id, "seconds": seconds,
        "tokens_per_second": len(sample.token_ids) / seconds, **memory()}
    model.generate = original
    model._prepare_generation_config = original_prepare
    del result, expected, policy
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.0, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    before = parameter_hash(model)
    model.train()
    batch = tokenizer("Thought: buy shoes. Action: search[blue shoes]", return_tensors="pt").to("cuda")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    torch.cuda.reset_peak_memory_stats()
    loss = model(**batch, labels=batch["input_ids"]).loss
    assert torch.isfinite(loss)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    assert torch.isfinite(norm) and norm > 0
    optimizer.step()
    after = parameter_hash(model)
    assert before != after
    report["lora"] = {"r": 8, "alpha": 16, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "dropout": 0, "trainable_params": trainable, "trainable_percent": trainable / report["params"] * 100,
        "loss": loss.item(), "grad_norm": norm.item(), "before_hash": before, "after_hash": after, **memory()}
    return report


def checkpoint_callback(trainer, expected_hash=None, stop_after_one=False):
    from transformers import TrainerCallback
    class Check(TrainerCallback):
        restored = None
        def on_train_begin(self, args, state, control, **kwargs):
            if expected_hash is not None:
                assert state.global_step == 1 and parameter_hash(trainer.model) == expected_hash
                assert trainer.optimizer.state and trainer.lr_scheduler.last_epoch == 1
                self.restored = {"step": state.global_step, "hash": expected_hash, "optimizer_scheduler": True}
        def on_step_end(self, args, state, control, **kwargs):
            if stop_after_one:
                control.should_training_stop = True
    callback = Check()
    trainer.add_callback(callback)
    return callback


def sft_stage(args, tokenizer, resume=False):
    import torch
    from datasets import Dataset
    from training.sft_data import tokenize_with_assistant_mask
    from training.sft import SFTConfigSpec, build_sft_trainer, train_sft
    from training.runtime import prepare_run, json_hash
    root = args.output_root / "sft"
    # Two bounded lengths, both through the real formatter. Long fixture is conditioning, not extra targets.
    rows = []
    for repetitions in (1, 300):
        messages = [{"role": "system", "content": "Shop."},
                    {"role": "user", "content": "Visible blue shoe product. " * repetitions},
                    {"role": "assistant", "content": "Thought: choose blue shoes.\nAction: search[blue shoes]"},
                    {"role": "user", "content": "terminal omitted"}]
        rows.append(tokenize_with_assistant_mask(tokenizer, messages, max_length=4096))
    spec = SFTConfigSpec(output_dir=root, num_train_epochs=1, effective_batch_size=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=1, lora_r=8, lora_alpha=16,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"), lora_dropout=0,
        gradient_checkpointing=True, save_steps=1, logging_steps=1)
    checkpoint = root / "checkpoints/checkpoint-1"
    prepare_run(root, config={"mode": "sft", "scenario": "single", "test_only": True},
                inputs={"fixture_hash": json_hash(rows)}, resume_from_checkpoint=checkpoint if resume else None)
    model = load_model(args.model_path)
    trainer = build_sft_trainer(model=model, tokenizer=tokenizer, train_dataset=Dataset.from_list(rows), config=spec)
    for row in rows:
        assert trainer.data_collator([row])["labels"][0].tolist() == row["labels"]
        assert row["input_ids"][-1] == row["labels"][-1] == tokenizer.eos_token_id
        assert "terminal omitted" not in tokenizer.decode(row["input_ids"])
    before = parameter_hash(trainer.model)
    expected = json.loads((args.output_root / "sft.json").read_text())["after_hash"] if resume else None
    check = checkpoint_callback(trainer, expected, stop_after_one=not resume)
    training_steps = []
    original_step = trainer.training_step
    def training_step(model, inputs, *a, **kw):
        torch.cuda.synchronize()
        started = time.monotonic()
        value = original_step(model, inputs, *a, **kw)
        torch.cuda.synchronize()
        training_steps.append({"sequence_length": inputs["input_ids"].shape[1],
                               "forward_backward_s": time.monotonic() - started, **memory()})
        return value
    trainer.training_step = training_step
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    train_sft(trainer, resume_from_checkpoint=checkpoint if resume else None)
    seconds = time.monotonic() - start
    after = parameter_hash(trainer.model)
    assert after != (expected or before)
    assert trainer.state.global_step == (2 if resume else 1)
    logs = [log for log in trainer.state.log_history if "loss" in log]
    assert logs and all(torch.isfinite(torch.tensor([log["loss"], log["grad_norm"]])).all()
                        and log["grad_norm"] > 0 for log in logs)
    assert (checkpoint / "optimizer.pt").is_file() and (checkpoint / "scheduler.pt").is_file()
    return {"sequence_lengths": [len(r["input_ids"]) for r in rows],
        "trainable_tokens": [sum(label != -100 for label in r["labels"]) for r in rows],
        "microbatch": 1, "accumulation": 1,
        "gradient_checkpointing": True, "logs": logs, "seconds_including_checkpoint": seconds,
        "before_hash": before, "after_hash": after, "restored": check.restored,
        "global_step": trainer.state.global_step, "training_steps": training_steps, **memory()}


def grpo_stage(args, tokenizer, resume=False, real=False):
    import torch
    from datasets import Dataset
    from training.grpo import build_grpo_trainer, load_grpo_config, task_dataset_rows, train_grpo
    from training.runtime import prepare_run, json_hash
    from env.teacher_env_client import TeacherEnvClient
    from tests.training.test_rollout import FakeEnv, step_payload
    config = load_grpo_config()
    root = args.output_root / ("real-grpo" if real else "grpo")
    config.update(scenario="single", output_dir=str(root), model_path=args.model_path, max_action_steps=2)
    config["grpo"].update(num_generations=2, task_groups_per_update=1, max_steps=2 if not real else 1,
        per_device_train_batch_size=1, lora_r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], save_steps=1)
    config["sampling"].update(max_new_tokens=128 if real else 40, max_context_tokens=8192,
                              chat_template_kwargs={"enable_thinking": False})
    schedule = [args.task_id] if real else ["TEST_A", "TEST_B"]
    checkpoint = root / "checkpoints/checkpoint-1"
    prepare_run(root, config=config, inputs={"schedule_sha256": json_hash(schedule), "test_only": True},
                resume_from_checkpoint=checkpoint if resume else None)
    envs, environment_times, generation_times = [], [], []
    class TimedEnv(TeacherEnvClient):
        def _request(self, *a, **kw):
            result = super()._request(*a, **kw)
            environment_times.append(result.latency_s)
            return result
    def factory():
        if real:
            env = TimedEnv(args.endpoint)
        else:
            env = FakeEnv([step_payload("EXTERNAL_OBSERVATION_O1"),
                           step_payload("TERMINAL_OMITTED", done=True, success=len(envs) % 2 == 0)])
        envs.append(env)
        return env
    model = load_model(args.model_path)
    trainer = build_grpo_trainer(model=model, tokenizer=tokenizer,
        dataset=Dataset.from_list(task_dataset_rows(schedule, "single")), config=config, env_factory=factory)
    if not real:
        original_generate = trainer.model.generate
        def generate(**kwargs):
            env = envs[-1]
            action = "search[shoes]" if not env.responses else "click[buy now]"
            word = "yes" if len(envs) % 2 else "no"
            candidates = [tokenizer(f"Thought: {word} {s}\nAction: {action}", add_special_tokens=False)["input_ids"]
                          + [tokenizer.eos_token_id] for s in ("a", "b")]
            length = kwargs["input_ids"].shape[1]
            def allowed(batch, sequence):
                prefix = sequence[length:].tolist()
                return sorted({c[len(prefix)] for c in candidates if c[:len(prefix)] == prefix and len(c) > len(prefix)})
            return original_generate(**kwargs, prefix_allowed_tokens_fn=allowed)
        trainer.model.generate = generate
    original_timed_generate = trainer.model.generate
    def timed_generate(**kwargs):
        torch.cuda.synchronize()
        started = time.monotonic()
        result = original_timed_generate(**kwargs)
        torch.cuda.synchronize()
        generation_times.append({"seconds": time.monotonic() - started,
                                  "tokens": result.sequences.shape[1] - kwargs["input_ids"].shape[1]})
        return result
    trainer.model.generate = timed_generate
    captured, gradient_checks = [], []
    original_rollout = trainer.rollout_func
    def rollout(prompts, current):
        assert all(p == [{"role": "user", "content": ""}] for p in prompts)
        metadata = list(current._shop_batch)
        result = original_rollout(prompts, current)
        for ids, logps, mask in zip(result["completion_ids"], result["logprobs"], result["env_mask"]):
            assert len(ids) == len(logps) == len(mask) and torch.isfinite(torch.tensor(logps)).all()
            if not real:
                assert 0 in mask and "TERMINAL_OMITTED" not in tokenizer.decode(ids)
        captured.append({"metadata": metadata, "empty_transport_prompts": True,
                         "rewards": result["rollout_reward"], "statuses": result["rollout_status"],
                         "token_lengths": [len(x) for x in result["completion_ids"]],
                         "policy_tokens": [sum(x) for x in result["env_mask"]],
                         "model_hash": parameter_hash(current.model), "result": result})
        return result
    trainer.rollout_func = rollout
    original_score = trainer._generate_and_score_completions
    def score(inputs):
        result = original_score(inputs)
        assert "old_per_token_logps" not in result
        source = captured[-1]["result"]
        for i, mask in enumerate(source["env_mask"]):
            assert result["tool_mask"][i, :len(mask)].tolist() == mask
            assert result["completion_mask"][i, :len(mask)].all()
        captured[-1]["advantages"] = result["advantages"].tolist()
        return result
    trainer._generate_and_score_completions = score
    original_loss = trainer._compute_loss
    original_logps = trainer._get_per_token_logps_and_entropies
    active = []
    def logps(*a, **kw):
        result = original_logps(*a, **kw)
        if result[0].requires_grad:
            own = active[-1].bool().clone()
            def grad_check(grad):
                assert torch.all(grad[~own] == 0)
                if not real:
                    assert torch.any(grad[own] != 0)
                gradient_checks.append(True)
            result[0].register_hook(grad_check)
        return result
    trainer._get_per_token_logps_and_entropies = logps
    def loss(model, inputs):
        active.append(inputs["completion_mask"] * inputs["tool_mask"])
        value = original_loss(model, inputs)
        assert torch.isfinite(value)
        active.pop()
        return value
    trainer._compute_loss = loss
    before = parameter_hash(trainer.model)
    expected = json.loads((args.output_root / "grpo.json").read_text())["after_hash"] if resume else None
    check = checkpoint_callback(trainer, expected, stop_after_one=not resume)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    train_grpo(trainer, resume_from_checkpoint=str(checkpoint) if resume else None)
    seconds = time.monotonic() - start
    after = parameter_hash(trainer.model)
    if not real:
        assert after != (expected or before) and gradient_checks
        logs = [row for row in trainer.state.log_history if "grad_norm" in row]
        assert logs and all(torch.isfinite(torch.tensor(row["grad_norm"]))
                            and row["grad_norm"] > 0 for row in logs)
    if resume:
        assert captured[0]["model_hash"] == expected
    assert (checkpoint / "optimizer.pt").is_file() and (checkpoint / "scheduler.pt").is_file()
    for batch in captured:
        batch.pop("result")
    return {"G": 2, "microbatch": 1, "accumulation": 2, "max_action_steps_fixture": 2,
            "batches": captured, "gradient_mask_checks": len(gradient_checks), "before_hash": before,
            "after_hash": after, "restored": check.restored, "seconds_including_checkpoint": seconds,
            "logs": trainer.state.log_history, "global_step": trainer.state.global_step,
            "project_metrics": [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()],
            "generation_calls": generation_times, "environment_wait_s": sum(environment_times),
            "real_update": ("INCONCLUSIVE_DUE_TO_REWARD_VARIANCE" if real and all(
                len(set(b["rewards"])) == 1 for b in captured) else "PASS"), **memory()}


def real_rollout_stage(args, tokenizer):
    from env.teacher_env_client import TeacherEnvClient
    from training.policy import GenerationConfig, QwenPolicy
    from training.rollout import AgentRollout
    model = load_model(args.model_path).eval()
    policy = QwenPolicy(model=model, tokenizer=tokenizer)
    times, sampled = [], []
    original_sample = policy.sample
    def sample(inputs, *, sampling):
        result = original_sample(inputs, sampling=sampling)
        sampled.append((list(inputs), result))
        return result
    policy.sample = sample
    class TimedEnv(TeacherEnvClient):
        def _request(self, *a, **kw):
            result = super()._request(*a, **kw)
            times.append(result.latency_s)
            return result
    sampling = GenerationConfig(do_sample=True, max_new_tokens=256, max_context_tokens=8192,
                                chat_template_kwargs={"enable_thinking": False})
    result = AgentRollout(policy=policy, scenario="single", env_factory=lambda: TimedEnv(args.endpoint),
                          max_action_steps=2).run(args.task_id, sampling=sampling)
    trace = result.token_trace
    assert len(trace.completion_ids) == len(trace.logprobs) == len(trace.env_mask)
    stream = trace.prompt_ids + trace.completion_ids
    for inputs, sample in sampled:
        assert stream[:len(inputs)] == inputs
        offset = len(inputs) - len(trace.prompt_ids)
        end = offset + len(sample.token_ids)
        assert trace.completion_ids[offset:end] == sample.token_ids
        assert trace.logprobs[offset:end] == sample.logprobs
        assert trace.env_mask[offset:end] == [1] * len(sample.token_ids)
    assert trace.completion_ids[-len(sampled[-1][1].token_ids):] == sampled[-1][1].token_ids
    return {"task_id": result.task_id, "status": result.status, "steps": result.steps,
            "actions": result.actions, "metrics": result.reward_metrics, "trace": asdict(trace),
            "generation_s": result.generation_time_s, "environment_s": sum(times),
            "total_s": result.wall_time_s, "policy_tokens": sum(trace.env_mask),
            "external_tokens": len(trace.env_mask) - sum(trace.env_mask),
            "generation_prefix_lengths": [len(inputs) for inputs, _ in sampled],
            "exact_conditioning_prefixes": True, "no_trailing_external_tokens": True, **memory()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("gate", "base", "sft", "sft-resume", "grpo",
                                                          "grpo-resume", "real-rollout", "real-grpo"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--task-id")
    args = parser.parse_args(argv)
    if args.stage.startswith("real-") and not (args.endpoint and args.task_id):
        parser.error("real smoke requires an explicit existing endpoint and training task ID")
    args.output_root.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import AutoTokenizer, set_seed
    torch.set_num_threads(2)
    set_seed(17)
    gate = runtime_gate()
    if args.stage == "gate":
        result = gate
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=True)
        if args.stage == "base": result = base_stage(args, tokenizer)
        elif args.stage.startswith("sft"): result = sft_stage(args, tokenizer, resume=args.stage.endswith("resume"))
        elif args.stage == "real-rollout": result = real_rollout_stage(args, tokenizer)
        else: result = grpo_stage(args, tokenizer, resume=args.stage.endswith("resume"), real=args.stage == "real-grpo")
    result.update(test_only=True, stage=args.stage,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output_root / f"{args.stage}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"stage": args.stage, "status": "PASS", "report": str(args.output_root / f"{args.stage}.json")}), flush=True)


if __name__ == "__main__":
    main()

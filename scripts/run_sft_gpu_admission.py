#!/usr/bin/env python3
"""Bounded, fail-fast SFT GPU admission for single_persona.

This is not formal SFT. It executes one real 32-example optimizer update and
stores only a test-only checkpoint/report.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_RUNS = (ROOT.parent / "runs").resolve()
DATASET_VERSION = "single_persona_budgeted_v1"
DATASET_SHA256 = "e61bb4515d07da9a1c589454b9bfeefd31c57ed306675ac809998c1040e6646c"
SANITIZER_VERSION = "persona-policy-sanitizer-v1"
SELECTION_SHA256 = "3e68e4aad2ebd1c7a0f236c3d19d6a390792cbf3a34d8059740eb12d8544af72"
FORMAL_TOTAL_UPDATES = 336  # ceil(2659 / effective_batch=32) * 4 epochs
PINNED = {"torch": "2.8.0", "transformers": "4.57.6", "trl": "1.12.0",
          "peft": "0.19.1", "accelerate": "1.15.0", "datasets": "4.8.5"}
MODEL_METADATA_SHA256 = {
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "model.safetensors.index.json": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
}


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_dependency_version(package: str, actual: str, expected: str) -> None:
    # CUDA PyTorch wheels use a PEP 440 local suffix (for example +cu128).
    # The frozen API/runtime contract pins the upstream release 2.8.0 while
    # retaining the CUDA build identifier as report provenance.
    normalized = actual.split("+", 1)[0] if package == "torch" else actual
    if normalized != expected:
        raise RuntimeError(f"PINNED_DEPENDENCY_MISMATCH:{package}:{actual}")


def phase_plan(selection: dict) -> dict[str, list[str]]:
    rows = selection["examples"]
    longest = [row for row in rows if "longest" in row["selection_roles"]]
    stratified = [row for row in rows if any(x in row["selection_roles"] for x in ("p50", "p90", "p99"))]
    if (len(longest) != 1 or len(rows) != 32 or len({row["accepted_id"] for row in rows}) != 32
            or any(sum(label in row["selection_roles"] for row in rows) != 1
                   for label in ("p50", "p90", "p99"))):
        raise ValueError("selection must contain one longest and p50/p90/p99 strata")
    return {"A": [longest[0]["accepted_id"]],
            "B": [row["accepted_id"] for row in stratified],
            "C": [row["accepted_id"] for row in rows]}


def validate_cuda(torch_module) -> None:
    count = torch_module.cuda.device_count()
    if count != 1:
        raise RuntimeError("USER_GPU_NOT_AVAILABLE" if count == 0 else "EXPECTED_EXACTLY_ONE_GPU")
    if not torch_module.cuda.is_available():
        raise RuntimeError("USER_GPU_NOT_AVAILABLE")
    if not torch_module.cuda.is_bf16_supported():
        raise RuntimeError("BF16_UNSUPPORTED")


def run_phase_sequence(run_a, run_b, run_c) -> dict:
    """Execute admission phases in fail-closed order; exceptions stop the chain."""
    result = {"phase_a": run_a()}
    result["phase_b"] = run_b()
    result["phase_c"] = run_c()
    return result


def validate_selection_payload(selection: dict) -> None:
    payload = dict(selection)
    actual = payload.pop("selection_sha256", None)
    if actual != SELECTION_SHA256 or canonical_hash(payload) != actual:
        raise ValueError("ADMISSION_SELECTION_HASH_MISMATCH")
    if payload.get("dataset_version") != DATASET_VERSION or payload.get("count") != 32:
        raise ValueError("ADMISSION_SELECTION_IDENTITY_MISMATCH")
    if payload.get("persona_sanitizer_version") != SANITIZER_VERSION:
        raise ValueError("PERSONA_SANITIZER_VERSION_MISMATCH")
    phase_plan(selection)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _hash_lora(model) -> str:
    import torch
    digest = hashlib.sha256()
    found = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_" in name.lower():
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode()); digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes()); found += 1
    if not found:
        raise RuntimeError("NO_TRAINABLE_LORA_PARAMETERS")
    return digest.hexdigest()


def _memory(torch) -> dict[str, int]:
    return {"allocated": int(torch.cuda.max_memory_allocated()),
            "reserved": int(torch.cuda.max_memory_reserved())}


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def validate_output_path(output: Path) -> None:
    if output.resolve().parent != RUNTIME_RUNS or output.resolve() == RUNTIME_RUNS:
        raise RuntimeError("UNSAFE_ADMISSION_OUTPUT_PATH")


def validate_dataset_identity(manifest: dict, train_jsonl: Path) -> None:
    if (manifest.get("dataset_version"), manifest.get("frozen_count"),
            manifest.get("dataset_content_sha256")) != (DATASET_VERSION, 2659, DATASET_SHA256):
        raise RuntimeError("FROZEN_DATASET_IDENTITY_MISMATCH")
    if _sha256(train_jsonl) != DATASET_SHA256:
        raise RuntimeError("FROZEN_DATASET_CONTENT_HASH_MISMATCH")
    if manifest.get("persona_sanitizer_version") != SANITIZER_VERSION:
        raise RuntimeError("PERSONA_SANITIZER_VERSION_MISMATCH")


def validate_success_report(report: dict) -> None:
    required = {"status", "dataset_version", "dataset_sha256", "selection_sha256",
                "sanitizer_version", "candidate_runtime", "phase_a", "phase_b", "phase_c", "checkpoint"}
    if report.get("status") != "PASS" or not required.issubset(report):
        raise RuntimeError("ADMISSION_REPORT_SCHEMA_INCOMPLETE")
    if (report["phase_c"].get("examples_processed"), report["phase_c"].get("optimizer_step"),
            report["phase_c"].get("effective_batch"), report["phase_c"].get("lora_changed")) != (32, 1, 32, True):
        raise RuntimeError("ADMISSION_REPORT_PHASE_C_INCOMPLETE")


def _preflight(args, dataset: Path, selection: Path, output: Path) -> dict:
    validate_output_path(output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("OUTPUT_DIR_NOT_EMPTY_REFUSING_OVERWRITE")
    for path in (dataset / "manifest.json", dataset / "train.jsonl", selection):
        if not path.is_file():
            raise RuntimeError(f"MISSING_REQUIRED_ARTIFACT:{path}")
    manifest = json.loads((dataset / "manifest.json").read_text())
    validate_dataset_identity(manifest, dataset / "train.jsonl")
    selected = json.loads(selection.read_text())
    validate_selection_payload(selected)
    accepted_root = (ROOT / "data/teacher_raw/single_persona" /
                     str(manifest["source_run_id"]) / "accepted").resolve()
    for row in selected["examples"]:
        source = (accepted_root / row["source_path"]).resolve()
        if source.parent != accepted_root or not source.is_file() or _sha256(source) != row["source_sha256"]:
            raise RuntimeError(f"SOURCE_ARTIFACT_HASH_MISMATCH:{row['accepted_id']}")
    dependencies = {}
    for package, expected in PINNED.items():
        actual = importlib.metadata.version(package)
        validate_dependency_version(package, actual, expected)
        dependencies[package] = actual
    model_root = Path(args.model_path).resolve()
    for name, expected in MODEL_METADATA_SHA256.items():
        path = model_root / name
        if not path.is_file():
            raise RuntimeError(f"MODEL_METADATA_MISSING:{name}")
        if _sha256(path) != expected:
            raise RuntimeError(f"MODEL_IDENTITY_MISMATCH:{name}")
    index = json.loads((model_root / "model.safetensors.index.json").read_text())
    if not all((model_root / name).is_file() for name in set(index.get("weight_map", {}).values())):
        raise RuntimeError("MODEL_WEIGHT_SHARD_MISSING")
    import torch
    validate_cuda(torch)
    disk_root = output.parent if output.parent.exists() else ROOT
    if shutil.disk_usage(disk_root).free < 10 * 1024**3:
        raise RuntimeError("INSUFFICIENT_DISK_SPACE")
    return {"manifest": manifest, "selection": selected, "accepted_root": accepted_root,
            "dependencies": dependencies}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default=str(ROOT / "data/sft_frozen" / DATASET_VERSION))
    parser.add_argument("--selection", default=str(ROOT / "configs/admission/single_persona_sft_gpu_admission_selection.json"))
    parser.add_argument("--output-dir", default=str(RUNTIME_RUNS / "sft-gpu-admission-single-persona-v1"))
    args = parser.parse_args(argv)
    dataset, selection, output = map(lambda x: Path(x).resolve(), (args.dataset, args.selection, args.output_dir))
    current_phase = "model_load"
    try:
        context = _preflight(args, dataset, selection, output)
    except Exception as exc:
        try:
            validate_output_path(output)
            if not output.exists():
                output.mkdir(parents=True)
                _write_report(output / "sft_gpu_admission.json", {
                    "status": "FAIL", "failure_phase": "preflight", "failure_reason": str(exc)})
        except Exception:
            pass
        print(f"SFT_GPU_ADMISSION: FAIL\nFAILURE_PHASE: preflight\nFAILURE_REASON: {exc}")
        return 2
    output.mkdir(parents=True)
    print("SFT_GPU_ADMISSION_PREFLIGHT: PASS", flush=True)
    report = {"status": "running", "dataset_version": DATASET_VERSION,
              "dataset_sha256": DATASET_SHA256, "selection_sha256": SELECTION_SHA256,
              "sanitizer_version": SANITIZER_VERSION, "dependencies": context["dependencies"],
              "attention_backend": "sdpa",
              "candidate_runtime": {
                  "microbatch": 1, "gradient_accumulation": 32, "effective_batch": 32}}
    report_path = output / "sft_gpu_admission.json"
    _write_report(report_path, report)
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        sys.path.insert(0, str(ROOT / "src"))
        from training.sft_data import project_messages, tokenize_with_assistant_mask
        from training.sft import SFT_ATTENTION_BACKEND, SFTConfigSpec, build_sft_trainer
        set_seed(1)
        print("SFT_GPU_ADMISSION_MODEL_LOADING: START", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=True)
        rows = []
        by_id = {row["accepted_id"]: row for row in context["selection"]["examples"]}
        for accepted_id in [x for phase in ("A", "B", "C") for x in phase_plan(context["selection"])[phase]]:
            source = context["accepted_root"] / by_id[accepted_id]["source_path"]
            record = json.loads(source.read_text())
            encoded = tokenize_with_assistant_mask(tokenizer, project_messages(record), max_length=32768)
            rows.append({"accepted_id": accepted_id, **encoded})
        unique = {row["accepted_id"]: row for row in rows}
        # Qwen3 eager attention materializes an O(sequence_length^2) score
        # tensor and OOMs at the real 29,741-token sample on 96 GB. SDPA
        # preserves attention semantics while selecting a memory-efficient
        # PyTorch CUDA kernel; this changes execution only.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16, attn_implementation=SFT_ATTENTION_BACKEND,
            local_files_only=True)
        spec = SFTConfigSpec(output_dir=output, num_train_epochs=4, learning_rate=1e-5,
                             effective_batch_size=32, per_device_train_batch_size=1,
                             gradient_accumulation_steps=32, max_length=32768, lora_r=8,
                             lora_alpha=16, target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                             lora_dropout=0.0, gradient_checkpointing=True, save_steps=1, logging_steps=1)
        trainer = build_sft_trainer(model=model, tokenizer=tokenizer,
                                    train_dataset=Dataset.from_list([{k: v for k, v in row.items() if k != "accepted_id"}
                                                                      for row in unique.values()]), config=spec)
        model = trainer.model
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        def run_one(accepted_id: str, phase: str) -> dict:
            torch.cuda.reset_peak_memory_stats(); model.zero_grad(set_to_none=True)
            row = unique[accepted_id]
            batch = trainer.data_collator([{k: v for k, v in row.items() if k != "accepted_id"}])
            batch = {key: value.to(model.device) for key, value in batch.items()}
            started = time.monotonic(); forward_start = time.monotonic()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
            forward = time.monotonic() - forward_start
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError("NONFINITE_LOSS")
            backward_start = time.monotonic(); loss.backward()
            backward = time.monotonic() - backward_start
            finite_grad = all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                               for parameter in model.parameters())
            if not finite_grad:
                raise RuntimeError("NONFINITE_GRADIENT")
            item = {"phase": phase, "accepted_id": accepted_id,
                    "total_tokens": int(by_id[accepted_id]["total_tokens"]),
                    "model_input_tokens": len(row["input_ids"]),
                    "supervised_tokens": int(sum(x != -100 for x in row["labels"])),
                    "loss": float(loss.detach().cpu()), "grad_finite": finite_grad,
                    "forward_seconds": forward, "backward_seconds": backward,
                    "phase_seconds": time.monotonic() - started, **_memory(torch)}
            model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            return item
        phases = {"A": [phase_plan(context["selection"])["A"][0]],
                  "B": phase_plan(context["selection"])["B"]}
        current_phase = "A"
        print("SFT_GPU_ADMISSION_PHASE_A: START", flush=True)
        report["phase_a"] = run_one(phases["A"][0], "A")
        _write_report(report_path, report)
        print("SFT_GPU_ADMISSION_PHASE_A: PASS", flush=True)
        current_phase = "B"
        print("SFT_GPU_ADMISSION_PHASE_B: START", flush=True)
        report["phase_b"] = [run_one(accepted_id, "B") for accepted_id in phases["B"]]
        _write_report(report_path, report)
        print("SFT_GPU_ADMISSION_PHASE_B: PASS", flush=True)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5,
                                      betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: max(0.0, 1.0 - step / FORMAL_TOTAL_UPDATES))
        current_phase = "C"
        print("SFT_GPU_ADMISSION_PHASE_C: START", flush=True)
        torch.cuda.reset_peak_memory_stats()
        before = _hash_lora(model); started = time.monotonic(); total_tokens = model_input_tokens = total_supervised = 0
        micro_losses = []
        optimizer.zero_grad(set_to_none=True)
        for row in [unique[x["accepted_id"]] for x in context["selection"]["examples"]]:
            batch = trainer.data_collator([{k: v for k, v in row.items() if k != "accepted_id"}])
            batch = {key: value.to(model.device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            if not torch.isfinite(loss): raise RuntimeError("NONFINITE_LOSS")
            micro_losses.append(float(loss.detach().cpu()))
            (loss / 32).backward()
            total_tokens += int(by_id[row["accepted_id"]]["total_tokens"])
            model_input_tokens += len(row["input_ids"])
            total_supervised += sum(x != -100 for x in row["labels"])
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads): raise RuntimeError("NONFINITE_GRADIENT")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not _finite(grad_norm): raise RuntimeError("NONFINITE_GRAD_NORM")
        used_lr = optimizer.param_groups[0]["lr"]; optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        after = _hash_lora(model)
        if before == after: raise RuntimeError("LORA_PARAMETERS_DID_NOT_CHANGE")
        checkpoint = output / "checkpoint-1"; checkpoint.mkdir()
        model.save_pretrained(checkpoint / "adapter"); tokenizer.save_pretrained(checkpoint / "tokenizer")
        torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt"); torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1, "effective_batch": 32}, indent=2) + "\n")
        (checkpoint / "admission_lineage.json").write_text(json.dumps({
            "dataset_version": DATASET_VERSION, "dataset_sha256": DATASET_SHA256,
            "selection_sha256": SELECTION_SHA256, "sanitizer_version": SANITIZER_VERSION,
            "model_path": str(Path(args.model_path).resolve()), "candidate_runtime": report["candidate_runtime"],
        }, indent=2) + "\n")
        report["phase_c"] = {"examples_processed": 32, "total_tokens": total_tokens,
                              "model_input_tokens": model_input_tokens,
                              "total_supervised_tokens": total_supervised, "effective_batch": 32,
                              "optimizer_step": 1, "loss_finite": True, "grad_finite": True,
                              "loss": sum(micro_losses) / len(micro_losses),
                              "grad_norm": grad_norm, "learning_rate_used": used_lr,
                              "learning_rate_after_scheduler_step": optimizer.param_groups[0]["lr"],
                              "scheduler_horizon_updates": FORMAL_TOTAL_UPDATES,
                              "wall_seconds": time.monotonic() - started,
                              "lora_changed": True, "lora_sha256_before": before,
                              "lora_sha256_after": after, **_memory(torch)}
        report.update({"status": "PASS", "checkpoint": str(checkpoint)})
        validate_success_report(report)
        _write_report(report_path, report)
        print("SFT_GPU_ADMISSION: PASS")
        print(json.dumps({"longest_tokens": report["phase_a"]["total_tokens"],
                          "longest_peak_allocated": report["phase_a"]["allocated"],
                          "longest_peak_reserved": report["phase_a"]["reserved"],
                          "update_wall_seconds": report["phase_c"]["wall_seconds"],
                          "effective_batch": 32, "loss_finite": True,
                          "grad_norm": report["phase_c"]["grad_norm"],
                          "lora_changed": True, "checkpoint": str(checkpoint)}, indent=2))
        return 0
    except Exception as exc:
        report.update({"status": "FAIL", "failure_phase": current_phase, "failure_reason": str(exc)})
        _write_report(report_path, report)
        print(f"SFT_GPU_ADMISSION: FAIL\nFAILURE_PHASE: {current_phase}\nFAILURE_REASON: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

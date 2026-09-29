#!/usr/bin/env python3
"""Create the deterministic, test-only SFT GPU admission selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_VERSION = "single_persona_budgeted_v1"
DATASET_SHA256 = "e61bb4515d07da9a1c589454b9bfeefd31c57ed306675ac809998c1040e6646c"
SANITIZER_VERSION = "persona-policy-sanitizer-v1"


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def select(rows: list[dict], *, count: int = 32) -> list[dict]:
    if len(rows) < count:
        raise ValueError("frozen dataset has fewer than 32 examples")
    ordered = sorted(rows, key=lambda row: (int(row["total_tokens"]), row["accepted_id"]))
    chosen: dict[str, dict] = {}

    def add(row: dict, phase: str) -> None:
        chosen.setdefault(row["accepted_id"], {**row, "selection_roles": []})
        if phase not in chosen[row["accepted_id"]]["selection_roles"]:
            chosen[row["accepted_id"]]["selection_roles"].append(phase)

    add(ordered[-1], "longest")
    for target, label in ((0.50, "p50"), (0.90, "p90"), (0.99, "p99")):
        add(ordered[round((len(ordered) - 1) * target)], label)
    # Fill the remaining slots at fixed quantiles. This gives broad length
    # coverage while remaining independent of filesystem ordering.
    for index in range(1, count):
        if len(chosen) >= count:
            break
        position = round((len(ordered) - 1) * (index / count))
        add(ordered[position], "stratified")
    for row in ordered:
        if len(chosen) >= count:
            break
        add(row, "stratified")
    if len(chosen) != count:
        raise AssertionError(f"selection produced {len(chosen)} rows, expected {count}")
    result = list(chosen.values())
    result.sort(key=lambda row: (row["selection_roles"][0], int(row["total_tokens"]), row["accepted_id"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "data/sft_frozen" / DATASET_VERSION))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default=str(ROOT / "configs/admission/single_persona_sft_gpu_admission_selection.json"))
    args = parser.parse_args()
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest.get("dataset_version") != DATASET_VERSION or manifest.get("dataset_content_sha256") != DATASET_SHA256:
        raise SystemExit("FROZEN_DATASET_IDENTITY_MISMATCH")
    if hashlib.sha256((dataset / "train.jsonl").read_bytes()).hexdigest() != DATASET_SHA256:
        raise SystemExit("FROZEN_DATASET_CONTENT_HASH_MISMATCH")
    if manifest.get("persona_sanitizer_version") != SANITIZER_VERSION:
        raise SystemExit("PERSONA_SANITIZER_VERSION_MISMATCH")
    source = json.loads((dataset / "selection_manifest.json").read_text())
    lengths = json.loads((dataset / "token_length_stats.json").read_text())
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from training.sft_data import load_selected_examples, tokenize_with_assistant_mask
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=True)
    accepted_root = (ROOT / "data/teacher_raw/single_persona" /
                     str(manifest["source_run_id"]) / "accepted")
    examples = load_selected_examples(dataset / "selection_manifest.json",
                                      accepted_root=accepted_root,
                                      scenario="single_persona")
    by_id = {}
    for example in examples:
        rendered = tokenizer.apply_chat_template(example.messages, tokenize=True, add_generation_prompt=False)
        encoded = tokenize_with_assistant_mask(tokenizer, example.messages, max_length=32768)
        by_id[example.accepted_id] = {"total_tokens": len(rendered),
                                      "supervised_tokens": sum(x != -100 for x in encoded["labels"])}
    rows = []
    for item in source["artifacts"]:
        row = by_id.get(item.get("accepted_id"))
        if row is None:
            raise SystemExit("TOKEN_LENGTH_TABLE_MISSING_ARTIFACT")
        rows.append({"accepted_id": item["accepted_id"], "task_id": item["task_id"],
                     "source_path": item["path"], "source_sha256": item["sha256"],
                     "total_tokens": int(row["total_tokens"]),
                     "supervised_tokens": int(row["supervised_tokens"])})
    selected = select(rows)
    payload = {"dataset_version": DATASET_VERSION, "scenario": "single_persona",
               "persona_sanitizer_version": SANITIZER_VERSION, "count": len(selected),
               "examples": selected}
    payload["selection_sha256"] = canonical_hash(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        old = json.loads(output.read_text())
        if old != payload:
            raise FileExistsError(f"refusing to overwrite selection: {output}")
    else:
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selection_sha256": payload["selection_sha256"],
                      "count": len(selected), "longest": max(selected, key=lambda x: x["total_tokens"])},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

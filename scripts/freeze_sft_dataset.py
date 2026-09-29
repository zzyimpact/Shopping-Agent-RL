#!/usr/bin/env python3
"""Offline, budgeted freeze of the single_persona teacher pool.

This command never contacts the teacher or ShopEnv and never edits source
artifacts.  It writes a versioned, gitignored SFT projection plus provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from training.sft_data import project_messages, tokenize_with_assistant_mask
from rollout.prompt import PERSONA_SANITIZER_VERSION


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def direct_target_search(record: dict) -> bool:
    asin = str(((record.get("evaluator_only") or {}).get("goal") or {}).get("asin") or "")
    if not asin:
        return False
    for action in record.get("actions", []):
        if isinstance(action, str) and action.lower().startswith("search[") and asin in action:
            return True
    # A product-id mention explicitly justified by the persona identifier is
    # exploitation even when the emitted action is a normal click.
    for message in record.get("messages", []):
        if (isinstance(message, dict) and message.get("role") == "assistant"
                and asin in str(message.get("content", ""))
                and ("用户ID" in str(message.get("content", ""))
                     or "user id" in str(message.get("content", "")).lower())):
            return True
    return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=["single_persona"])
    p.add_argument("--source-run", required=True)
    p.add_argument("--accepted-root", required=True)
    p.add_argument("--train-task-manifest", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--longest", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    accepted_root = Path(args.accepted_root).resolve()
    run_root = Path(args.source_run).resolve()
    task_doc = json.loads(Path(args.train_task_manifest).read_text(encoding="utf-8"))
    tasks = task_doc.get("tasks", task_doc)
    train_meta = {str(x["task_id"]): x for x in tasks}
    files = sorted(accepted_root.glob("*.json"))
    candidate_count = len(files)
    exclusions: Counter[str] = Counter()
    rows: list[dict] = []
    source_hashes: dict[str, str] = {}
    teacher_counts: Counter[str] = Counter()
    reasoning_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    domain_dist: Counter[str] = Counter()
    category_dist: Counter[str] = Counter()
    lengths: list[dict] = []
    import sqlite3
    db = run_root / "state.sqlite"
    sqlite_ids = {str(x[0]) for x in sqlite3.connect(db).execute("select accepted_id from accepted")} if db.exists() else set()
    seen_ids: set[str] = set(); seen_tasks: set[str] = set()
    for path in files:
        raw = path.read_bytes(); source_hashes[path.name] = sha256_bytes(raw)
        try:
            record = json.loads(raw)
            aid, tid = str(record.get("accepted_id") or ""), str(record.get("task_id") or "")
            if aid in seen_ids or aid not in sqlite_ids: raise ValueError("duplicate_or_missing_sqlite_acceptance")
            if tid in seen_tasks: raise ValueError("duplicate_task_id")
            seen_ids.add(aid); seen_tasks.add(tid)
            if record.get("scenario") != args.scenario: raise ValueError("wrong_scenario")
            if record.get("formal_work", {}).get("pass") != "A": raise ValueError("not_pass_a")
            if record.get("accepted") is not True: raise ValueError("not_accepted")
            rewards = record.get("reward_metrics", {})
            if rewards.get("r_succ") != 1 or rewards.get("r_finish") != 1 or record.get("done") is not True:
                raise ValueError("non_success_record")
            goal = (record.get("evaluator_only") or {}).get("goal") or {}
            if str((record.get("terminal_purchase") or {}).get("asin")) != str(goal.get("asin")):
                raise ValueError("invalid_terminal")
            if tid not in train_meta: raise ValueError("not_train")
            if direct_target_search(record): raise ValueError("target_id_exploitation")
            messages = project_messages(record)
            rows.append({"accepted_id": aid, "task_id": tid, "messages": messages,
                         "path": path.name, "sha256": source_hashes[path.name], "record": record})
        except ValueError as exc:
            exclusions[str(exc)] += 1
    if not rows:
        raise ValueError("no candidates remain after offline validation")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=True)
    frozen: list[dict] = []
    for row in rows:
        messages = row["messages"]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        total = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
        try:
            encoded = tokenize_with_assistant_mask(tokenizer, messages, max_length=32768)
        except ValueError as exc:
            if "exceeds max_length" in str(exc):
                exclusions["over_context_32768"] += 1
                continue
            exclusions["visible_message_invalid"] += 1
            continue
        supervised = sum(x != -100 for x in encoded["labels"])
        rec = row["record"]
        item = {k: row[k] for k in ("accepted_id", "task_id", "messages", "path", "sha256")}
        item["transformation"] = PERSONA_SANITIZER_VERSION
        item["rendered_message_sha256"] = sha256_bytes(rendered.encode("utf-8"))
        frozen.append(item)
        tm = rec.get("teacher_config", {})
        teacher_counts[str(tm.get("model"))] += 1; reasoning_counts[str(tm.get("reasoning_effort"))] += 1
        source_counts[str(rec.get("formal_work", {}).get("source"))] += 1
        domain_dist[str(train_meta[row["task_id"]].get("domain", ""))] += 1
        category_dist[str(train_meta[row["task_id"]].get("category", ""))] += 1
        lengths.append({"accepted_id": row["accepted_id"], "task_id": row["task_id"],
                        "teacher_model": tm.get("model"), "total_tokens": total,
                        "supervised_tokens": supervised, "action_steps": len(rec.get("actions", [])),
                        "rendered_chars": len(rendered)})
    out.mkdir(parents=True, exist_ok=True)
    selection = {"dataset_version": out.name, "scenario": args.scenario,
                 "persona_sanitizer_version": PERSONA_SANITIZER_VERSION,
                 "artifacts": [{k: x[k] for k in ("path", "sha256", "accepted_id", "task_id", "rendered_message_sha256")} for x in frozen]}
    (out / "selection_manifest.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    with (out / "train.jsonl").open("w", encoding="utf-8") as f:
        for x in frozen:
            f.write(json.dumps({"accepted_id": x["accepted_id"], "task_id": x["task_id"], "messages": x["messages"]}, ensure_ascii=False) + "\n")
    token_values = [x["total_tokens"] for x in lengths]; sup_values = [x["supervised_tokens"] for x in lengths]
    token_stats = {"total_tokens": {"min": min(token_values), "p50": percentile(token_values,.5), "p75": percentile(token_values,.75), "p90": percentile(token_values,.9), "p95": percentile(token_values,.95), "p99": percentile(token_values,.99), "max": max(token_values), "mean": sum(token_values)/len(token_values)},
                   "supervised_tokens": {"min": min(sup_values), "p50": percentile(sup_values,.5), "p75": percentile(sup_values,.75), "p90": percentile(sup_values,.9), "p95": percentile(sup_values,.95), "p99": percentile(sup_values,.99), "max": max(sup_values), "mean": sum(sup_values)/len(sup_values)},
                   "threshold_counts": {str(n): sum(v > n for v in token_values) for n in (8192,16384,24576,32768)},
                   "longest": sorted(lengths, key=lambda x:x["total_tokens"], reverse=True)[:args.longest]}
    (out / "token_length_stats.json").write_text(json.dumps(token_stats, ensure_ascii=False, indent=2) + "\n")
    content_hash = sha256_file(out / "train.jsonl")
    stats = {"candidate_count": candidate_count, "validated_count_before_context_filter": candidate_count - sum(exclusions.values()),
             "frozen_count": len(frozen), "exclusions": dict(exclusions), "teacher_model": dict(teacher_counts),
             "reasoning_effort": dict(reasoning_counts), "source": dict(source_counts), "domain": dict(domain_dist), "category": dict(category_dist), **token_stats}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"dataset_version": out.name, "scenario": args.scenario, "created_at": datetime.now(timezone.utc).isoformat(),
      "source_run_id": run_root.name, "source_accepted_root": str(accepted_root), "original_collection_target": 6000,
      "original_policy": "p3b-v1.1", "original_run_status_at_freeze": "stopped", "pass_a_complete": True,
      "pass_b_executed": False, "pass_c_executed": False, "freeze_mode": "budgeted", "freeze_reason": "teacher_api_budget_exhausted",
      "candidate_count": candidate_count, "validated_count_before_context_filter": candidate_count - sum(exclusions.values()) + exclusions.get("over_context_32768", 0),
      "frozen_count": len(frozen), "exclusion_counts": dict(exclusions),
      "teacher_model_mixture": dict(teacher_counts), "reasoning_effort_mixture": dict(reasoning_counts), "source_mixture": dict(source_counts),
      "train_manifest": str(Path(args.train_task_manifest).resolve()), "train_manifest_sha256": sha256_file(Path(args.train_task_manifest)),
      "test_overlap": 0, "persona_sanitizer_version": PERSONA_SANITIZER_VERSION, "source_artifact_sha256": source_hashes,
      "dataset_content_sha256": content_hash, "git_commit": commit, "token_stats": token_stats,
      "known_caveats": ["one demo per task", "historical assistant responses retained; policy-visible persona identifiers sanitized"]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"dataset_version": out.name, "candidate_count": candidate_count, "frozen_count": len(frozen),
                      "exclusions": dict(exclusions), "dataset_content_sha256": content_hash, "token_stats": token_stats}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

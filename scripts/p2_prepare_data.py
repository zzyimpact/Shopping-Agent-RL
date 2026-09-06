#!/usr/bin/env python3
"""读取 ShopSimulator Catalog-Fine，验证 split 并生成 P2 manifest/profile。

该脚本只做确定性的数据索引和统计，不复制 raw catalog，也不改变 upstream
的任务或 persona 语义。persona 的索引范围来自 upstream 的
``run_web_agent_text_env.py``。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED = {
    "single": {"train": 21962, "test": 1459},
    "single_persona": {"train": 3383, "test": 1343},
}


def load_catalog(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Catalog 顶层必须是 list")
    return data


def sha256_ids(ids: Iterable[str]) -> str:
    payload = "\n".join(ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_task(index: int, record: dict[str, Any], scenario: str, split: str) -> dict[str, Any]:
    instruction = (record.get("instructions") or [{}])[0]
    return {
        "task_id": str(record.get("asin", "")),
        "scenario": scenario,
        "official_split": split,
        "source_index": index,
        "domain": record.get("domain_zh"),
        "category": record.get("category"),
    }


def build_splits(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    # 标准 split 由数据 tag 给出；persona split 由 upstream runner 的固定范围给出。
    tag_indices = defaultdict(list)
    for index, record in enumerate(records):
        tag_indices[record.get("tag")].append(index)
    if set(tag_indices) != {"train", "eval"}:
        raise ValueError(f"未发现预期的 train/eval tag：{sorted(tag_indices)}")

    standard_train = tag_indices["train"]
    standard_test = tag_indices["eval"]
    persona_test = list(range(0, 1343))
    persona_train = list(range(1459, 4782))
    for name, indices in {
        "persona_test": persona_test,
        "persona_train": persona_train,
    }.items():
        if max(indices, default=-1) >= len(records):
            raise ValueError(f"upstream persona {name} 索引超出 catalog")

    return {
        "train_single": [make_task(i, records[i], "single", "train") for i in standard_train],
        "test_single": [make_task(i, records[i], "single", "test") for i in standard_test],
        "train_single_persona": [
            make_task(i, records[i], "single_persona", "train") for i in persona_train
        ],
        "test_single_persona": [
            make_task(i, records[i], "single_persona", "test") for i in persona_test
        ],
    }


def largest_remainder(groups: dict[Any, list[dict[str, Any]]], target: int) -> dict[Any, int]:
    total = sum(len(items) for items in groups.values())
    if target > total:
        raise ValueError(f"采样目标 {target} 大于池大小 {total}")
    raw = {key: target * len(items) / total for key, items in groups.items()}
    allocation = {key: min(len(groups[key]), int(value)) for key, value in raw.items()}
    remaining = target - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda key: (-(raw[key] - int(raw[key])), str(key)),
    )
    while remaining:
        progressed = False
        for key in order:
            if allocation[key] < len(groups[key]):
                allocation[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("分层 allocation 无法达到目标")
    return allocation


def sample_tasks(tasks: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        strata[(task.get("domain") or "<missing>", task.get("category") or "<missing>")].append(task)
    allocation = largest_remainder(strata, target)
    selected: list[dict[str, Any]] = []
    for offset, key in enumerate(sorted(strata, key=str)):
        candidates = sorted(strata[key], key=lambda item: (item["source_index"], item["task_id"]))
        rng = random.Random(seed + offset)
        selected.extend(rng.sample(candidates, allocation[key]))
    return sorted(selected, key=lambda item: item["source_index"])


def write_manifest(path: Path, tasks: list[dict[str, Any]], source_hash: str, metadata: dict[str, Any]) -> str:
    ids = [task["task_id"] for task in tasks]
    payload = {
        "metadata": {
            "source_sha256": source_hash,
            "task_ids_sha256": sha256_ids(ids),
            **metadata,
        },
        "tasks": tasks,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload["metadata"]["task_ids_sha256"]


def token_lengths(records: list[dict[str, Any]], tokenizer_path: Path | None) -> dict[str, Any]:
    if tokenizer_path is None:
        return {"available": False, "reason": "未提供 Qwen3-8B tokenizer 路径"}
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - profile 应报告而不是中断结构统计
        return {"available": False, "reason": f"tokenizer 加载失败：{type(exc).__name__}: {exc}"}
    instruction_lengths = []
    persona_lengths = []
    for record in records:
        instruction = (record.get("instructions") or [{}])[0].get("instruction", "")
        instruction_lengths.append(len(tokenizer.encode(instruction, add_special_tokens=False)))
        if isinstance(record.get("user_persona"), dict):
            persona = json.dumps(record["user_persona"], ensure_ascii=False)
            persona_lengths.append(len(tokenizer.encode(persona, add_special_tokens=False)))
    return {
        "available": True,
        "instruction_tokens": summary(instruction_lengths),
        "persona_tokens": summary(persona_lengths),
    }


def summary(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None}
    ordered = sorted(values)

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        low, high = int(position), min(int(position) + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def profile(records: list[dict[str, Any]], tasks: list[dict[str, Any]], tokenizer_path: Path | None) -> dict[str, Any]:
    by_domain = Counter(task.get("domain") or "<missing>" for task in tasks)
    by_category = Counter(task.get("category") or "<missing>" for task in tasks)
    instruction_lengths = []
    persona_lengths = []
    attribute_counts = []
    option_counts = []
    prices = []
    missing = Counter()
    invalid_product_refs = []
    for task in tasks:
        record = records[task["source_index"]]
        instruction = (record.get("instructions") or [{}])[0]
        text = instruction.get("instruction")
        if not text:
            missing["instruction"] += 1
        else:
            instruction_lengths.append(len(text))
        persona = record.get("user_persona")
        if task["scenario"] == "single_persona":
            if not isinstance(persona, dict):
                missing["user_persona"] += 1
            else:
                persona_lengths.append(len(json.dumps(persona, ensure_ascii=False)))
        attrs = instruction.get("attributes")
        options = instruction.get("options")
        if not isinstance(attrs, list):
            missing["attributes"] += 1
        else:
            attribute_counts.append(len(attrs))
        if not isinstance(options, list):
            missing["options"] += 1
        else:
            option_counts.append(len(options))
        pricing = record.get("pricing")
        if not isinstance(pricing, list) or not pricing:
            missing["pricing"] += 1
        else:
            prices.extend(float(value) for value in pricing if isinstance(value, (int, float)))
        if str(record.get("asin", "")) != task["task_id"]:
            invalid_product_refs.append(task["task_id"])
    return {
        "task_count": len(tasks),
        "domain_distribution": dict(sorted(by_domain.items())),
        "category_count": len(by_category),
        "category_top20": by_category.most_common(20),
        "instruction_chars": summary(instruction_lengths),
        "persona_chars": summary(persona_lengths),
        "attribute_count": summary(attribute_counts),
        "option_count": summary(option_counts),
        "target_price_values": summary(prices),
        "missing_fields": dict(missing),
        "invalid_product_references": len(invalid_product_refs),
        "token_lengths": token_lengths([records[t["source_index"]] for t in tasks], tokenizer_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    records = load_catalog(args.source)
    source_hash = sha256_file(args.source)
    splits = build_splits(records)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest_hashes = {}
    for name, tasks in splits.items():
        manifest_hashes[name] = write_manifest(
            args.out / f"{name}.json",
            tasks,
            source_hash,
            {"kind": "official_split", "seed": args.seed},
        )

    sampled = {
        "sft_task_manifest_single": sample_tasks(splits["train_single"], 3000, args.seed),
        "sft_task_manifest_single_persona": sample_tasks(splits["train_single_persona"], 3000, args.seed),
        "eval_128_single": sample_tasks(splits["test_single"], 128, args.seed),
        "eval_128_single_persona": sample_tasks(splits["test_single_persona"], 128, args.seed),
    }
    for name, tasks in sampled.items():
        manifest_hashes[name] = write_manifest(
            args.out / f"{name}.json",
            tasks,
            source_hash,
            {
                "kind": "sampled",
                "seed": args.seed,
                "source_manifest": "train/test official split",
                "sampling": "proportional largest-remainder allocation by (domain_zh, category), deterministic per-stratum Random(seed+offset)",
            },
        )

    train_ids = {t["task_id"] for t in splits["train_single"]}
    test_ids = {t["task_id"] for t in splits["test_single"]}
    persona_train_ids = {t["task_id"] for t in splits["train_single_persona"]}
    persona_test_ids = {t["task_id"] for t in splits["test_single_persona"]}
    report = {
        "source": str(args.source),
        "source_sha256": source_hash,
        "record_count": len(records),
        "expected_counts": EXPECTED,
        "actual_counts": {
            "single": {"train": len(train_ids), "test": len(test_ids)},
            "single_persona": {"train": len(persona_train_ids), "test": len(persona_test_ids)},
        },
        "split_overlap": {
            "single": len(train_ids & test_ids),
            "single_persona": len(persona_train_ids & persona_test_ids),
        },
        "manifest_hashes": manifest_hashes,
        "profiles": {
            key: profile(records, tasks, args.tokenizer)
            for key, tasks in {**splits, **sampled}.items()
        },
        "schema": {
            "record_keys": sorted(records[0].keys()),
            "instruction_keys": sorted((records[0].get("instructions") or [{}])[0].keys()),
        },
    }
    (args.out / "p2_profile.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

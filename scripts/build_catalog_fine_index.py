#!/usr/bin/env python3
"""按 pinned Princeton WebShop Lucene 流程为 Catalog-Fine 构建检索索引。

只实现官方 converter 的 JSONL 投影：Title、Description、首个
BulletPoint（Catalog-Fine 没有该字段时为空）和 options。不会加入 query、gold
字段或其他项目自定义内容，因此检索语义不会依赖未定义的派生字段。
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Catalog 顶层必须是 list")
    return records


def option_text(record: dict[str, Any]) -> str:
    values: list[str] = []
    options = record.get("customization_options") or {}
    for name, contents in options.items():
        if contents is None:
            continue
        normalized = []
        for item in contents:
            value = str(item.get("value", "")).strip().replace("/", " | ").lower()
            normalized.append(value)
        values.append(f"{str(name).lower()}: {', '.join(normalized)}")
    return ", and ".join(values)


def make_document(record: dict[str, Any]) -> dict[str, Any]:
    asin = str(record.get("asin", ""))
    if not asin or asin == "nan" or len(asin) > 20:
        raise ValueError(f"非法 Catalog-Fine asin: {asin!r}")
    contents = " ".join(
        [
            str(record.get("title", "")),
            str(record.get("full_description", "")),
            "",  # Catalog-Fine 没有 WebShop small_description/BulletPoints
            option_text(record),
        ]
    ).lower()
    return {"id": asin, "contents": contents, "product": record}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.source)
    documents = [make_document(record) for record in records]
    ids = [doc["id"] for doc in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Catalog-Fine asin 不唯一，拒绝静默去重")
    resource_dir = args.search_root / "resources"
    index_dir = args.search_root / "indexes"
    resource_dir.mkdir(parents=True, exist_ok=True)
    with (resource_dir / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(resource_dir),
            "--index",
            str(index_dir),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            "1",
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ],
        check=True,
    )
    print(json.dumps({"documents": len(documents), "resources": str(resource_dir), "index": str(index_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

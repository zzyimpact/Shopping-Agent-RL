#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-/root/ShopSimulator}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python 不存在：${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import importlib
import importlib.metadata as metadata
import pathlib
import sys

expected = [
    "torch", "transformers", "datasets", "accelerate", "peft", "trl",
    "safetensors", "sentencepiece", "pytest",
    "flask", "gym", "pyserini", "faiss-cpu", "selenium", "spacy", "thefuzz", "bs4", "rich",
]
module_names = {"bs4": "bs4", "faiss-cpu": "faiss"}
missing = []
for name in expected:
    module = module_names.get(name, name.replace("-", "_"))
    if importlib.util.find_spec(module) is None:
        missing.append(name)
    else:
        try:
            print(f"{name}={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            print(f"{name}=importable (metadata unavailable)")

import torch
print(f"python={sys.executable}")
print(f"python_version={sys.version.split()[0]}")
print(f"torch_cuda_compiled={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")

import spacy
if not spacy.util.is_package("zh_core_web_sm"):
    missing.append("zh_core_web_sm")

if missing:
    raise SystemExit(f"缺少关键 package: {', '.join(missing)}")

for module in ("transformers", "datasets", "accelerate", "peft", "trl"):
    importlib.import_module(module)
print("training imports: OK")
PY

if [[ -d "${UPSTREAM_ROOT}" ]]; then
  command -v javac >/dev/null 2>&1 || { echo "缺少 javac/OpenJDK" >&2; exit 2; }
  export PYTHONPATH="${UPSTREAM_ROOT}/shop_env:${UPSTREAM_ROOT}/single_eval:${PYTHONPATH:-}"
  "${PYTHON_BIN}" - <<'PY'
import importlib

modules = [
    "web_agent_site.engine.normalize",
    "web_agent_site.engine.engine",
    "web_agent_site.engine.goal",
    "web_agent_site.envs.web_agent_text_env",
    "env",
]
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        raise SystemExit(f"upstream import failed: {name}: {type(exc).__name__}: {exc}")
    print(f"upstream import: {name}=OK")
else:
    print("upstream imports: OK")
PY
else
  echo "未找到 upstream checkout：${UPSTREAM_ROOT}" >&2
  exit 2
fi

echo "CPU verification completed; CUDA unavailable is allowed."

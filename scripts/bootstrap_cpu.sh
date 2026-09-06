#!/usr/bin/env bash
set -euo pipefail

# P1 CPU bootstrap：复用当前 conda base，不创建新环境，不安装 GPU/RL 专用后端。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
PIP_BIN="${PIP_BIN:-/root/miniconda3/bin/pip}"
TORCH_CONSTRAINT="${TORCH_CONSTRAINT:-/tmp/p1-torch-constraints.txt}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${PIP_BIN}" ]]; then
  echo "找不到预期的 conda base Python/pip：${PYTHON_BIN} / ${PIP_BIN}" >&2
  exit 1
fi

printf 'torch==2.8.0+cu128\n' > "${TORCH_CONSTRAINT}"

# 版本固定到本轮已验证的 stable 组合；pip 会复用已安装依赖。
"${PIP_BIN}" install --no-input \
  --constraint "${TORCH_CONSTRAINT}" \
  "transformers==5.16.1" \
  "tokenizers==0.23.2" \
  "datasets==5.0.1" \
  "accelerate==1.14.0" \
  "peft==0.20.0" \
  "trl==1.12.0" \
  "safetensors==0.8.0" \
  "sentencepiece==0.2.2" \
  "pytest==9.1.1" \
  "flask==3.1.3" \
  "gym==0.26.2" \
  "spacy==3.8.16" \
  "pyserini==0.17.1" \
  "faiss-cpu==1.15.0" \
  "lightgbm==4.7.0" \
  "nmslib==2.1.2" \
  "selenium==4.48.0" \
  "thefuzz==0.22.1" \
  "cleantext==1.1.4" \
  "rank_bm25==0.2.2" \
  "openai==3.8.0"

if ! "${PYTHON_BIN}" -c 'import spacy; raise SystemExit(0 if spacy.util.is_package("zh_core_web_sm") else 1)'; then
  # 官方模型托管站点在远程机器上可能不可达；不要因此回滚已完成的 CPU 依赖准备。
  if command -v timeout >/dev/null 2>&1; then
    timeout "${SPACY_DOWNLOAD_TIMEOUT:-60}" "${PYTHON_BIN}" -m spacy download zh_core_web_sm || \
      echo "警告：zh_core_web_sm 下载失败；upstream goal/text env 验证将保持阻塞。" >&2
  else
    "${PYTHON_BIN}" -m spacy download zh_core_web_sm || \
      echo "警告：zh_core_web_sm 下载失败；upstream goal/text env 验证将保持阻塞。" >&2
  fi
fi

if ! command -v javac >/dev/null 2>&1; then
  echo "缺少 javac；请按 upstream setup 安装 OpenJDK 21 后重试。" >&2
  exit 2
fi

mkdir -p \
  "${PROJECT_ROOT}/configs/runtime" \
  /root/data/shopsim \
  /root/data/models \
  /root/runs/shopsim-rl \
  /root/.cache/shopsim-rl

echo "CPU bootstrap 完成；未创建新环境、未修改 torch、未下载大模型。"

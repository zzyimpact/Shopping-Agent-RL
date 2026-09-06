# P1 环境审计

## OS / Python

盘点目标为 SSH alias `rtx-pro-6000-3`；远端 `hostname` 输出为 `autodl-container-vwjczv9tz3-ef252ce2`。

```text
uname -a: Linux autodl-container-vwjczv9tz3-ef252ce2 5.15.0-78-generic #85-Ubuntu SMP Fri Jul 7 15:25:09 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
python --version: command not found on PATH
which python: command not found on PATH
pip --version: command not found on PATH
usable Python: /root/miniconda3/bin/python (Python 3.12.3)
usable pip: /root/miniconda3/bin/pip (pip 24.0)
conda: /root/miniconda3/bin/conda 24.4.0
```

当前只有 conda `base` 环境（`/root/miniconda3`），没有发现已激活的 venv；未创建新环境。

## 已安装的相关包

通过 `/root/miniconda3/bin/python` 检查 package metadata：

| 包 | 版本 / 状态 |
|---|---|
| torch | `2.8.0+cu128` |
| numpy | `2.3.2` |
| requests | `2.31.0` |
| PyYAML | `6.0.2` |
| 其余训练/环境包 | 见下方缺失列表 |

Torch 额外检查：compiled CUDA `12.8`；`torch.cuda.is_available()` 为 `False`，符合当前 CPU-only 条件。`nvidia-smi` 文件存在但在当前容器中因权限被拒绝，未进行 GPU 操作。

## 环境与直接复用判断

- 远端 base Python 可复用，且已有 CUDA-enabled PyTorch wheel；当前不需要重装或升级。
- `requests`、NumPy、PyYAML 可支持部分轻量脚本。
- 远端未找到 upstream ShopSimulator checkout、项目目录或论文 PDF；本地 checkout 可作为当前源码审计对象，但正式远程运行前需按 WORKFLOW.md 补齐 checkout 和数据/index 路径。
- 本地 upstream 的 `normalize.py` 可独立导入；`goal.py`、`engine.py`、`web_agent_text_env.py` 的导入分别受 `spacy`、`tqdm/pyserini`、`gym` 等依赖阻塞。

## 已知缺失依赖

未安装：`transformers`、`tokenizers`、`accelerate`、`peft`、`trl`、`deepspeed`、`ray`、`vllm`、`flash-attn`、`datasets`、`sentencepiece`、`safetensors`、`faiss/faiss-cpu`、`pandas`、`scipy`、`scikit-learn`、`pytest`、`ROLL`、`verl`、`flask`、`gym/gymnasium`、`pyserini`、`spacy`、`thefuzz`、`rich`、`openai`。

## 后续可能需要安装

后续 P1/P2 可能需要按上游 `shop_env/requirements.txt` 补齐 Flask/Gym/pyserini/spaCy/thefuzz/rich 等环境依赖，并按训练 backend 选择安装 Transformers、PEFT、TRL 或 ROLL/veRL 相关组件。具体版本应在依赖冲突和 GPU_READY 前验证后再决定，不在本任务猜测或安装。

本任务没有执行任何大包安装、重装、升级或降级；没有下载模型权重，也没有调用 teacher API。

## 审计限制

远程容器的系统 `python`/`pip` 不在 PATH，使用绝对路径完成了盘点；这不是项目级环境决策。由于远端缺少上游 checkout，本阶段没有启动 ShopSimulator 服务，也没有进行正式训练或评测。

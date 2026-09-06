# P1 Remote CPU 环境审计

**文档状态**：P1 bootstrap 记录；需要用户审阅：否（远程 checkout 缺失属于后续 open decision）。

## 远程定位

- SSH alias：`rtx-pro-6000-3`
- `hostname`：`autodl-container-vwjczv9tz3-ef252ce2`
- `pwd`：`/root`
- 预期项目路径：`/root/shopping-agent-rl`，当前不存在。
- 预期 upstream 路径：`/root/ShopSimulator`，当前不存在。
- 已定向查找：`/root`、`/tmp`、`/workspace`、`/workspaces`、`/project`、`/projects`、`/data`、`/mnt`、`/opt`、`/srv`；未发现项目、upstream 或论文 PDF。

## Python / conda

```text
Python: /root/miniconda3/bin/python 3.12.3
pip: /root/miniconda3/bin/pip 24.0
conda: /root/miniconda3/bin/conda 24.4.0
环境：仅 base，未创建新的 conda/venv
```

系统 `python` / `pip` 不在 PATH；P1 脚本使用上述绝对路径。

## torch / CUDA

```text
torch: 2.8.0+cu128
compiled CUDA: 12.8
torch.cuda.is_available(): False
```

当前为 CPU-only，CUDA unavailable 不视为失败；本轮没有修改 torch。

## 本轮安装与复用

已有并继续复用：`torch==2.8.0+cu128`、`numpy==2.3.2`、`PyYAML==6.0.2`。通过显式 `torch==2.8.0+cu128` constraint 安装了：

| 包 | 版本 |
|---|---:|
| transformers | 5.16.1 |
| tokenizers | 0.23.2 |
| datasets | 5.0.1 |
| accelerate | 1.14.0 |
| peft | 0.20.0 |
| trl | 1.12.0 |
| safetensors | 0.8.0 |
| sentencepiece | 0.2.2 |
| pytest | 9.1.1 |
| requests | 2.34.2 |
| pandas | 3.0.5 |
| pyarrow | 25.0.1 |

上游 CPU 依赖安装尝试中，`pyserini==2.4.0` 明确要求 `torch>=2.9`，与冻结的 torch 冲突，已停止该版本安装；没有通过升级 torch 绕过冲突。改用兼容当前 torch 的 `pyserini==1.4.0`，并已完成安装。

## TRL compatibility

已确认 `import trl`、`transformers`、`datasets`、`accelerate`、`peft` 的版本组合可安装于 Python 3.12，并保留 torch 2.8.0+cu128。TRL exact version：`1.12.0`。本轮没有编写 GRPO integration。

## 未安装项目

没有安装 ROLL、veRL、vLLM、SGLang、DeepSpeed、FlashAttention、TensorRT-LLM 或其他 GPU-specific engine；没有下载 Qwen3-8B/其他大模型，也没有调用 teacher API。

## 远程项目 / upstream 阻塞

由于远程缺少 `shopping-agent-rl/` 和 ShopSimulator checkout，无法在远程完成项目 Git 同步、upstream commit pin、关键 upstream module import 或服务启动验证。当前本地 checkout 不能替代 remote validation；`zh_core_web_sm` 因 GitHub 下载超时未安装，因此即使源码恢复，`goal.py` 的导入仍需补齐该模型。这不是 P2 工作，也未生成任何 task manifest、dataset profile 或 reward artifact。

## 运行时路径

路径约定已写入 `configs/runtime/remote.yaml`：项目代码 `/root/shopping-agent-rl`，upstream `/root/ShopSimulator`，数据 `/root/data/shopsim`，模型 `/root/data/models`，runs `/root/runs/shopsim-rl`，cache `/root/.cache/shopsim-rl`。这些目录尚未因远程项目缺失而自动创建。

明确结论：没有创建新虚拟环境；没有修改 torch；没有下载模型；没有调用付费 API；没有执行 P2。

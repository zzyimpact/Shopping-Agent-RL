# P1 Remote CPU 环境审计

**文档状态**：P1 bootstrap 记录；需要用户审阅：否。

## 远程定位

- SSH alias：`rtx-pro-6000-3`
- `hostname`：`autodl-container-vwjczv9tz3-ef252ce2`
- `pwd`：`/root`
- 项目路径：`/root/shopping-agent-rl`，分支 `p1/remote-cpu-bootstrap`，提交 `96e0b64`，远程 Git status clean。
- upstream 路径：`/root/ShopSimulator`，已恢复源码和 Catalog-Fine 数据快照；目录没有 upstream `.git` 元数据。
- 初次定向查找后确认目录存在；因远程 GitHub clone/认证不可用，使用已审计快照恢复源码与数据，没有覆盖既有内容。

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
| pyserini | 0.17.1 |
| faiss-cpu | 1.15.0 |
| lightgbm | 4.7.0 |
| nmslib | 2.1.2 |
| selenium | 4.48.0 |
| Flask | 3.1.3 |
| Gym | 0.26.2 |
| spaCy | 3.8.16 |
| thefuzz | 0.22.1 |
| beautifulsoup4 | 4.13.4 |
| rich | 15.0.0 |
| scikit-learn | 1.9.0 |
| OpenJDK | 21.0.12 |

上游 CPU 依赖安装尝试中，`pyserini==2.4.0` 明确要求 `torch>=2.9`，与冻结的 torch 冲突，已停止该版本安装；没有通过升级 torch 绕过冲突。`pyserini==1.4.0` 虽可安装但导入时触发 OpenAI credential side effect，因此改用上游 requirements 时代的 `pyserini==0.17.1`，并补齐 `faiss-cpu`、Selenium 与 OpenJDK 21。

## TRL compatibility

已确认 `import trl`、`transformers`、`datasets`、`accelerate`、`peft` 的版本组合可安装于 Python 3.12，并保留 torch 2.8.0+cu128。TRL exact version：`1.12.0`。本轮没有编写 GRPO integration。

## 未安装项目

没有安装 ROLL、veRL、vLLM、SGLang、DeepSpeed、FlashAttention、TensorRT-LLM 或其他 GPU-specific engine；没有下载 Qwen3-8B/其他大模型，也没有调用 teacher API。

## 远程项目 / upstream 阻塞

remote-only 验证已确认 `normalize.py`、`engine.py` 和单轮 `env.py` 可导入；`pip check` 通过。`web_agent_text_env.py` / `goal.py` 仍因 `zh_core_web_sm` 缺失无法导入：官方 GitHub 下载超时，PyPI 镜像对应的 `999.9.9` 是防依赖混淆 stub，未安装。`verify_cpu_env.sh` 将该项明确报告为 optional upstream import warning，不伪装为完整环境可用。该模型是 upstream setup 的 CPU 依赖，未提供 API key，也未调用 OpenAI。没有生成任何 task manifest、dataset profile 或 reward artifact。

## 运行时路径

路径约定已写入 `configs/runtime/remote.yaml`：项目代码 `/root/shopping-agent-rl`，upstream `/root/ShopSimulator`，数据 `/root/data/shopsim`，模型 `/root/data/models`，runs `/root/runs/shopsim-rl`，cache `/root/.cache/shopsim-rl`；数据、模型、runs、cache 目录已创建为空目录。

明确结论：没有创建新虚拟环境；没有修改 torch；没有下载 Qwen3-8B 或其他大模型；没有调用付费 API；没有执行 P2。`pyserini==2.4.0` 与冻结的 torch 2.8.0 存在明确冲突，因此保留兼容的 `pyserini==0.17.1`，`pip check` 无 broken requirements。

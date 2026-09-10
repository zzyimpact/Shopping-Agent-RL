# Engineering Round D1 — Real training environment CPU preflight

2026-09-10，起点 `62477006ddaaf6ee9da951e953fa0ff76f500779`，branch `main`，初始工作区干净。

**TRAINING_ENV_CPU_PREFLIGHT: PASS**。实际 TRL/HF CPU construction、forward/backward、
optimizer、checkpoint/resume 均已执行。没有 GPU/MPS、Qwen3-8B weight load、远程 ShopEnv、
teacher API 或远程进程操作；teacher collector/config/artifact/SQLite 均未修改。

## Isolated environment

本地隔离路径：`Shopping-Agent-RL/.cache/training-preflight-venv/`（现有 `.cache/` ignore）。
使用现有 `/Users/zzy/.local/bin/python3.10` 创建 venv，未改系统 Python/teacher environment。
系统默认 Python 3.14.7 未用于此训练栈。

| Package | 实际安装版本 |
| --- | --- |
| Python | 3.10.21 |
| torch | 2.8.0（macOS arm64 wheel，测试设备 CPU） |
| transformers | 4.57.6 |
| trl | **1.12.0** |
| peft | 0.19.1 |
| accelerate | 1.15.0 |
| datasets | 4.8.5 |
| tokenizers | 0.22.2 |
| pytest | 9.1.1 |

全部在原 proposal 范围内解析成功；`pip check`: **No broken requirements found**。
真实 import 六个训练包及项目 `training.policy/rollout/sft_data/sft/grpo` 成功。
项目使用 `PYTHONPATH=src`，模块名为 `training.*`。

`requirements-training.txt` 保存工作的 top-level exact pins，额外固定 tokenizer 与必要
project client/config dependencies。GPU 环境使用相同 Python/API versions，torch 选择
目标 Linux/GPU/driver 支持的 **2.8.0 CUDA build**；Mac wheel 不可复制去 Linux。
这不是跨平台 GPU runtime 已验证的声明，也没有新增环境管理框架。

已安装 TRL `grpo_trainer.py` SHA256 与 Round C 审计文件完全一致：
`cbdda3ff10accab8fe36b6e0059dff916037b49a0d8ffca829e25781c9dfb2a3`。
Round C 审计的 sdist SHA256：
`494ec6cfd07097bb8116cc7bec299b067dd765d1c19ed3eca2d874a29979f229`。

## Real Qwen tokenizer / SFT labels

仅 `AutoTokenizer.from_pretrained(..., local_files_only=True, use_fast=True)` 加载本地
`/Users/zzy/Desktop/Coding/shop-rl/models/Qwen3-8B/` tokenizer；实际类为 `Qwen2TokenizerFast`，
这是该 Qwen3 artifact 的 tokenizer class，并非加载了其他模型。
只读取 Round B 的四个 `.cache/round_b_review/*.json` 副本，读取前后 hash 一致。
没有扫描 accepted directory 或读取 SQLite。

| Review example | Total tokens | Assistant TRAIN | MASK | Assistant EOS | Context 32768 |
| --- | ---: | ---: | ---: | ---: | --- |
| Single 944750557106 | 4357 | 574 | 3783 | 5 | PASS |
| Single 941565736582 | 3737 | 456 | 3281 | 5 | PASS |
| Persona 922242361311 | 3415 | 293 | 3122 | 4 | PASS |
| Persona 740656872880 | 3860 | 313 | 3547 | 4 | PASS |

四条都实测通过 native rendering、fast offsets、逐 token labels。system/user/intermediate
observation/header 为 `-100`，assistant content + 每个 EOS 为原 token ID。
最终 token 全为 `<|im_end|>`（151645），末尾 terminal observation 不进入 Trainer input。
将 max_length 设为实际长度减一均显式失败，没有静默截断。source/review 不变。

## SFTTrainer runtime

随机 tiny `Qwen3ForCausalLM(Qwen3Config(...))`：hidden=16、intermediate=32、1 layer、
2 heads、1 KV head、head_dim=8、vocab=len(real tokenizer)、tied embeddings；少于 3M params，
CPU FP32。LoRA rank=2/alpha=4、q_proj/v_proj 只是 fixture，不能推广为正式 8B 配置。
使用短 fixture，**未用四条长真实轨迹做训练**。

- 真实 Dataset / SFTTrainer / collator construction：PASS；collator 原样保留 labels。
- 独立 labels-free logits 的 `cross_entropy(ignore_index=-100)` 与真实 masked loss 相等。
- TRL 1.12 原生 chunked CE 在带 labels 时返回 `logits=None`，不是 runtime bug；测试通过
  不带 labels 的 forward 获取独立参考 logits，没有禁用或修改 TRL chunked CE。
- 初始计划 2 steps，callback 在 step 1 停止并保存；重新构造同配置 Trainer，HF resume 到
  step 2：PASS。`optimizer.pt` 有 state，`scheduler.pt.last_epoch=1`；恢复后的 adapter hash、
  global_step=1、optimizer state、scheduler epoch 均检查通过。
- step 1/2 loss 为 `11.9052 / 11.9048`，grad norm `0.091324 / 0.092492`，finite 且参数更新。
- final adapter export 缺少 Trainer state，被项目 resume 边界明确拒绝。

## GRPOTrainer runtime / observed ordering

使用真实 `ShopGRPOTrainer(GRPOTrainer)`、`rollout_func`、Dataset、RepeatSampler、reward
extras 和已有 AgentRollout。Tiny config：G=2、2 groups/update、microbatch=1、accumulation=4、
steps_per_generation=4、num_iterations=1、max_steps=2；全部仅用于测试。

任务 schedule：`A(index 0), B(1), A(2), B(3)`。**实际 dataloader**：

```text
[0,0,1,1] 重复 4 个 dataloader batches
[2,2,3,3] 重复 4 个 dataloader batches
```

实际 rollout callback 仅执行两批，buffer 没有重复跑环境：

| 时间 | global_step | metadata schedule indices | task IDs | rewards |
| --- | ---: | --- | --- | --- |
| checkpoint 前 | 0 | 0,0,1,1 | A,A,B,B | 1,0,1,0 |
| checkpoint resume 后 | 1 | 2,2,3,3 | A,A,B,B | 1,0,1,0 |

两批 advantages 实测都是 `[+0.7070068,-0.7070068,+0.7070068,-0.7070068]`。
每成员独立 env instance/reset；共 8 resets、16 steps（每 episode 两步），每批 reward callback
一次；callback 不重跑 environment/scorer。基础设施 fake disconnect 经过真实 parent path
向上传播，metadata bridge 清空、session 释放，不伪装成 reward=0。

metadata bridge 的 transport prompts 实测全为空 user message；真实 tokenized initial prompt
来自 reset + 原有 prompt builder。token stream 中无 fixture task ID、schedule_index 或隐藏
evaluator marker。没有 prompt-text lookup。

## Multi-turn mask、logprob、真实 update

随机 tiny model 无法自然完成购物协议，因此 **multi-turn fixture 限定了 generate 的候选
token 前缀**，让它实际采样两种可解析 Thought/Action 之一并生成 EOS。仍调用当前
QwenPolicy.sample → 真实 Transformers generate / transition scores；没有伪造 PolicySample。
该约束仅在测试中，不进入正式 policy/config。另一个测试单独验证完全无约束 stochastic generation。

每条真实 completion 为 `sampled A1/EOS → inserted O1/header/prefix → sampled A2/EOS`。
验证了真实第二轮 model.generate 输入包含 O1；最终 terminal observation 不在 completion。
`completion_ids/logprobs/env_mask` 长度相等，采样 IDs 与 backend sequences 完全一致，logprobs finite。

真实 TRL scoring/prepare/loss 路径中：

| Token ownership | completion attention | tool_mask / policy loss mask | 对 per-token policy logprob 的 autograd 梯度 |
| --- | ---: | ---: | --- |
| Model sampled（含 EOS） | 1 | 1 | 非零 |
| External observation / inserted header | 1 | 0 | **严格为零** |

中间 assistant EOS 之后 attention 与 policy tokens 仍保留；不是遇 EOS 截断。
共 8 microbatch 的实际 loss backward 均检查上述梯度。外部 observation 仍作为 conditioning，
这里排除的是其自身的 policy-token loss，不是切断后续 token 对上下文的依赖。

**两个 logprob 的用途不同：**

- Sampling logprobs：QwenPolicy 真实采样分布的逐 token 概率，返回给 custom rollout contract，
  实测原样进入 `sampling_per_token_logps`。无约束生成得到 3 个 backend IDs，对齐 normalized
  `compute_transition_scores`；没有 decode→retokenize 生成 token。
- Old-policy likelihood：当前 non-vLLM、单次 iteration、generation/update 对齐路径，真实 prepared
  inputs **不含 `old_per_token_logps`**；loss 使用当前 likelihood 的 detached 值。测试核对
  ratio=1 时真实 forward loss 等于 `-advantage / accumulation`，并检查 backward 非零。
  没有把 sampling logprob 填成 old likelihood。fixture 的 constrained sampling 也不冒充
  无约束的 policy likelihood；正式 top_p 等 hyperparameters 没有被此次测试冻结。

真实 GRPO aggregate loss 两步都为有限的 `0.0`（对称 group advantages 的预期值），
grad norm `0.0036511 / 0.0036555`；可训练 adapter hash：

```text
before:        08fe8d1f0ac62605a3a749b69419cf0b429cd3a71728a4cf0aeaf0d832a5d062
after update1: 82ddf636286e9c0dad682ea8ce6a639d9555f70e35823ca40ae1dc139345f899
after update2: 3deef6648264334fbf620165cd22c0195705519df5e4cffde21f0b7b7818ae4a
```

resume 后下一次 callback unwrap 的对象就是 Trainer 当前 model，hash 等于 update1 的结果，
不同于初始化。HF 实际恢复 adapter/global_step/optimizer/scheduler，fresh buffer 从 indices
2,3 重新 rollout；没有重跑 index 0，没有跳 group，没有 replay 旧 episode。
项目 run identity 检查同一 schedule hash。未提交 update 仍遵循从最近 checkpoint 重跑的语义。

CLI strict/loose/alpha runtime 检查分别得到 alpha=1/0/.5；fake partial terminal 得到
scalar=.5/.8/.65，互斥参数被拒绝。现有八指标与 reward 定义未改。

## Minimal changes / reproducibility

没有发现 Round B/C token、group、reward 或 resume 的核心 runtime bug。
为实际 CPU preflight 增加了必要设备边界：两个 trainer factory 可显式 `use_cpu=True`，
使用 FP32、关闭 pinned-memory；默认仍是原 BF16 GPU 路径，正式 CLI 不新增 CPU training 模式。
GRPO metrics callback 只在 CUDA device 读取 GPU memory，CPU 日志不产生虚假的 GPU 指标。

`tests/training/runtime/test_cpu_preflight.py` 为 opt-in regression tests；默认轻量 suite 跳过它。
生成 checkpoint/report 全在 `.cache/d1-preflight/`；不提交 venv/cache/model/checkpoint。

复现（首次仅在独立 venv 安装，已有本轮 venv 可直接运行测试）：

```bash
cd /Users/zzy/Desktop/Coding/shop-rl/Shopping-Agent-RL
/Users/zzy/.local/bin/python3.10 -m venv .cache/training-preflight-venv
.cache/training-preflight-venv/bin/python -m pip install -r requirements-training.txt pytest==9.1.1
.cache/training-preflight-venv/bin/python -m pip check

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 RUN_TRAINING_PREFLIGHT=1 \
.cache/training-preflight-venv/bin/python -m pytest tests/training -q \
  --basetemp=.cache/d1-preflight/checkpoints
```

最终结果：**51 passed, 11 warnings in 4.64s**。warnings 是 TRL experimental API 提示、随机
内存模型没有 source config 路径时 PEFT 的 vocab 保存提示；checkpoint 实际恢复通过。
无训练依赖的默认轻量命令也执行过：46 passed, 5 skipped。

## Model staging / GPU-only remaining

本地 config/tokenizer/index/SHA256SUMS 均存在，index 引用的五个 safetensor shard 全部存在，
总字节 16,381,516,776。只 stat 文件/读小 metadata，没有重 hash 或加载权重。

当前 `configs/runtime/remote.yaml` 指向 `rtx-pro-6000-3`，models_root 为 `/root/data/models`，
约定下一轮完整模型目标为 `/root/data/models/Qwen3-8B`；原 tokenizer-only path 是
`/root/data/models/Qwen3-8B-tokenizer`。本轮没有 SSH 查询远程状态，完整权重远程存在性
**未确认，按 NEEDS STAGING/确认处理**。以下仅给用户后续执行，本轮未传输：

```bash
ssh rtx-pro-6000-3 'mkdir -p /root/data/models/Qwen3-8B'
rsync -a --partial --progress --exclude '.cache/' \
  /Users/zzy/Desktop/Coding/shop-rl/models/Qwen3-8B/ \
  rtx-pro-6000-3:/root/data/models/Qwen3-8B/
```

下一轮 GPU-only checks 已收敛到：CUDA/目标卡与 torch build、8B BF16 + LoRA load、真实
generation/logprob 数值、8B SFT/GRPO tiny update、checkpoint reload、显存/吞吐。
真实 ShopEnv episode 需要单独安排允许的远程负载窗口。LoRA/batch/generation 正式参数
尚未冻结；sequential rollout 为 **PERFORMANCE-PROFILING REQUIRED**，本轮无吞吐优化。

**READY TO ENABLE GPU: YES**（CPU contract gate 通过；开启后仍需完成上述 GPU runtime checks）。

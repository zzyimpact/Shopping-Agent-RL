# Engineering Round A: Training Readiness Audit

日期：2026-09-10
范围：只读审计与轻量架构冻结，不启动训练，不改变 teacher collection。

## 1. Executive Verdict

当前项目已经具备可靠的 ShopSimulator protocol、prompt、action parser、远程
multi-session environment client、官方 reward 包装和冻结的 task manifests；Qwen3-8B
权重与 tokenizer 文件也已在本地 artifact 目录存在。当前还不能称为
`TRAINING_CODE_READY`：缺口集中在 local policy runtime、统一 agent rollout、独立
evaluation、SFT formatter/trainer，以及 TRL GRPO 的薄 integration。

没有发现会推翻既定方案的 blocker。最大的实际边界是：当前机器没有训练依赖，且
GRPO 的 `rollout_func` 是 TRL 的 experimental API；应在独立 training environment
中 pin 后做一次 GPU preflight，再开始正式训练。teacher collection 的两个运行进程
本轮没有停止、重启、修改或测试。

本轮状态拆分为：

```text
TRAINING_CODE_READY  = NO
MODEL_ARTIFACT_READY = PARTIAL (本地文件齐全，远程部署未在本轮验证)
ENV_READY            = YES (已有 CPU/remote protocol；local training client 仍需薄接线)
SFT_DATA_READY       = IN PROGRESS (teacher collection 尚未达到最终冻结点)
GPU_RUNTIME_READY    = NO (本轮明确不开 GPU)
```

## 2. Current HEAD / Repo State

| 项目 | 当前值 |
|---|---|
| branch | `main` |
| HEAD | `51daa72 feat: add optional teacher auto resume with fixed delay` |
| working tree | clean（本轮文档写入前） |
| teacher collection | 用户报告 `single` 与 `single_persona` 均持续运行；本轮不触碰 |
| GPU | 未启用；不加载模型、不做 forward/backward |
| local training packages | `torch`、`transformers`、`trl`、`peft`、`accelerate` 均未安装 |
| historical remote package record | `docs/P1_ENV_AUDIT.md` 记录过 TRL `1.12.0`，本轮未登录修改或重验远程 environment |
| upstream | `ShopSimulator-main` 本地 checkout 可读；训练模块未见于 upstream tree |

模型 artifact 在仓库外的 `/Users/zzy/Desktop/Coding/shop-rl/models/Qwen3-8B/`：
五个 safetensors shard、`model.safetensors.index.json`、`config.json`、
`tokenizer.json`、`tokenizer_config.json`、`generation_config.json` 和
`SHA256SUMS.txt` 均存在。项目没有单独的 model manifest；远程约定路径见
`configs/runtime/remote.yaml`，本轮没有验证远程文件存在性。

## 3. Training Readiness Matrix

| component | status | existing implementation | reuse | missing work | needs GPU |
|---|---|---|---|---|---|
| Qwen weights | PARTIAL | 本地 Qwen3-8B shards/config/checksum | 直接使用 artifact path | 远程 staging/manifest gate | NO（存在性可无卡确认） |
| tokenizer | READY | `tokenizer.json` + Qwen chat template | 直接使用 | 训练环境加载检查 | NO |
| local model loader | MISSING | 无 project loader | `AutoModelForCausalLM`/`AutoTokenizer` | 一个小 policy runtime | YES（首次真实 load） |
| LoRA/PEFT | MISSING | 无 PEFT 代码；依赖未装 | `peft.LoraConfig` + TRL/Trainer | target modules/rank 由 GPU preflight 决定 | YES |
| ShopEnv client | READY | `src/env/teacher_env_client.py` + remote service | REUSE AS-IS | local policy adapter 调用同一 reset/step | NO（CPU protocol 已有） |
| policy prompt | READY | `src/rollout/prompt.py` + upstream snapshot | REUSE AS-IS | local tokenizer boundary | NO |
| action parser | READY | `src/rollout/protocol.py`，对齐 upstream | REUSE AS-IS | 无 | NO |
| online trajectory loop | PARTIAL | `formal_collector.execute_rollout` 是 teacher/API 专用 | THIN WRAPPER，不复用 ledger/teacher client | policy `generate` + env loop + terminal record | YES（policy generation） |
| reward wrapper | READY | `src/rewards/shopsim_reward.py`；8 metrics + `r_alpha` | REUSE AS-IS | 接入 online terminal payload | NO（需 upstream scorer） |
| 8-metric evaluator | PARTIAL | metrics 逻辑在 profiler/collector 内 | THIN WRAPPER | standalone evaluator/aggregator | YES（模型 rollout） |
| 128 eval subset | READY | P2 manifests/hash 已记录 | REUSE AS-IS | evaluator 读取路径 | YES（模型 rollout） |
| full eval | PARTIAL | upstream `single_eval` scripts/env 可读 | THIN WRAPPER | 统一 project output/metrics | YES |
| SFT formatter | MISSING | accepted JSON 有 policy-visible `messages` 与 evaluator-only 分区 | 新增小 formatter | frozen accepted manifest → chat template/labels | NO（tokenizer 可 CPU） |
| SFT loss masking | MISSING | 无 labels/mask 代码 | 新增小 preprocessing function | assistant-only labels；先验证 Qwen template mask | NO（tokenizer 可 CPU） |
| SFT trainer | MISSING | 无 trainer/config | TRL `SFTTrainer` + PEFT | data collator/config/checkpoint | YES |
| GRPO backend | PARTIAL | 设计冻结为 HF TRL；本机未安装 | `GRPOTrainer` | 独立 env pin/验证 API | YES（初始化/训练） |
| GRPO rollout integration | MISSING | 无 custom rollout | TRL `rollout_func` + thin ShopEnv adapter | token IDs/logprobs/env mask/group mapping | YES |
| GRPO reward callback | MISSING | reward scorer 不接 TRL | `reward_funcs` callback 调 scorer | 返回每条 completion 的 `r_alpha`，保留 diagnostics | NO（逻辑可 CPU） |
| checkpoint | PARTIAL | teacher SQLite/JSON 不是 model checkpoint | HF Trainer checkpoint | adapter + optimizer/scheduler/RNG/task cursor | YES |
| resume | PARTIAL | 仅 teacher resume | HF Trainer resume + task sampling state | 训练 run state/compatibility | YES（真实 reload） |
| run manifest | PARTIAL | teacher manifest/storage 已有 | 复用 provenance 字段思路 | training manifest/config/hash | NO |
| metrics logging | PARTIAL | `collector.log`/summary 仅 teacher | JSONL writer 小复用 | reward/group/rollout/update/eval metrics | NO |
| GPU preflight | GPU-PREFLIGHT | workflow 已定义 gate，未执行 | 现有 P5 计划 | CUDA/BF16/LoRA/one update checks | YES |
| experiment configs | PARTIAL | `configs/spec/project.yaml` + 空 `experiments/` | 保留 shared spec | 只新增小型 scenario/mode overrides | NO |

## 4. Reuse Map

| 功能 | 决策 | 边界 |
|---|---|---|
| upstream Catalog/search/ShopSimulator semantics | REUSE AS-IS | 不复制或重写搜索和商品环境 |
| remote `TeacherEnvClient` protocol | REUSE AS-IS | policy rollout 仍使用同一 reset/step contract；训练环境容量由 GPU preflight/运行配置处理 |
| system prompt/persona projection | REUSE AS-IS | `src/rollout/prompt.py` 是唯一 policy-visible message builder |
| Thought/Action extraction/parsing | REUSE AS-IS | `src/rollout/protocol.py`；不改成 tool calling |
| terminal reward | REUSE AS-IS | `src/rewards/shopsim_reward.py` 委托 upstream scorer |
| teacher `formal_collector.py` / ledger / accepted schema | DO NOT USE | 它是 paid teacher collection；不把 local policy rollout 写成 teacher artifact |
| local policy `generate(messages)` | NEW SMALL MODULE | 一个薄 Qwen/PEFT runtime，不做 backend registry |
| Base/SFT/GRPO agent loop | NEW SMALL MODULE | 一个统一 `reset → generate → parse → step → append → terminal` loop |
| evaluator | THIN WRAPPER | 复用同一 agent loop + 8 metrics；固定 128 与 full manifests |
| SFT formatter | NEW SMALL MODULE | 只接受 selected/frozen accepted manifest；过滤 evaluator-only/reward |
| SFT trainer | THIN WRAPPER | `SFTTrainer` + PEFT；不建通用 trainer framework |
| GRPO trainer | THIN WRAPPER | `GRPOTrainer` + `rollout_func`/`reward_funcs`；不实现 GRPO 算法 |
| run/checkpoint/metrics | NEW SMALL MODULE | JSON manifest + JSONL + HF checkpoints，避免 DB/dashboard |

## 5. Target Lightweight Architecture

```text
experiment YAML
  ↓
run manifest / seed / provenance
  ↓
QwenPolicy (Base | SFT adapter | GRPO adapter)
  ↓ generate(messages)
AgentRollout
  ↓ reset / parse Action / step
TeacherEnvClient → ShopSimulator service
  ↓ terminal payload
ShopSimulatorReward → 8 metrics + r_alpha
  ├─ Evaluator → eval_128 / full metrics
  ├─ SFTFormatter → assistant-only labels → TRL SFTTrainer
  └─ GRPORolloutFunc → prompt_ids/completion_ids/logprobs/env_mask
                         ↓ reward_funcs → TRL GRPOTrainer → checkpoint
```

`AgentRollout` 是 Base evaluation、SFT smoke 后 evaluation 和 GRPO online rollout
共同的语义边界。它不负责保存 teacher accepted JSON，也不拥有训练 optimizer。

## 6. TRL Integration Decision

后端固定为 **Hugging Face TRL + Transformers + PEFT**。ROLL 与 verl 不进入本项目
v1；vLLM 只作为未来 profiling 后的可选性能优化，第一版 correctness path 使用
Transformers generation。

本轮从 PyPI source archive 只读检查了 TRL `1.12.0` 和 `1.13.0`：两者都有：

```text
GRPOTrainer(..., reward_funcs=..., args=..., train_dataset=...,
            processing_class=..., peft_config=..., rollout_func=...)
rollout_func(prompts, trainer) -> {
    prompt_ids, completion_ids, logprobs,
    optional env_mask, plus per-completion extra fields
}
```

`env_mask` 会被 TRL 当作 `tool_mask`，参与 `completion_mask * env_mask` 的 loss
mask；因此 ShopEnv observation 拼接到 completion 序列时可以标为 0，policy-generated
tokens 标为 1。`reward_funcs` 收到 `prompts`、`completions`、`completion_ids` 和
rollout 返回的 extra fields，可读取 trajectory-level `r_alpha` 及 diagnostics。

`num_generations=8` 由 TRL sampler 保持 prompt group；generation batch 必须能被 8
整除。custom rollout 必须为每个 prompt 返回恰好 8 条 completion，不能把 8 条预生成
数据伪装成 online RL。由于 rollout API 只把 structured prompts 和 trainer 传给
callback，task ID 不得写进 policy-visible prompt；下一轮应通过
`environment_factory`/batch environment mapping 或一个明确的非泄漏 metadata bridge
取得 task ID，并在 GPU tiny preflight 中验证 mapping，不能使用 prompt 中的 evaluator
字段作为隐式 key。

GRPO config 必须显式写：

```yaml
loss_type: grpo
num_generations: 8
beta: 0.0
scale_rewards: group
learning_rate: 1.0e-6
```

`loss_type` 特别重要：当前 TRL 1.12/1.13 默认是 `dapo`，不能依赖默认值。exact
clipping、warmup、optimizer 和 LoRA target modules 属于 project implementation
choice，先不在无 GPU 环境拍板。

这套 API 仍是 TRL 标注的 experimental feature；本轮没有安装或 import 它，因此
上面的结论是 source-level compatibility，不是 runtime compatibility。

## 7. Dependency Pin Proposal

不在本轮安装。建议在独立 training environment 采用下面的初始约束，并在 GPU
preflight 记录实际解析版本：

```text
torch==2.8.0              # 现有 P1 记录；Blackwell/CUDA 由 GPU preflight 确认
transformers>=4.56.2,<5.0
trl==1.12.0               # 现有 P1 记录，且 source 已确认 rollout_func/env_mask
peft>=0.13,<0.20
accelerate>=1.4,<2
datasets>=4.7,<5
```

理由：TRL 1.12.0 的 package metadata 要求 `transformers>=4.56.2`、
`accelerate>=1.4`、`datasets>=4.7`；Qwen artifact 的 config metadata 写的是
`transformers 4.51.0`，这不是训练环境的强制 pin，不能直接与 TRL requirement
混用。TRL 1.13.0 也具备本轮需要的接口，但不是当前历史 remote record，因此先不
静默升级；如果 GPU environment 需要 1.13，应作为显式 dependency/provenance change。
vLLM、bitsandbytes、DeepSpeed 不属于第一版必需依赖。

## 8. Reward Interface

唯一 canonical training scalar：

```text
R_alpha = alpha * R_strict + (1 - alpha) * R_loose
0 <= alpha <= 1
```

因此：`strict → alpha=1.0`，`loose → alpha=0.0`，`--alpha 0.5` 代表代码级
mixed reward，不自动加入正式实验矩阵。CLI 只保留一个 canonical `reward_alpha`，
`--reward strict|loose` 与 `--alpha FLOAT` 互斥。无论 scalar 取值，evaluation 和
rollout diagnostics 始终保存：

```text
Rloose Rstrict Rsucc Rfinish Rcategory Rattribute Roption Rprice
```

## 9. Minimal File Plan

当前轮不创建这些训练文件；下一轮按以下最小边界实现：

| 路径 | 类型 | 用途 |
|---|---|---|
| `src/training/policy.py` | NEW SMALL MODULE | Qwen `from_pretrained`、可选 PEFT adapter、`generate(messages)` |
| `src/training/rollout.py` | NEW SMALL MODULE | 统一 AgentRollout；复用 prompt/protocol/env client |
| `src/training/eval.py` | NEW SMALL MODULE | 128/full manifest evaluator + 8 metrics aggregator |
| `src/training/sft_data.py` | NEW SMALL MODULE | accepted manifest → policy-visible chat records + assistant labels |
| `src/training/sft.py` | THIN WRAPPER | `SFTTrainer`/PEFT、epoch checkpoint/eval |
| `src/training/grpo.py` | THIN WRAPPER | explicit `GRPOConfig` + custom rollout/reward callbacks |
| `scripts/train_sft.py` | NEW SMALL MODULE | scenario-specific SFT run CLI |
| `scripts/train_grpo.py` | NEW SMALL MODULE | Direct/SFT-init GRPO CLI、reward selection |
| `scripts/eval_policy.py` | NEW SMALL MODULE | Base/SFT/GRPO 共用 evaluator CLI |
| `configs/experiments/*.yaml` | NEW SMALL CONFIG | 只写 scenario/mode/init/reward/output；公共参数集中读取 |

运行产物保持简单：`run_manifest.json`、`metrics.jsonl`、`checkpoints/`、`eval/`
和 stdout/stderr。训练 manifest 必须记录 git/upstream commit、model/tokenizer
paths（不写 secret）、scenario、reward_alpha、seed、dataset hashes、TRL/dependency
versions 和 checkpoint lineage。teacher raw/accepted schema 不改、不迁移。

## 10. GPU Dependency Boundary

### 无 GPU 可以完成

- formatter 的 evaluator-only leakage 静态检查；
- tokenizer-only chat template / assistant label preprocessing；
- reward alpha 与 8 metrics 的 pure logic；
- experiment config、run manifest、JSONL logging；
- AgentRollout 的 fake/local protocol unit tests（不启动远程 teacher collection）；
- dependency/source compatibility review。

### 必须 GPU-PREFLIGHT 才能确认

- Qwen3-8B BF16 load 与显存余量；
- LoRA rank、target modules、microbatch、gradient accumulation；
- single generation 与一条 ShopEnv episode；
- tiny SFT forward/backward；
- G=8 online group rollout；
- 一次 GRPO optimizer update、checkpoint save/reload；
- 训练吞吐、32K context 真实成本、是否需要 vLLM。

本轮不执行以上任何 GPU 操作，也不把 `GPU_RUNTIME_READY` 提前标为 YES。

## 11. Round B Plan

Round B 与 teacher collection 并行：policy/rollout/evaluator/formatter/SFTTrainer 的
代码实现不依赖 collection 完成。只有最终 accepted 数量、selected manifest/hash、
final SFT JSONL freeze 和正式 SFT training 需要等最终数据冻结。

1. 实现 `policy.py` 与 `rollout.py`，先只支持 Base/PEFT `generate(messages)`。
2. 实现 `sft_data.py`：只消费 policy-visible messages；拒绝
   `evaluator_only`、goal、reward metadata；不按 6000 写死。
3. 实现统一 evaluator 与 `eval_128_single` / `eval_128_single_persona` 输出，
   再接 full test counts（single 1459、persona 1343）。
4. 接 `SFTTrainer + PEFT`，建立每 scenario 独立 run/checkpoint/metrics；先完成
   CPU tokenizer/static checks，GPU forward/backward 留给 P5。

Round B acceptance：Base/SFT 使用同一 AgentRollout 语义；formatter 无 evaluator
leakage；8 metrics 与 reward wrapper 一致；checkpoint manifest 可解释且 scenario
隔离；不改变 teacher artifacts。

## 12. Round C Plan

1. pin 并安装训练 environment 后，写 `GRPOConfig` 和 explicit reward CLI。
2. 以 `environment_factory`/明确 batch mapping 构造 ShopEnv episode；custom
   `rollout_func` 返回每个 completion 的 token IDs、sampling logprobs、`env_mask`
   和 task/episode diagnostics。
3. 用 `reward_funcs` 将 `r_alpha` 传入 TRL；确认 G=8 的同 task group 不跨任务，
   zero-variance group 和 malformed/invalid action 统计进入 `metrics.jsonl`。
4. 实现 Direct Base→fresh LoRA 与 SFT adapter→GRPO 两条 lineage；不共用 scenario
   checkpoint。
5. GPU preflight 通过后再做 tiny one-update、checkpoint reload、128 eval，最后才
   扩展到 project config 的 200 updates。

Round C acceptance：是真 online rollout（不是预生成 response dataset）；ShopEnv
observation token 不进入 policy loss；`loss_type=grpo`、G=8、`beta=0`、group
scaling、LR=1e-6 全部来自显式 config；resume 不重置 task sampling/RNG；Base/SFT/
GRPO 共用 evaluator。

## 13. Changes Made This Round

新增本审计文档；没有修改 teacher collector、policy、config、artifact、SQLite 或
远程环境，也没有创建训练代码骨架。

## 14. Validation

本轮执行的只读检查：

- 仓库文件、现有 source、upstream env/agent/reward/eval 路径审阅；
- model/tokenizer 文件名、config/index/checksum manifest 存在性和 JSON metadata 读取；
- `importlib.metadata` 读取本地 package 版本（训练包均未安装）；
- 只读查询 PyPI TRL 1.12.0/1.13.0 source 与 metadata，确认 `rollout_func`、
  `env_mask`、`GRPOConfig` 字段和依赖约束。

```text
Paid API calls: NONE
GPU used: NO
Teacher collection interrupted: NO
Remote heavy workload started: NO
Training tests executed: NONE
Model loaded: NO
```

## 15. Git

本轮仅新增本文档，提交与工作树状态在完成静态检查后记录于最终回复。

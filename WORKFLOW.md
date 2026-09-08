# Shopping Agent RL
## A ShopSimulator RL Post-Training Reproduction
### 实施工作流 v1

**用途**：配套 `DESIGN.md` 使用。`DESIGN.md` 负责项目 scope、实验矩阵和参数决策；本文件只规定实施顺序、Codex/本地/远程 CPU/远程 GPU 的职责和验收方式。

---

# 1. 总体原则

项目采用：

```text
Codex：实现 / 复用 / 测试 / 汇报 / branch、commit、push、正常 merge
用户：审阅阶段输出、重要实验结果、重要设计/分析文档和项目级 open decisions
Git：唯一代码 source of truth
本地电脑：teacher API controller、collection、monitoring；大模型下载/大文件中转仍由用户负责
远程 CPU：ShopSimulator、search、reward、environment、数据与 profiling 支撑
远程 GPU：Base inference、SFT、GRPO、正式 evaluation
```

核心原则：

> **GPU 是执行层，不是开发层。**

GPU 启用前，应尽可能完成环境、数据、reward、teacher 数据、模型权重、配置、日志与训练脚本准备。

---

# 2. Codex 决策边界

Codex可以：
- 阅读 upstream；
- 最大化复用已有代码；
- 新增最小必要模块；
- 写测试；
- 准备脚本；
- 跑 CPU validation/profiling；
- 汇报 open decisions。

Codex不得自行：
- 修改 `DESIGN.md` 中 `[FROZEN]`；
- 缩小论文规模参数；
- 新增 scenario/model；
- 合并 Single 与 Single&Pers 训练；
- 启动大规模付费 teacher collection；
- 启动正式 SFT/GRPO；
- 删除正式 checkpoint；
- 重新生成已冻结 SFT dataset；
- 把 alpha 接口扩展为 mixed-reward 正式实验。

昂贵操作原则：

> **Codex 准备命令，用户最终触发 GPU/API 昂贵操作。**

---

# 3. Git 工作流

```text
Codex 修改代码
→ tests / git diff
→ git commit
→ git push
→ 执行批准版本
```

用户不需要逐文件 review source code；Codex 自主管理正常的 branch、commit、push 和 merge。远程服务器原则上不手工改 source code。

每个正式实验必须能映射到：
- project git commit；
- pinned upstream ShopSimulator commit；
- experiment config；
- seed；
- dataset manifest/hash；
- checkpoint；
- result。

---

# 4. 配置分层

## 4.1 Project Spec

例如 `configs/spec/project.yaml`。

保存 DESIGN.md 中已决定的：
- Qwen3-8B；
- Single / Single&Pers；
- Catalog-Fine；
- max_action_steps=30；
- seed=1；
- SFT 6000 success/scenario、3000 unique tasks/scenario；
- GRPO G=8、32 groups/update、200 updates、32K context。

Codex默认无权自行修改。

## 4.2 Experiment Config

例如：

```text
single_base.yaml
single_sft.yaml
single_direct_grpo_strict.yaml
single_sft_grpo_strict.yaml
single_persona_base.yaml
...
```

只定义 scenario、initialization、reward endpoint、checkpoint/output。

## 4.3 Runtime Config

只保存机器相关内容：

```text
model_path
data_path
cache_path
output_path
service_port
num_workers
```

机器路径变化不能改变实验定义。

---

# 5. 实施阶段

```text
P0  Upstream audit + project skeleton
P1  Remote CPU bootstrap
P2  Data + environment + reward + profiling
P3-0 API & collection infrastructure smoke
P3a Teacher profiling
P3b Collection policy freeze
P3c Formal collection
P3d Dataset freeze
P4  Model / large artifact staging
G0  GPU_READY Gate
P5  GPU preflight
P6  Base evaluation + SFT
P7  Core GRPO
P8  Final evaluation + analysis
P9  Optional Direct GRPO + Rloose
P10 Open-source / README / resume packaging
```

P10 只记录存在，当前不展开。

---

# 6. P0 — Upstream Audit + Project Skeleton

**GPU：不需要**

Codex第一步不是大量写代码，而是阅读 ShopSimulator 并提交 **Reuse Map**。

默认策略（总体遵循，可微调）：

| 功能 | 策略 |
|---|---|
| Shop environment | 直接复用 |
| Search/index | 直接复用 |
| Single-turn env logic | 直接复用或薄包装 |
| Agent prompt | 优先复用 |
| Action parser | 优先复用 |
| Persona injection | 优先复用 |
| Reward/scoring | 优先复用官方 scorer |
| Eval output format | 尽量兼容 |
| Teacher rollout | 新写 |
| SFT formatting | 新写 |
| SFT training | 新写/接成熟 trainer |
| GRPO online integration | 新写 integration |
| Logging | 新写最小必要部分 |

原则：**先证明 upstream 不能直接复用，再写新代码。**不把大量时间和会话上下文用在重写或准备环境。

P0 交付：
1. pinned upstream commit；
2. reuse map；
3. proposed project tree；
4. dependency list；
5.需要新增的模块；
6.open decisions。

P0 阶段输出供用户审阅；无 project-level conflict 时 Codex 可直接进入 P1。

---

# 7. P1 — Remote CPU Bootstrap

**GPU：不需要**

目标：把“只有基础 Python 的空服务器”准备成 attach GPU 后可直接执行正式项目的 persistent environment。

此阶段至少完成：
- Python/project environment；
- ShopSimulator dependencies；
- Shop environment service；
- search/index；
- data/model/cache/run dirs；
- training dependencies；
-可以提前安装的 CUDA-enabled PyTorch/training packages。

项目 RL backend 固定为 Hugging Face TRL；P1 不安装 ROLL、veRL、vLLM、SGLang、DeepSpeed 或 FlashAttention。

RTX PRO 6000 Blackwell 的 PyTorch/CUDA/backend 版本必须确认支持该架构，不允许随便固定旧版本。

---

# 8. P2 — Data + Environment + Reward + Profiling

**GPU：不需要**

生成并冻结：

```text
train_single.json
test_single.json
train_single_persona.json
test_single_persona.json

sft_task_manifest_single.json
sft_task_manifest_single_persona.json

eval_128_single.json
eval_128_single_persona.json
```

正式 SFT task IDs 和 128 eval IDs 生成后冻结，不重新随机。

Dataset profiling 至少包括：
- task counts；
- domain/category distribution；
- instruction/persona token length；
- attribute/option distribution；
- missing/invalid refs；
- train/test overlap；
- SFT stratified sample distribution。

以上dataset profiling的具体数据必须落在一个文档里供用户审阅。其中每个senario必须给出至少两个真实的完整数据例子，让用户清晰直观地理解“数据到底什么样”。

当前 public/upstream snapshot 的 Single&Pers train pool 实际为 3,323；论文报告的 3,383 作为 paper fact 保留，项目使用可复现的 3,323，不自行补齐或重新切分。

当前 Catalog-Fine snapshot 缺少 upstream 读取的 `query`。项目 compatibility layer 不生成 query；缺失时固定 `query_match=False`，并在 P2 reward validation 与正式实验结果中报告这一 public-snapshot deviation，不等待 upstream clarification。

Reward：

- 优先 wrap/reuse upstream scorer；
- 验证 8 metrics：
  `Rloose/Rstrict/Rsucc/Rfinish/Rcategory/Rattribute/Roption/Rprice`；
  -写 deterministic unit tests；
  -用官方 examples/outputs 做 parity sanity check。

GPU 阶段不应该再发现 reward 基础实现错误。

---

# 9. P3 — Teacher Rollout + SFT Dataset

**Remote CPU + Teacher API；不租 GPU。** Teacher controller 在本地运行，ShopSimulator
留在 remote CPU，通过 SSH tunnel + HTTP 访问；不要在本地复制 ShopSimulator、Lucene、
Pyserini、Java 或 spaCy。

远程 CPU 的正式 rollout 链为：

```text
local controller → teacher API
local controller → SSH tunnel → ShopSimulator.reset/step
→ textual action parse → ShopEnv.step → observation → … → reward
```

## 9.1 P3-0 API & collection infrastructure smoke

先建立并验证薄 `TeacherClient`（chat-completions/responses adapter、15s connect /
300s read timeout、最多 3 次 transient retry、Retry-After、诊断与 failure 分类）、
SQLite + atomic JSON durable storage、immutable run manifest/resume、graceful Ctrl+C、
remote task-scoped environment adapter、SSH up/down 脚本和 local/remote 分离 smoke。
P3-0 不调用真实 teacher API、不批量采集 trajectory、不生成 SFT 数据。

## 9.2 P3a Teacher profiling — COMPLETE

P3-0 通过后已用 single worker 完成 canonical profiling，并完成 multi-session concurrency
extension 与 2/8/10-worker probe。Canonical statistics 见 `docs/P3A_TEACHER_PROFILE.md`。

P3a 使用每个 scenario 24 条、来自冻结 primary SFT TRAIN manifest 的 deterministic
profiling tasks（seed=1）。每 task 的 first-success 最多 3 次尝试，取得第一条 success 后
再做最多 2 次独立 second-demo exploration；`max_action_steps=30`、单 worker。上述数字仅是
profiling operational limits，**不是**正式 collection caps。P3a 输出物理隔离于
`data/teacher_raw/`，不得直接进入 SFT。

## 9.3 P3b Collection policy freeze — COMPLETE

`p3b-v1` 已冻结：每 scenario hard target 6,000 accepted trajectories；约 3,000 unique 是
coverage target；default workers=8；Pass A first-success cap=2、deterministic same-stratum
reserve；Pass B 最多 2 次；B+C post-success budget=3；每 task accepted 1/2/3；exact
behavioral duplicate hard reject、near duplicate diagnostic-only。Machine-readable source of
truth 是 `configs/teacher/formal_collection_p3b_v1.yaml`，解释与证据见
`docs/P3B_COLLECTION_POLICY.md`。

## 9.4 P3c Formal Collector Implementation — NEXT

下一阶段实现 formal A/B/C scheduler、default 8-worker queue、deterministic reserve、
acceptance/duplicate/budget accounting、durable resume/global stop、append-only
`data/teacher_raw/collector.log`、`scripts/inspect_teacher_run.py --latest/--last N` 与 immutable
policy version/hash。先跑 no-paid tests，再由用户触发 tiny Responses compatibility/formal
smoke；不得在实现阶段直接开始 6,000 trajectory collection。

## 9.5 P3d Dataset freeze

生成 `sft_single.jsonl` 与 `sft_single_persona.jsonl`，并保存 dataset manifest/hash、teacher
model/version/generation config、source task manifest 与 collection stats；正式数据生成后
冻结，重做必须作为新的 data version。

---

# 10. P4 — Model / Large Artifact Staging

**GPU：不需要**

由于 remote 从模型网站下载慢：

```text
Model Hub
→ 本地电脑高速下载
→ 第三方高速网盘/中转
→ Remote persistent disk
```

建议模型目录：

```text
/data/models/Qwen3-8B/
```

本地生成 `model_manifest.json`：
- filename；
- size；
- sha256。

远程传输完做 checksum verification。

原则：

> 不在开 GPU 后才等待大模型下载。

---

# 11. GPU_READY Gate

只有以下全部 PASS 才开 GPU：

- repo/upstream pinned；
- Python environment；
- ShopSimulator service；
- Catalog-Fine/search index；
- official splits；
- fixed SFT task manifests；
- fixed 128 eval subsets；
- dataset profiling；
- reward tests；
- formal teacher dataset；
- 12K successful trajectories；
- SFT JSONL；
- Qwen3-8B weights；
- model checksum；
- experiment configs；
- logging/output dirs；
- training dependencies/backend；
- disk capacity；
- GPU commands ready。

脚本：

```bash
bash scripts/check_gpu_ready.sh
```

目标：

```text
GPU_READY=YES
```

否则不开 GPU。

---

# 12. P5 — GPU Preflight

启用 GPU 后第一步：

```bash
bash scripts/gpu_preflight.sh
```

检查：
- nvidia-smi；
- CUDA/PyTorch/Blackwell support；
- Qwen3-8B load；
- LoRA attach；
- forward/backward；
- generation；
- ShopEnv episode；
- checkpoint save/reload；
- GRPO backend；
- one-update smoke test（如适用）。

Smoke test 只用于 correctness，不计入正式实验。

---

# 13. P6 — Base Evaluation + SFT

## 13.1 Base

正式运行：

```text
Qwen3-8B
→ Single full eval
→ Single&Pers full eval
```

得到 Base Table 3-style rows，以及真实 trajectory length/context/throughput 数据。

## 13.2 SFT

分别：

```text
Single → SFT_single
Single&Pers → SFT_single_persona
```

使用 DESIGN.md 第一版规模：
- LoRA/PEFT；
- 4 epochs；
- effective batch=32；
- once-per-epoch checkpoint；
- fixed 128 eval each epoch；
- final full eval。

---

# 14. P7 — Core GRPO

原则：

> **一个命令 = 一个完整正式实验。**

逻辑命令：

```bash
bash scripts/run_experiment.sh single_direct_grpo_strict
bash scripts/run_experiment.sh single_sft_grpo_strict
bash scripts/run_experiment.sh single_persona_direct_grpo_strict
bash scripts/run_experiment.sh single_persona_sft_grpo_strict
```

脚本负责：
-读取 config；
-确认 environment；
-加载 model/adapter；
-记录 git/config；
-执行 online rollout；
-执行 GRPO update；
-checkpoint；
-fixed-128 eval；
-resume/logging。

正式规模仍按 DESIGN.md：

```text
G=8
32 task groups/update
200 updates
```

不预先缩水。

---

# 15. 长时间训练与 GPU 使用

正式 experiment 必须：
- resumable；
- checkpointed；
- logged。

建议 tmux 或等效持久运行。

自动 cadence：

```text
step 20 → checkpoint + eval
...
step 200 → checkpoint + eval
```

失败保留：
- latest checkpoint；
- stdout/stderr；
- metrics；
- run metadata。

GPU可阶段性关闭：

```text
开 GPU
→ 跑批准实验
→ 关 GPU
→ CPU 分析
→ 决定下一实验
→ 再开 GPU
```

避免 GPU 在分析阶段空转。

---

# 16. P8 — Final Evaluation + Analysis

最终正式比较：

```text
Single:
  Base
  SFT
  Direct GRPO Strict
  SFT + GRPO Strict

Single & Pers:
  Base
  SFT
  Direct GRPO Strict
  SFT + GRPO Strict
```

完整 held-out test：
- Single 1459；
- Single&Pers 1343。

统一报告 8 metrics。

CPU 负责：
- aggregation；
- Table 3-style tables；
- training curves；
- rollout stats；
- deviation report；
- compute/resource summary。

完成后 v1 核心项目完成。

---

# 17. P9 — Optional Paper Experiment

只有核心 v1 完成后，再考虑：

```text
Direct GRPO + Rloose
```

它属于最高优先级 optional-paper experiment，优先级高于论文外研究。

---

# 18. Codex Task Contract

以后给 Codex 的任务尽量使用：

```text
Goal
Allowed
Must reuse
Forbidden
Expected output
Acceptance
```

例如：

```text
Goal:
实现 official split manifest。

Allowed:
读取 upstream data/schema；
新增 src/data/* 和 tests/data/*。

Must reuse:
官方 task IDs 和 split。

Forbidden:
重新随机 train/test split；
修改 persona；
修改 Catalog-Fine；
新增外部数据。

Expected output:
scenario manifests；
统计报告；
tests。

Acceptance:
指定 pytest/validation commands 全部通过。
```

---

# 19. Codex 标准汇报格式

每次完成任务固定输出：

```text
Implemented
Reused from upstream
Files changed
Validation
Artifacts
Open decisions
Deviation from DESIGN.md
Next command
```

没有问题时明确：

```text
Open decisions: None
Deviation: None
```

---

# 20. Remote Storage Layout

建议：

```text
/workspace/shopsim-rl   # code
/data/shopsim           # official/manifests/teacher/SFT
/data/models            # Qwen3-8B
/runs                   # checkpoints/logs/evaluation
```

具体根目录可由 runtime config 调整。

---

# 21. Secrets

Teacher API key 不进 Git。

使用 `.env` 或环境变量：

```text
TEACHER_API_KEY
```

禁止：
- key 写 YAML；
- key 出现在 logs；
- key commit。

---

# 22. 阶段验收总表

| 阶段 | 内容 | GPU | Codex自主执行 | 用户负责 |
|---|---|---:|---|---|
| P0 | Upstream audit + skeleton | 否 | 是 | review |
| P1 | Remote bootstrap | 否 | 是 | 执行/验收 |
| P2 | Data + reward + profiling | 否 | 是 | review |
| P3-0 | API & collection infrastructure smoke | 否 | 是；不调用真实 teacher API | 本地配置后 smoke |
| P3a | Teacher profiling | 否 | 小规模 profiling | 触发/验收 |
| P3b | Collection policy freeze | 否 | 根据 profiling 固定参数 | review |
| P3c | Formal collection | 否 | 需批准后执行 | 配 key/批准 |
| P3d | Dataset freeze | 否 | formatter/hash/freeze | review |
| P4 | 大文件 staging | 否 | 辅助 | 本地下载/传输 |
| G0 | GPU_READY | 否 | 是 | 决定开卡 |
| P5 | GPU preflight | 是 | 准备脚本 | 触发 |
| P6 | Base + SFT | 是 | 准备脚本 | 批准/触发 |
| P7 | GRPO strict | 是 | 准备脚本 | 批准/触发 |
| P8 | Final eval + analysis | 部分 | CPU分析可 | 结果判断 |
| P9 | Direct GRPO loose | 是 | 可选 | 决定是否做 |
| P10 | Open-source / docs / resume packaging | 少量或无 | 后期再定 | 项目展示 |

---

# 23. P10 — 项目完成后的开源与简历整理

**当前只记录，不展开。**

项目主体完成后，可以把 GitHub repository 整理成一个供他人学习和复现的公开项目，并同时完成简历展示。

届时可考虑：
- README；
- setup / data / SFT / GRPO / eval 文档；
- reproducibility commands；
- paper deviations；
-实验表、训练曲线；
-架构图；
-可公开 artifact 说明；
-项目简历 bullet；
-面试讲解版本；
-明确 upstream 复用与本人实现边界。

这一阶段的目标是：

> **把已经完成的 solid 工程与实验清晰呈现出来，而不是重新扩大 scope。**

现在不展开 P10，也无需为了实现P10调整之前的计划或决策。

---

# 24. 实施总流程

```text
Codex 实现
→ 用户 review
→ Remote CPU 准备到 100%
→ Teacher/SFT 数据完成
→ 模型权重到位
→ GPU_READY
→ 开 GPU
→ Preflight
→ 一个命令一个正式实验
→ 关 GPU
→ CPU 分析
→ 批准下一实验
```

核心目标：

> **GPU 启用不是“开始开发”，而是“执行已经准备好的正式实验”。**

---

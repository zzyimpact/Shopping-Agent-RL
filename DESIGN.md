# Shopping Agent RL
## A ShopSimulator RL Post-Training Reproduction
### 技术设计 v1

**文档状态**：Scope Frozen / Implementation v1  
**日期**：2026-09-05  
**项目定位**：基于论文 *ShopSimulator: Evaluating and Exploring RL-Driven LLM Agent for Shopping Assistants* 的任务、数据、环境与训练思路，完成一个以 **LLM post-training / online agent RL / GRPO** 为核心的可运行复现项目。  
**主要用途**：项目实施、Codex/开发协作、实验记录、最终 README/简历材料的上游设计文档。

**项目语言约定**：Codex 回复、项目文档和新写代码的注释/docstring 以中文为主；RL、LLM、SFT、GRPO、rollout、checkpoint、PEFT、LoRA、API 等术语可使用英文。upstream 原有代码不因语言约定修改。

---

# 0. 文档使用约定

本项目不是“逐字逐参数复刻论文”，也不是为了追求论文 Table 3 的绝对数值。目标是：

> **依托论文公开的 ShopSimulator 数据与环境，使用论文的 Qwen3-8B、SFT cold-start、GRPO、loose/strict reward 和 Table 3-style evaluation 思路，补齐并跑通一个完整的 agentic RL post-training 闭环。**

本项目允许工程实现和部分实验规模根据实际硬件与运行情况调整，但所有调整必须明确记录，不能在实现过程中静默改变实验定义。

本文使用以下标签：

- **`[FROZEN]`**：已冻结的项目定义。原则上不能因为“跑得慢”随意修改；如需修改，必须重新讨论项目 scope。
- **`[PAPER-DEFAULT / ADJUSTABLE]`**：论文明确给出的设置。第一版直接按论文设置，不预先缩水；只有真实 profiling、OOM、wall-clock 或其他明确证据表明不可接受时，才允许调整。
- **`[PROJECT-FIXED]`**：论文没有明确规定或存在歧义，由本项目自行定义并暂时冻结的实现选择。
- **`[PROJECT-CHOICE / CHANGEABLE-BEFORE-RUN]`**：实现方向已确定，但具体对象尚未最终冻结。例如强 teacher 的具体 API 模型。正式生成数据或训练开始后应冻结。
- **`[OPTIONAL-PAPER]`**：不是最小必做，但属于论文实际报告过的实验，可在算力和时间充足时补做。
- **`[OPTIONAL-COMPATIBLE]`**：使用论文已有的模型、reward、训练组件，但论文未在 Table 3 中报告该具体组合。
- **`[OUTSIDE-PAPER / NOT-V1]`**：论文严格复现之外的探索。可以在代码接口中预留，但 v1 不作为训练目标。
- **`[OBSERVE]`**：不是预先拍脑袋决定的超参数，而是必须通过 dataset/rollout profiling 实际测出来的量。

---

# 1. 项目目标与非目标

## 1.1 核心目标

项目必须形成下面这个真实闭环：

```text
Official ShopSimulator tasks
        ↓
Qwen3-8B policy
        ↓
multi-step online rollout in ShopEnv
        ↓
trajectory + terminal purchase
        ↓
reward computation
        ↓
GRPO group-relative update
        ↓
updated policy
        ↓
next online rollout
```

并完整覆盖：

1. 官方数据读取、验证、scenario 切分与 profiling；
2. Qwen3-8B Base baseline；
3. 强 teacher 在 ShopSimulator 中真实 rollout；
4. environment-based success filtering；
5. SFT cold-start；
6. Base → Direct GRPO；
7. SFT → GRPO；
8. held-out evaluation；
9. Table 3-style 全 reward/metric 分项报告；
10. 训练过程中的 checkpoint、固定 eval subset 和基本 RL 诊断日志。

核心价值必须落在 **训练闭环与 RL**，而不是数据爬取、搜索基础设施、通用框架包装或模型规模堆叠。

## 1.2 明确不做

v1 不做：

- harness/general benchmark framework；
- Multi-Turn；
- Multi-Turn & Personalization；
- Shopper LLM；
- Qwen3-14B 或其他模型规模；
- small-model-first 正式路线；
- 重新爬取淘宝数据；
- 重新构造商品 catalog；
- 重新人工构造 shopping tasks；
- 重新生成 persona；
- self-generated SFT 主方案；
- 新 RL 算法；
- 全四 scenario 完整复现；
- 追求论文绝对性能数值；
- 多 seed 学术级统计显著性；
- mixed reward / reward curriculum 正式实验；
- 把 infra/distributed training 当项目主体。

---

# 2. 论文与公开仓库：本项目依赖的事实

## 2.1 数据与任务

论文报告：

- 总计约 28K tasks；
- 约 25K train、2.8K eval；
- 四种 scenario：Single-Turn、Single-Turn & Personalization、Multi-Turn、Multi-Turn & Personalization；
- Non-personalized instructions：Train **21,962** / Test **1,459**；
- Personalized instructions：Train **3,383** / Test **1,343**；
- 同一 instruction 可用于 single-turn 和 multi-turn；
- Catalog-Fine 约 **24K** curated products；
- 后续 evaluation/training 使用 Catalog-Fine，而不是 1.3M Catalog-Full；
- 每个最细子类约有 120 个高度相似商品；
- Product attribute count：mean 4.53, median 4；
- Product option count：mean 17.12, median 7。

**Paper anchors**：Figure 2, §2.2, §2.4。

## 2.2 Single-Turn interaction

Single-turn 仍然是多步 agent-environment sequential decision process，而不是“一次生成一个回答”。

Agent 仍需要 search / browse / click / inspect / select option/SKU / purchase。

Single-turn 最大 action step：**30**。

**Paper anchors**：§2.1, §3.1。

## 2.3 Reward

论文核心 reward：

\[
R_{loose}
=
R_{cat}
\cdot
\frac{
|U_{att}\cap Y_{att}|
+
|U_{opt}\cap Y_{opt}|
+
\mathbf{1}[Y_{price}\le U_{price}]
}{
|U_{att}|+|U_{opt}|+1
}
\]

\[
R_{strict}
=
R_{cat}
\cdot
rac{|U_{att}\cap Y_{att}|}{|U_{att}|}
\cdot
rac{|U_{opt}\cap Y_{opt}|}{|U_{opt}|}
\cdot
\mathbf{1}[Y_{price}\le U_{price}]
\]

Strict 是 bottleneck/multiplicative reward。

Appendix 进一步说明 category/type consistency、attribute fuzzy matching、option fuzzy matching、price hard indicator。

论文记号中存在 `Rcat / Rtype / Rcategory` 命名差异。本项目代码统一使用 `r_category`，但语义必须与论文/官方评分实现对齐。

**Paper anchors**：§2.3, Appendix C.3。

## 2.4 Table 3 evaluation metrics

统一报告 **8 个指标**：

1. `Rloose`
2. `Rstrict`
3. `Rsucc`
4. `Rfinish`
5. `Rcategory`
6. `Rattribute`
7. `Roption`
8. `Rprice`

其中：

- `Rloose / Rstrict`：可作为 GRPO scalar training objective；
- `Rsucc`：所有 required dimensions 均精确满足时的 full success；
- `Rfinish`：是否最终完成推荐/购买，不考虑准确性；
- category/attribute/option/price：能力诊断指标。

## 2.5 论文训练设置

论文训练对象：**Qwen3-8B**

SFT：
- 6K successful GPT-4.1 trajectories；
- batch size 32；
- learning rate 1e-5；
- 4 epochs。

RL：
- ROLL framework；
- Project backend：Hugging Face TRL（相对论文 ROLL 的 deviation）；
- GRPO；
- Rloose / Rstrict；
- omit KL loss；
- max context 32K；
- learning rate 1e-6；
- 200 training steps；
- each step: 32 samples × 8 trajectory rollouts。

论文说明 RL 按 scenario 分别训练。

**重要歧义**：论文只写“collect 6K successful trajectories”，未明确 6K 是每个 scenario、全部 scenario 合计，还是共享 SFT 数据。本项目必须显式记录自己的处理方式。

**Paper anchors**：§4.1, Appendix D.1。

## 2.6 公开 GitHub 仓库

项目依托：

`https://github.com/ShopAgent-Team/ShopSimulator`

截至 2026-09-05 检查 main 分支，顶层主要包括：

```text
assets/
multi_eval/
shop_env/
single_eval/
README.md
get_score.py
```

README 主要提供 shopping environment、single/multi-turn evaluation 和 result scoring。当前 main 顶层未展示与论文 SFT/GRPO 实验对应的完整 training module，因此本项目的主要工程工作是补齐 post-training pipeline，而不是简单 clone 后运行训练脚本。

实现原则：优先保持 upstream ShopSimulator 原代码可追踪，不大规模魔改；新增训练代码放在本项目独立目录中，通过 integration layer 调用 upstream environment。

---

# 3. 已冻结的项目定义

## 3.1 模型 `[FROZEN]`

```text
Policy model = Qwen3-8B
```

不做 14B、其他模型对比、小模型正式路线或 scaling study。

## 3.2 Scenario `[FROZEN]`

只做两个独立 track：

```text
Track A: Single-Turn
Track B: Single-Turn & Personalization
```

分别训练、分别生成 SFT 数据、分别 GRPO、分别评估。

禁止把两个 scenario 合成一个 joint SFT / joint GRPO policy，除非未来重新讨论 scope。

## 3.3 Catalog / split `[FROZEN]`

```text
Catalog = Catalog-Fine
```

使用官方 train/test split，不重新随机切。

任务池：

```text
Single:
  train = 21,962
  test  = 1,459

Single & Pers:
  train = 3,383
  test  = 1,343
```

## 3.4 Max action `[FROZEN]`

```text
max_action_steps = 30
```

不使用 max action 作为节省算力的第一手段。

## 3.5 参数更新方式 `[PROJECT-FIXED]`

SFT 与 GRPO 均使用：

```text
Qwen3-8B + LoRA/PEFT
```

基础权重优先 BF16，不预先引入 4-bit QLoRA。

LoRA rank、target modules 等后续再冻结。

## 3.6 Random seed `[PROJECT-FIXED]`

```text
training_seed = 1
```

每个 configuration 一个正式 training seed。

---

# 4. 实验矩阵

## 4.1 最小必须完成 `[FROZEN / MUST]`

对每个 scenario：

### E0 — Base
```text
Qwen3-8B Base → held-out evaluation
```

### E1 — SFT
```text
Qwen3-8B Base → scenario-specific SFT → held-out evaluation
```

### E2 — Direct GRPO + Strict
```text
Qwen3-8B Base → online GRPO with Rstrict → held-out evaluation
```

### E3 — SFT + GRPO + Strict
```text
Qwen3-8B Base
→ scenario-specific SFT
→ online GRPO with Rstrict
→ held-out evaluation
```

v1 最小 RL training runs：

```text
2 scenarios × 2 RL runs/scenario = 4 GRPO runs
```

## 4.2 最高优先级可选论文复现 `[OPTIONAL-PAPER / PRIORITY-1]`

### E4 — Direct GRPO + Loose

```text
Qwen3-8B Base
→ online GRPO with Rloose
→ held-out evaluation
```

这是论文 Table 3 实际报告过的实验。

## 4.3 论文组件兼容但未在 Table 3 报告 `[OPTIONAL-COMPATIBLE]`

例如：

```text
SFT → GRPO with Rloose
```

不在 v1 必做范围。

## 4.4 Mixed reward abstraction `[PROJECT-CODE-DESIGN]`

代码层统一实现：

\[
R_alpha=\alpha R_{strict}+(1-\alpha)R_{loose}
\]

```text
alpha = 1.0 → Rstrict
alpha = 0.0 → Rloose
```

`0 < alpha < 1`、dynamic alpha、curriculum 均为 `[OUTSIDE-PAPER / NOT-V1]`。

---

# 5. 数据 pipeline

整体：

```text
Official ShopSimulator data
        ↓
schema inspection
        ↓
split validation
        ↓
scenario manifests
        ↓
dataset profiling
        ↓
teacher rollout task selection
        ↓
teacher online trajectories
        ↓
success filtering
        ↓
SFT dataset formatting
```

## 5.1 官方数据处理 `[FROZEN]`

不重新造 catalog / instruction / task / persona / search index semantics。

只做：

1. unpack / load；
2. schema inspection；
3. task_id / product_id integrity validation；
4. official split validation；
5. selected scenario manifests；
6. policy-visible / evaluator-hidden field separation；
7. profiling。

## 5.2 数据泄漏规则 `[FROZEN]`

Policy 不得看到：

- gold target product；
- hidden reward fields；
- ground-truth required option labels（除非其本来属于用户可见 instruction/persona）；
- eval-only metadata。

Gold 信息仅允许 reward evaluator / offline analysis 使用。

必须 assertion：

```text
train_ids ∩ test_ids == ∅
```

---

# 6. Dataset profiling

Profiling 服务于 data quality、compute estimation、context estimation、teacher budget、RL rollout budget。

## 6.1 训练前 `[OBSERVE]`

每个 scenario 统计：

- train/test task count；
- domain/category distribution；
- instruction token length；
- persona token length（Pers only）；
- attribute count distribution；
- option count distribution；
- target price distribution；
- missing field / invalid reference；
- train/test overlap。

## 6.2 Rollout 后 `[OBSERVE]`

Base 和 teacher：

- actions per trajectory；
- p50/p90/p95/p99 action length；
- full trajectory token length；
- p50/p90/p95/p99 token length；
- finish rate；
- success rate；
- `Rstrict` / `Rloose` distribution；
- zero-reward rate；
- teacher success rate；
- rollout throughput（trajectories/hour）；
- token generation throughput。

GRPO group：

- within-group reward mean；
- within-group reward std；
- zero-variance group ratio。

这些是 debugging/observability，不是额外研究实验。

---

# 7. SFT teacher 数据设计

## 7.1 Teacher 方法 `[FROZEN]`

```text
strong teacher
→ real interaction with ShopSimulator
→ environment evaluation
→ successful trajectory filtering
→ SFT
```

不采用 self-generated SFT 作为主线。

## 7.2 Teacher 具体模型 `[PROJECT-CHOICE / CHANGEABLE-BEFORE-RUN]`

当前暂定：

```text
GPT-5.6 Sol API
```

具体 strong teacher 可在正式数据采集前替换。

一旦正式 SFT trajectory collection 开始，冻结 teacher model/version/config；正式数据集中不要混用多个 teacher，除非显式记录为单独数据版本。

论文使用 GPT-4.1；本项目使用其他 teacher 时必须写入 Deviation from Paper。

---

# 8. SFT 数据规模

## 8.1 总量 `[PROJECT-FIXED]`

每个 scenario：

```text
6,000 successful trajectories
```

两 scenario 合计：

```text
12,000 successful trajectories
```

必须明确记录：论文未说明 6K 是 per-scenario 还是 total；本项目定义为 6K per selected scenario。

## 8.2 Unique task / demo 比例 `[PROJECT-FIXED]`

目标：

```text
3,000 unique tasks / scenario
×
2 successful trajectories / task
=
6,000 SFT trajectories / scenario
```

两个 scenario 使用同一策略。

## 8.3 不采用不同覆盖方式

不采用：

```text
Pers: 3383 tasks × 2
Single: ~6766 tasks × 1
```

避免同时改变 unique-task coverage、demonstrations/task 和 trajectory diversity structure。

## 8.4 Coverage ratio caveat

Single：

```text
3000 / 21962 ≈ 13.7%
```

Pers：

```text
3000 / 3383 ≈ 88.7%
```

因此最终解释原则：

> 最可信的是 scenario 内部 Base/SFT/GRPO 的比较；跨 scenario 的绝对提升差异只作为现象，不做严格因果结论。

---

# 9. SFT task sampling `[PROJECT-FIXED]`

使用 stratified sampling，尽量保持原 scenario training pool 的 domain/category 分布：

```text
official train pool
→ stratified select ~3000 unique tasks
```

---

# 10. Teacher trajectory collection

## 10.1 Coverage first `[PROJECT-FIXED]`

Pass 1：

```text
尽量给全部 selected 3000 tasks 获取
1 successful trajectory/task
```

Pass 2：

```text
再获取第 2 条 successful trajectory/task
```

优先 breadth，再增加 demonstration diversity。

## 10.2 Success criterion `[PROJECT-DEFAULT / ADJUSTABLE]`

论文只写 “successful trajectories”。

v1 默认：

```text
Rsucc == 1
```

才进入 SFT dataset。

若官方代码存在更明确 successful semantics，优先与官方对齐并记录。

## 10.3 Duplicate control `[PROJECT-DEFAULT / ADJUSTABLE]`

同 task 第二条 trajectory：

- 必须成功；
- 不应与第一条完全/近乎完全重复。

可使用轻量检查：

- action-type sequence；
- search query sequence；
- clicked product sequence；
- normalized action string；
- text similarity。

threshold 后续根据真实 teacher 轨迹确定。

## 10.4 Difficult task / attempt cap `[PROJECT-DEFAULT / ADJUSTABLE]`

需要：

```text
max_teacher_attempts_per_task
```

超过 cap 仍不足成功轨迹：

- 标记 teacher-failure；
- 从同一 sampling stratum 选 replacement task；
- 保持最终 success trajectory 总量和大致 distribution。

具体 cap 根据 teacher success-rate profiling 决定。

---

# 11. SFT trajectory 数据格式

推荐 JSONL/sharded JSONL。

概念 schema：

```json
{
  "task_id": "...",
  "scenario": "single | single_persona",
  "policy_input": {
    "instruction": "...",
    "persona": null
  },
  "messages": [],
  "actions": [],
  "observations": [],
  "final_purchase": {},
  "reward": {
    "loose": 0.0,
    "strict": 0.0,
    "succ": 0,
    "finish": 0,
    "category": 0.0,
    "attribute": 0.0,
    "option": 0.0,
    "price": 0.0
  },
  "teacher_metadata": {}
}
```

hidden goal/evaluator fields 可以保存在 raw artifact 中，但 SFT formatter 必须与 policy-visible message 严格拆开。

---

# 12. SFT training

## 12.1 两条独立 SFT track `[FROZEN]`

```text
SFT_single
SFT_single_persona
```

不共享一个混合 SFT checkpoint 作为正式实验 checkpoint。

## 12.2 模型更新 `[PROJECT-FIXED]`

```text
Qwen3-8B BF16 base + LoRA/PEFT
```

## 12.3 Paper-scale 参数 `[PAPER-DEFAULT / ADJUSTABLE]`

```text
epochs = 4
effective_batch_size = 32
```

论文 learning rate = 1e-5。

本项目 exact LR、LoRA rank、warmup 等后续单独冻结。

## 12.4 Loss masking `[PROJECT-DEFAULT / NOT-YET-FROZEN]`

推荐：

```text
assistant/action tokens participate in CE loss
system/user/environment observation tokens masked
```

即 assistant-only causal SFT。

论文没有公开 token-level preprocessing，因此正式编码前可再确认。

## 12.5 SFT checkpoint/eval cadence `[PROJECT-FIXED]`

```text
checkpoint once per epoch
intermediate eval once per epoch
```

固定 128-task held-out subset。

第 4 epoch 后做完整 test evaluation。

---

# 13. Reward implementation

## 13.1 统一接口 `[PROJECT-FIXED]`

```python
reward = alpha * r_strict + (1 - alpha) * r_loose
```

同时永远计算：

```text
r_loose
r_strict
r_succ
r_finish
r_category
r_attribute
r_option
r_price
```

训练 reward 与 evaluation metrics 分开存。

## 13.2 Endpoints

```text
alpha = 1.0 → strict
alpha = 0.0 → loose
```

v1 必做：`alpha = 1.0`。

Optional-paper Direct GRPO loose：`alpha = 0.0`。

## 13.3 Reward correctness

必须：

1. 对齐论文 §2.3 + Appendix C.3；
2. 尽可能对照官方 `get_score.py` / repo scorer；
3. 添加 deterministic unit tests；
4. 使用官方公开 evaluation outputs 做 sanity check；
5. 明确处理 missing attribute、empty option、no purchase、over-price、invalid action/timeout。

未经记录不得自行“改进”论文 reward semantics。

---

# 14. Base evaluation

开始任何 SFT/RL 前，两个 scenario 都先跑 Base full evaluation：

```text
Qwen3-8B Base
→ full agent rollout
→ reward evaluation
```

至少得到：

- Table 3-style 8 metrics；
- action length distribution；
- token length distribution；
- finish rate；
- reward distributions。

主要目的：验证环境、action parser、reward，并建立 compute/context profiling。

---

# 15. GRPO online training

GRPO 必须是真正 online agent RL，不允许把预生成 response dataset 交给普通 prompt-response trainer 后称为完成。

核心 loop：

```text
sample training tasks
        ↓
for each task generate G trajectories
        ↓
each trajectory repeatedly:
    policy.generate
    → parse action
    → ShopEnv.step
    → receive observation
    → continue
        ↓
episode ends
        ↓
compute scalar reward + all metrics
        ↓
build GRPO groups
        ↓
group-relative advantage
        ↓
policy update
        ↓
updated policy used in next rollout
```

这是项目最重要的 Definition of Done。

## 15.1 初始化

Direct GRPO：

```text
Qwen3-8B Base
+ fresh RL LoRA
→ GRPO
```

SFT + GRPO：

```text
Qwen3-8B Base
+ scenario-specific SFT LoRA
→ continue GRPO
```

不同 scenario/config adapters/checkpoints 分开存。

---

# 16. GRPO 规模

## 16.1 Group size `[PAPER-DEFAULT / ADJUSTABLE]`

```text
G = 8 trajectories / task
```

如需缩规模时，最优先希望保留。

## 16.2 Samples per update `[PAPER-DEFAULT / ADJUSTABLE]`

```text
32 task groups / update
```

标准 update：

```text
32 × 8 = 256 trajectories
```

实现可用 microbatch / gradient accumulation。

## 16.3 RL steps `[PAPER-DEFAULT / ADJUSTABLE]`

```text
200 updates / run
```

每个 RL run：

```text
200 × 32 × 8 = 51,200 trajectories
```

## 16.4 v1 最小 RL rollout budget

```text
2 scenarios
× 2 mandatory RL configs
× 51,200
= 204,800 online RL trajectories
```

这是目标规模，不预先缩水。

## 16.5 Context `[PAPER-DEFAULT / ADJUSTABLE]`

```text
max_context = 32K
```

第一版直接用论文值。

只有实际 profiling 表明 p99 trajectory length 远低于 32K，且 32K 明显造成吞吐/显存浪费时才调整。

## 16.6 KL `[PAPER-DEFAULT]`

v1：

```text
KL penalty = off
```

未来测试 KL 属于额外 ablation。

## 16.7 Optimization backend `[PROJECT-FIXED]`

论文使用 ROLL。

本项目使用 Hugging Face TRL；不安装或并行实现 ROLL、veRL。

不论后续采用 TRL 的 environment factory、custom rollout function 或其他薄 adapter，都必须保持 online ShopEnv rollout、group-size semantics、reward semantics、Base vs SFT initialization、checkpoint/eval semantics。

后续只实现最薄的 ShopSimulator ↔ TRL integration，不重写完整 distributed RL infra。

---

# 17. Evaluation

## 17.1 Intermediate eval `[PROJECT-FIXED]`

每个 scenario 从官方 held-out test 固定抽：

```text
128 tasks
```

从项目开始固定，所有 checkpoint 和所有方法都用同一份 IDs。

## 17.2 RL cadence `[PROJECT-FIXED]`

```text
checkpoint every 20 GRPO steps
intermediate eval every 20 GRPO steps
```

即：

```text
20, 40, 60, ..., 200
```

## 17.3 Final evaluation `[FROZEN]`

最终结果跑完整官方 test：

```text
Single: 1,459
Single & Pers: 1,343
```

128 subset 不能替代最终结果。

---

# 18. 最终结果表

每个 scenario 至少：

| Method | Rloose | Rstrict | Rsucc | Rfinish | Rcategory | Rattribute | Roption | Rprice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | | | | | | | | |
| SFT | | | | | | | | |
| Direct GRPO + Strict | | | | | | | | |
| SFT + GRPO + Strict | | | | | | | | |
| Direct GRPO + Loose* | | | | | | | | |

`*` optional-paper。

跨 scenario 比较仅作为辅助现象；主要解释 within-scenario。

---

# 19. “规模参数”的正式分类

## 19.1 第一类：不可作为算力旋钮随意修改 `[FROZEN]`

```text
Qwen3-8B
Single + Single&Pers
two tracks trained separately
Catalog-Fine
official train/test split
max_action_steps = 30
main reward = strict
all 8 evaluation metrics
LoRA/PEFT for SFT + GRPO
seed = 1
```

## 19.2 第二类：第一版直接按论文规模，实际有问题再调 `[PAPER-DEFAULT / ADJUSTABLE]`

```text
GRPO group size G = 8
32 task groups / update
200 RL steps
32K max context
SFT 4 epochs
SFT effective batch = 32
KL off
```

原则：**不先给缩小版，不先按缩小版开发。**

## 19.3 第三类：论文没写清，本项目自行定义

`[PROJECT-FIXED]`

```text
6,000 successful SFT trajectories / scenario
~3,000 unique SFT tasks / scenario
target 2 successful demonstrations / task
stratified task sampling
coverage-first teacher collection
LoRA/PEFT
seed = 1
checkpoint/eval every 20 RL steps
fixed 128-task intermediate eval subset
full test final evaluation
SFT checkpoint once/epoch
```

`[PROJECT-CHOICE]`

```text
strong teacher exact model
teacher attempt cap
near-duplicate threshold
LoRA rank/target modules
training backend
```

---

# 20. 如果规模真的跑不动：调整原则

v1 不预先指定缩小后的数字。

只有 profiling 提供证据后才调整。

## 20.1 先做不改变实验语义的工程优化

例如：

- microbatch；
- gradient accumulation；
- gradient checkpointing；
- rollout batching；
- KV cache 管理；
- faster inference backend；
- 合理的 rollout pipeline。

## 20.2 Context 只在真实长度支持时缩

如果：

```text
p99 trajectory length << 32K
```

可降低 max context，必须记录 profiling evidence。

## 20.3 GRPO scale 若必须减少

优先保护：

```text
G = 8
```

如果仍必须改变：
- 先讨论 sample groups/update；
- 再讨论 total update/rollout budget；
- 本 v1 不预先给小规模替代值。

## 20.4 修改记录

任何第二类参数调整必须写入：

```text
experiments/deviations.md
```

至少包含：
- 论文值；
- 原计划值；
- 实际值；
- 修改原因；
- profiling evidence；
- 对可比性的潜在影响。

---

# 21. Compute / throughput profiling

## 21.1 Teacher profiling

估算：

```text
teacher success rate
average attempts per successful trajectory
API token usage
cost / 100 successful trajectories
```

最终目标是 12,000 successful trajectories，原始 API rollout 数取决于 teacher success rate。

## 21.2 RL profiling

测：

```text
trajectory/hour
tokens/sec
average actions/trajectory
average tokens/trajectory
GPU memory peak
environment latency
```

优先用接近论文形状的：

```text
32 task groups × 8 rollouts
```

做真实 throughput benchmark，然后估算 51,200 trajectories/run 和 4 mandatory runs 的 wall-clock。

---

# 22. 推荐代码组织

不做 generic harness，但保持模块清晰：

```text
shopsim-rl/
├── README.md
├── DESIGN.md
├── third_party/
│   └── ShopSimulator/
├── configs/
│   ├── data/
│   ├── sft/
│   ├── grpo/
│   └── eval/
├── src/
│   ├── data/
│   ├── env/
│   ├── agent/
│   ├── rollout/
│   ├── rewards/
│   ├── sft/
│   ├── rl/
│   ├── eval/
│   └── analysis/
├── data/
│   ├── manifests/
│   ├── teacher_raw/
│   ├── sft/
│   └── eval_subsets/
├── outputs/
│   ├── base/
│   ├── sft/
│   ├── grpo/
│   └── evaluation/
└── experiments/
    ├── configs/
    ├── logs/
    └── deviations.md
```

目录结构属于建议，Codex 可调整文件名，但不要改变模块职责。

---

# 23. Upstream ShopSimulator 集成原则

1. 固定 upstream commit hash；
2. 尽量不修改 third_party 原文件；
3. 必要 patch 单独记录；
4. ShopEnv 与 search index 使用官方 repo；
5. single-turn / persona action/observation semantics 以官方 single_eval 为依据；
6. training code 调用 environment，不重写搜索系统；
7. reward parity 优先对照官方 scorer。

---

# 24. Trajectory record

统一内部表示建议：

```python
TrajectoryRecord(
    task_id,
    scenario,
    policy_visible_context,
    messages,
    actions,
    observations,
    termination_reason,
    final_purchase,
    rewards,
    generation_metadata,
)
```

`termination_reason` 至少区分：

```text
purchase
max_steps
invalid/unrecoverable action
environment error
```

GRPO 另外保存 backend 所需 token/logprob rollout information。

---

# 25. Policy-visible vs evaluator-visible

必须代码层隔离：

```text
TaskRecord
├── policy_visible
│   ├── instruction
│   ├── persona (Pers only)
│   └── environment observations
│
└── evaluator_only
    ├── target product
    ├── target attributes
    ├── target options
    ├── target price constraint
    └── reward metadata
```

任何 prompt/formatter 只读取 `policy_visible`。

建议使用不同 dataclass，而不是依赖 Codex“记住不要用”。

---

# 26. Logging

每个 RL update 至少保存：

```text
step
scenario
init_type = base | sft
reward_type / alpha
num_groups
group_size
num_trajectories
mean_reward
mean_Rstrict
mean_Rloose
success_rate
finish_rate
Rcategory
Rattribute
Roption
Rprice
within_group_reward_std
zero_variance_group_ratio
mean_action_steps
mean_trajectory_tokens
rollout_wall_time
train_wall_time
GPU memory
```

Train rollout 与 held-out eval 结果必须分开。

---

# 27. Checkpoint 命名建议

```text
checkpoints/
  single/
    sft/
      epoch_01/
      ...
      epoch_04/
    direct_grpo_strict/
      step_020/
      ...
      step_200/
    sft_grpo_strict/
      step_020/
      ...
      step_200/

  single_persona/
    ...
```

Optional：

```text
direct_grpo_loose/
```

每个 checkpoint 保存 config snapshot 和 git commit。

---

# 28. Milestones

## M0 — Repository + Environment

完成 pinned ShopSimulator、Catalog-Fine setup、environment service、Single / Single&Pers baseline episode。

**DoD**：完整 episode 能跑完，reward 能返回。

## M1 — Data pipeline + profiling

完成 official split manifest、schema validation、leakage separation、3000-task stratified SFT manifests、fixed 128 eval subsets、dataset profiling。

**DoD**：counts 可复现，train/test 无 overlap。

## M2 — Reward parity + Base evaluation

完成 8 metrics、reward unit tests、Base full evaluation、trajectory/action/token profiling。

**DoD**：可生成第一张 Base Table 3-style row。

## M3 — Teacher SFT data

完成 strong teacher collection、coverage-first、success filtering、duplicate check、6000 successes/scenario、SFT JSONL。

**DoD**：12K successful trajectories + 完整 collection statistics。

## M4 — SFT

完成两 scenario SFT、epoch checkpoints、intermediate eval、full eval。

**DoD**：两条 SFT result rows。

## M5 — Direct GRPO Strict

完成两 scenario Base → GRPO strict、200-step target、checkpoint/eval cadence、full final eval。

**DoD**：Direct GRPO strict 两条 result rows。

## M6 — SFT + GRPO Strict

完成两 scenario SFT → GRPO strict、full final eval。

**DoD**：最小完整实验矩阵完成。

## M7 — Final reproduction report

完成 Table 3-style tables、training curves、Base/SFT/Direct RL/SFT+RL comparison、paper deviations、compute/data statistics、README 命令。

到此 v1 完整。

## M8 — Optional Paper Reproduction

若算力/时间允许：

```text
Direct GRPO + Rloose
```

优先级高于任何论文外小研究。

---

# 29. 项目最低成功标准

即使绝对 performance 不漂亮，只要满足：

1. Qwen3-8B Base 可在两个 scenario 稳定 rollout；
2. reward 8 metrics 可验证；
3. strong-teacher → success-filter → SFT dataset pipeline 完整；
4. 两 scenario SFT 独立完成；
5. Direct GRPO 是真实 online env rollout；
6. SFT+GRPO 从 scenario-specific SFT policy 继续 RL；
7. updated policy 确实进入下一轮 rollout；
8. held-out eval 与 train 严格分离；
9. Base/SFT/Direct RL/SFT+RL 有统一 Table 3-style comparison；
10. logs 足够解释训练发生了什么；
11. 所有偏离论文之处明确记录。

不要求达到论文数值，也不要求所有趋势都与论文一致。

---

# 30. Deviation from Paper：初始清单

### D1 — Teacher model
Paper：GPT-4.1  
Project：strong teacher，当前 provisional GPT-5.6 Sol。

### D2 — 6K SFT trajectories interpretation
Paper：6K successful trajectories，未说明 per-scenario / total。  
Project：6K per selected scenario。

### D3 — PEFT
Paper：只写 fine-tune Qwen3-8B，未披露 full FT / LoRA。  
Project：LoRA/PEFT。

### D4 — SFT data distribution
Paper：未披露 unique-task coverage / demos per task。  
Project：约 3000 unique tasks，target 2 successful demos/task，stratified，coverage-first。

### D5 — Teacher success filter
Paper：只写 successful trajectories。  
Project default：`Rsucc == 1`。

### D6 — SFT token masking
Paper：未披露。  
Project recommended default：assistant-only loss。

### D7 — GRPO backend
Paper：ROLL。  
Project：Hugging Face TRL；不安装 ROLL 或 veRL。

所有新增 deviation 必须进入 `experiments/deviations.md`。

---

# 31. 后续仍需单独冻结的参数

Teacher：
- exact model/version；
- generation temperature/top-p；
- max attempts/task；
- duplicate threshold。

LoRA：
- rank；
- alpha；
- dropout；
- target modules；
- adapter merge strategy。

SFT：
- exact learning rate；
- scheduler；
- warmup；
- max grad norm；
- packing；
- exact loss masking implementation。

GRPO：
- exact backend；
- optimizer；
- learning rate；
- clipping；
- generation temperature；
- rollout batching；
- logprob implementation；
- async/sync rollout。

Storage：
- full trajectory retention；
- compression/sharding；
- checkpoint retention beyond mandatory cadence。

---

# 32. Codex 实施约束

1. 不自动扩大 scope。
2. 不加入 Multi-Turn。
3. 不加入 Shopper LLM。
4. 不加入其他 base model。
5. 不因 `alpha` 接口存在而自动做 mixed reward。
6. 不修改 official train/test split。
7. 不让 policy 读取 evaluator-only gold。
8. 不把 offline response training 冒充 online GRPO。
9. 不把 Single 与 Single&Pers 混合训练。
10. 不在未经记录的情况下修改论文规模参数。
11. 优先复用 ShopSimulator environment/search/scoring。
12. 论文没写清时，先标 ambiguity，再采用本文 `[PROJECT-*]` 决策，不猜作者实现。
13. 每个正式 experiment 保存完整 config snapshot。
14. 每次正式实验记录 git commit、upstream commit、config、seed、dataset manifest hash、checkpoint 和 result path。
15. 第二类参数如需调整，先写 profiling evidence，再改 config。

---

# 33. Source-of-truth 优先级

实现冲突时：

1. 本文明确 `[FROZEN] / [PROJECT-FIXED]` 的项目决策；
2. 论文正文 / Appendix 明确披露的训练与 reward semantics；
3. 官方 ShopSimulator repo 的 environment/scoring behavior；
4. 本文 `[PROJECT-DEFAULT]`；
5. 开发便利性。

如果论文与 repo scoring 存在差异，不静默选择；记录并单独确认。

---

# 34. v1 一句话定义

> **在 ShopSimulator 的 Single-Turn 与 Single-Turn & Personalization 两个独立 scenario 上，以 Qwen3-8B 为统一 base policy，利用强 teacher 的成功 environment trajectories 完成 scenario-specific SFT cold-start，并分别从 Base 与 SFT policy 出发进行基于 Rstrict 的 online GRPO post-training，最后使用论文 Table 3 的 8 个指标在完整 held-out test set 上统一评估；第一版按论文 GRPO 规模设计，不预先缩水，所有必要工程调整通过 profiling 证据和 deviation log 管理。**

---

# 35. v1 核心配置摘要

```yaml
project:
  model: Qwen3-8B
  scenarios:
    - single
    - single_persona
  catalog: Catalog-Fine
  joint_training: false
  seed: 1

environment:
  max_action_steps: 30

sft:
  method: LoRA/PEFT
  successful_trajectories_per_scenario: 6000
  target_unique_tasks_per_scenario: 3000
  target_success_demos_per_task: 2
  sampling: stratified
  collection: coverage_first
  epochs: 4
  effective_batch_size: 32
  teacher: strong_api_teacher   # exact model TBD

reward:
  formula: alpha * strict + (1-alpha) * loose
  mandatory_alpha: 1.0
  optional_paper_alpha: 0.0

grpo:
  method: online_GRPO
  group_size: 8
  task_groups_per_update: 32
  updates: 200
  max_context: 32K
  kl_penalty: false
  parameter_update: LoRA/PEFT
  backend: Hugging Face TRL

evaluation:
  intermediate_subset_size: 128
  grpo_eval_every_steps: 20
  sft_eval_every_epoch: 1
  final_single_test: 1459
  final_single_persona_test: 1343
  metrics:
    - Rloose
    - Rstrict
    - Rsucc
    - Rfinish
    - Rcategory
    - Rattribute
    - Roption
    - Rprice

mandatory_experiments:
  - Base
  - SFT
  - Direct_GRPO_Strict
  - SFT_GRPO_Strict

optional_paper_experiments:
  - Direct_GRPO_Loose

not_v1:
  - MultiTurn
  - MultiTurn_Persona
  - other_model_scales
  - self_generated_SFT
  - mixed_reward_alpha
  - dynamic_alpha
  - new_RL_algorithms
  - generic_harness
```

---

# 36. 参考定位

**Paper**  
*ShopSimulator: Evaluating and Exploring RL-Driven LLM Agent for Shopping Assistants*

重点：
- §2.1 Task Formulation
- §2.2 Environment Construction
- Figure 2 Statistics
- §2.3 Rewarding
- §3.1 Evaluation Settings
- Table 2
- §4.1 Training Settings
- Table 3
- Figure 5
- Figure 6
- Appendix C.3 Reward Details
- Appendix D.1 Implementation of Training

**Repository**  
https://github.com/ShopAgent-Team/ShopSimulator

---

**End of v1 technical plan.**

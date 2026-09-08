# P3b Teacher Collection Policy

**状态**：`p3b-v1` 已冻结。本文只定义正式 collection policy；本轮没有调用 teacher
API、没有重跑 P3a、没有采集 formal trajectory，也没有实现 P3c collector。

机器可读 source of truth：
`configs/teacher/formal_collection_p3b_v1.yaml`；policy hash：
`b502cd79e2e95e63680126595bd701b3108909169c5fa40ed6572fca6797a59f`。

本文中的标签含义：

- **观察**：来自 canonical P3a 或 2/8/10-worker probe，不是 benchmark 结论。
- **已冻结**：属于 `p3b-v1`，变更必须创建新 policy version，不能静默修改或跨配置 resume。
- **延后但未遗忘**：当前证据不支持加入 v1，保留明确的重审触发条件。

## 1. P3a evidence

### 1.1 Teacher behavior

| scenario | first success | first-phase genuine attempts | burden / acquired first success | second demo | first-second exact duplicate |
|---|---:|---:|---:|---:|---:|
| Single | 19/24 | 34 | 1.789 | 38/38 | 6/38（15.79%） |
| Single&Pers | 20/24 | 35 | 1.750 | 36/40 | 4/36（11.11%） |

**观察**：Single 在第 1 次以后没有新增 first success；Persona 第 2 次多获得 1 个、
第 3 次再多获得 1 个 first success。失败 genuine attempt 的 mean wall time 分别约为
200.38s 与 53.04s，hard retry 有明显成本。成功 attempt 的 API time 占总 wall time约
97.88% / 97.94%，主要瓶颈是 teacher HTTP wait，而不是 environment step。

**已冻结**：Pass A 每个 primary/replacement task 最多 2 次 genuine first-success
attempt。这个 cap 接受少量潜在 coverage 损失，以避免对 hard task 做无界且昂贵的 retry。
Infrastructure interruption 不进入 genuine attempt denominator。

### 1.2 Diversity 与 telemetry

**观察**：P3a 的所有 within-task successful pairs 中，Single exact duplicate 为 7/57，
Persona 为 6/53；first-second 口径见上表。Provisional near duplicate 分别为 14/57 与
14/53，但 n=24/scenario，无法支撑稳定的 near-duplicate hard threshold。Thought 文本从未
参与 behavioral fingerprint。

现有 113 条 canonical successful trajectories 都保留了完整 policy-visible messages 与
逐步 action，113/113 可以按 pre-action observation 做 deterministic no-progress audit；
“连续 3 次相同 normalized action 且 observation 字节完全不变”命中 0 条。因此该规则可以
可靠实现为 narrow hard hygiene reject，不需要主观 quality classifier。

旧 P3a artifacts 只可靠保存 final response retry count，没有完整 retry-event history；历史
retry telemetry 明确标为 incomplete，不根据终端文字回填或猜测。当前 client 已能生成 redacted
retry events，P3c 必须逐 API step 持久化。

## 2. Concurrency decision

| workers | tasks/outcome | makespan | throughput | retries / 429 | isolation errors |
|---:|---|---:|---:|---:|---:|
| 2 | 2 success | 84.96s | 84.75 traj/h | 0 / 0 | 0 |
| 8 | 10 success | 203.94s | 176.52 traj/h | 0 / 0 | 0 |
| 10 | 9 success, 1 terminal unsuccessful | 279.18s | 116.05 traj/h | 15 / 15 | 0 |

**已冻结**：正式默认 `workers=8`；CLI 仍允许 `--workers N`（guard 1–32）。实际值必须写入
run manifest，并作为 immutable config；resume 不得跨 worker count。Scheduler 按 seed=1
确定性创建当前 pass 的 queue，API completion order 可非确定。Pass 必须严格 A → B → C，
不得跨 pass speculative execution。

一个 remote service 继续共享 Catalog-Fine、Lucene/search 与 SimServer，多 session 使用独立
slot；短 environment operation 串行，昂贵 API wait 并发。10-session isolation 已通过，10
worker 的退化来自 provider 429/latency，而不是 task/goal/session cross-talk。

**延后但未遗忘**：`p3b-v1` 不实现 provider-wide shared 429 throttle。8-worker probe 没有
429，先保留现有 per-request `Retry-After` + 2/5/10s backoff。若正式 8-worker run 出现持续
rate limiting，再以新 policy/runtime change 单独加入 shared throttle；不在 v1 预先引入复杂
provider scheduler。

## 3. Teacher and protocol configuration

**已冻结**：

```yaml
model: gpt-5.6-sol
api_style: responses
reasoning_effort: high
request_semantics: p3a-compatible
temperature: omitted
top_p: omitted
max_action_steps: 30
```

`p3a-compatible` 指 exact upstream Single/Persona prompt、Persona 的 `instruction_simple` +
upstream persona injection、当前 policy observation/action parser、每次请求完整 visible history，
以及不新增 sampling fields。Teacher 只输出 visible `Thought:`（简短 rationale）和 `Action:`；
不请求或保存 hidden CoT，不使用 diversity-conditioned prompt。

### Transport provenance 差异

Canonical P3a 与 2/8/10-worker probe 的 manifests 实际记录的是 `chat_completions`，不是
`responses`。附件明确要求 formal `p3b-v1` 冻结为 `responses`，因此 transport adapter 是
P3a → P3c 的有意变化，不能被描述成“P3a 原样 transport”。P3c 在正式 collection 前必须由
用户触发一个 tiny `responses` compatibility smoke，验证 visible text、usage、returned model、
prompt/history 与 action extraction；门禁未通过时不得开始正式 run。此 smoke 不进入 formal
dataset。

Returned model 必须与配置的 `gpt-5.6-sol` 一致；明显/精确 mismatch 按现有严格校验立即
global stop 并保存状态，不 blind retry。Run manifest 记录 sanitized request schema/config hash、
prompt hash 与 protocol versions，但绝不记录 API URL、API key 或 Authorization。

## 4. Formal targets and acceptance

**已冻结**：Single 与 Single&Pers 分开运行，各自 hard target 为 **6,000 accepted successful
trajectories**，合计 12,000。Primary manifest 各 3,000 task；约 3,000 unique successful
tasks/scenario 是 coverage target，不是 hard target。每 task accepted demo `min/target/max =
1/2/3`，绝不超过 3。

Accepted candidate 必须同时满足：

1. `Rsucc == 1`、`done == true`、存在 valid terminal purchase；
2. 无 environment fatal error、provider/infrastructure interruption 或 parser corruption；
3. 通过 narrow hygiene checks；
4. 不与同 task 已接受 trajectory 构成 exact behavioral duplicate。

每条 raw artifact 记录 `Rloose/Rstrict/Rsucc/Rfinish/Rcategory/Rattribute/Roption/Rprice` 全部
8 metrics。Gold/reward target 只能置于 `evaluator_only`，policy prompt 与未来 SFT formatter
禁止读取。

Quality 不要求 shortest path。Hard reject 只包括 exact behavioral duplicate、导致 integrity
不可信的 environment/parser/malformed/invalid action、evaluator/gold leakage，以及上述 exact
3-step no-progress loop。

**已冻结 diversity**：exact normalized behavioral action trajectory equality hard reject；该
attempt 仍消耗 teacher budget，但不计入 6,000。Provisional near duplicate 只做 diagnostic
flag，只要其余条件通过即可 accepted。禁止 embedding similarity、LLM-as-judge 和 Thought
text similarity。

## 5. Pass A — coverage

Primary 使用 P2 已冻结的 3,000-task manifests，不重采样：

| scenario | TRAIN pool | primary task_ids SHA-256 | reserve size |
|---|---:|---|---:|
| Single | 21,962 | `870c399fbb2034f58a0dc4b18b8563d4a8748a35d78a9e89de509db5ac2b969d` | 18,962 |
| Single&Pers | 3,323 | `f8ca0e5d3982661d19343df337a9713da16211ac43a361074a2c5d9c35fb6521` | 323 |

每个 primary task 最多 2 次 genuine first-success attempt。两次均失败则标记
`primary_unsolved`，从同 scenario TRAIN minus frozen primary 中，以 seed=1 选择尚未使用的
同 `(domain_zh, category)` reserve task；replacement 也最多 2 次，失败后继续同 stratum 下一个
unused reserve，直到 coverage slot success 或 same-stratum reserve exhausted。禁止跨 domain、
category 或 scenario fallback。

Pass A 在 3,000 unique success 或所有 coverage slots 已 success / same-stratum reserve exhausted
时结束。后者是正常 bounded terminal state，不是 crash；必须报告 reserve utilization、
`reserve_exhausted` 与 final unique coverage。Persona reserve 只有 323，达到 3,000 unique 很可能
偏紧，但不因此无限 retry 或放宽 success。

## 6. Pass B — natural second demonstration

对 Pass A 获得 first success 的每个 task，先做 exactly 1 次 independent second-demo attempt：
fresh reset，不看第一条 trajectory，不要求“换路线”。若 success + hygiene pass + not exact
duplicate，则该 task accepted count 变为 2。

若 B1 failure、exact duplicate 或 hygiene reject，只允许 1 次 recovery；所以 Pass B 最多
2 attempts。Failure/reject 均消耗 post-first-success budget，但不增加 accepted count。

## 7. Pass C — quota fill and distribution

Pass C 只使用已经获得 first success、accepted demos < 3 且尚有 post-first-success budget 的
tasks；不重新开启 coverage/reserve search。优先 accepted count=1，再考虑 count=2。

以 frozen primary 3,000 manifest 的 `(domain_zh, category)` proportions，通过 largest remainder
得到 6,000-demo stratum targets。优先 deficit strata；同 stratum 内 seed=1 deterministic task
order。若 stratum capacity 耗尽，记录 `stratum_shortfall`，允许把剩余 quota spill over 到其他
仍有 eligible capacity 的 strata，并完整记录。

达到 6,000 即 `complete`；bounded capacity 耗尽仍不足则 `quota_unmet`。不得自动增加 cap、
接收 exact duplicate、超过 3 demos/task 或修改 reserve rule。

## 8. Attempt budgets

| budget | frozen value |
|---|---:|
| Pass A first-success genuine attempts/task | 2 |
| Pass B second-demo attempts/task | 2（1 natural + 1 recovery） |
| Post-first-success attempts/task（B+C 合计） | 3 |
| Accepted demos/task | 1 / 2 / 3（min / target / max） |

B1 accepted 后还剩 2 次 post-success opportunity，但 accepted 总数仍最多 3；B1 reject + B2
accepted 后只剩 1 次；B1/B2 都失败也只剩 1 次。Infrastructure interruption 不消耗任何
genuine teacher attempt，resume 时 fresh reset。

## 9. Infrastructure, retry, stop and resume

Infrastructure classes：connection、timeout、HTTP 408/429/500/502/503/504、
`provider_protocol_error`。沿用初始请求后最多 3 次 retry、`Retry-After` 优先和 2/5/10s
backoff。这些不计 teacher failure 或 genuine attempt。

Retry exhausted 时 global graceful stop：停止调度新 work；其他 worker 不再发新 API request；
允许 in-flight response 返回并持久化；保存 interrupted attempts；best-effort release sessions；
flush SQLite/JSON/log 后退出。Resume 从 fresh environment reset 开始，partial 不消耗 quota。

HTTP 400/401/403/404/422、本地 config error 与 returned-model mismatch 不 blind retry，立即
global stop + preserve state，也不归类为 teacher failure。Workers、policy hash 或其他 immutable
run config 不匹配时拒绝 resume。

## 10. Logging and storage contract for P3c

P3c 必须复用 `data/teacher_raw/<scenario>/<run_id>/` 的 `run_manifest.json`、`state.sqlite`、
`attempts/`、`accepted/`，以及 atomic JSON、durable SQLite、Ctrl+C persistence 和 immutable
resume protection。

只新增两项轻量 debug UX：

1. 总体 append-only `data/teacher_raw/collector.log`，记录 run/pass/task/attempt/step context 与
   start/resume/retry/accept/reject/reserve/quota/stop/complete 等事件；不记录 URL、key、
   Authorization 或 provider raw body。
2. `scripts/inspect_teacher_run.py --scenario <...> --latest` 与 `--last N`，输出当前 pass、
   coverage/accepted/attempt/retry/token/elapsed/last event、精确 resume command 及近期 log。

Attempt JSON + SQLite 仍是 source of truth；不建设 dashboard、tracing system、event pipeline 或
monitoring server。每个 API step 保存 redacted retry events（timestamp、step、reason、status、
retry ordinal、backoff），由此修复 P3a historical retry telemetry 不完整的问题。

Accepted 与 rejected attempt artifacts 至少保留 task/scenario/run/pass/attempt identity、sanitized
teacher config、messages、visible responses、actions、policy-visible observations、terminal
purchase、8 rewards、timing、tokens、request IDs（若 relay 提供）、retry events、behavior
fingerprint、duplicate flags、acceptance 与 rejection reason。Provider raw body 禁止落盘；分析
需要的 gold 字段只能进入 `evaluator_only`。

## 11. Runtime projection and limitations

**观察**：P3a cost-aware serial baseline 约为 Single 169.16h / 6,000、Persona 117.71h /
6,000；理想线性除以 8 只是约 21.1h / 14.7h 的数学下界。另一方面，8-worker easy-task probe
176.52 traj/h 对应约 34h / 6,000。两组证据只支持“每 scenario 大致为数十小时量级”，不是
SLA，也不是正式 P3c runtime prediction。

Hard tasks、duplicate/hygiene rejection、reserve、provider variance、rate limiting 与 outage 都会
改变实际时间。P3a 每 scenario 只有 24 task；并发 probe 只有 10 easy Single tasks，不能外推为
teacher benchmark 或 guaranteed throughput。Public snapshot 的 Persona TRAIN 是 3,323 而不是
论文的 3,383；缺少 query 时继续固定 `query_match=False` deviation。

## 12. P3c implementation checklist

下一阶段只按本 policy 实现：

1. formal A/B/C scheduler 与 deterministic queue/reserve；
2. default 8 workers，worker count / policy version+hash immutable；
3. success/hygiene/exact-duplicate acceptance 与 max 3 demos/task；
4. post-success budget、quota/distribution/spillover accounting；
5. durable attempts/accepted ledger、global stop、Ctrl+C/resume；
6. append-only `collector.log` 与 `inspect_teacher_run.py --latest/--last N`；
7. sanitized request hash、prompt/protocol/returned-model validation；
8. no-paid unit/integration tests；
9. 用户触发 tiny `responses` compatibility/formal smoke；
10. smoke 通过后才允许正式 collection。

P3c 不得静默改变本文件任一已冻结值；规则变更必须产生新的 policy version/hash。

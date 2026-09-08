# P3 Teacher Collection（P3-0 基础设施）

**状态**：P3-0、canonical P3a、concurrency extension 与 P3b policy freeze 已完成；下一步是
P3c formal collector implementation。正式冻结规则以 `P3B_COLLECTION_POLICY.md` 与
`configs/teacher/formal_collection_p3b_v1.yaml` 为准。

## 1. 架构

Teacher controller 在用户本地运行，ShopSimulator 只在 `rtx-pro-6000-3:/root/shopping-agent-rl` 的 CPU 环境运行：

```text
local controller → teacher relay API
local controller → SSH local port forward → remote ShopSimulator HTTP service
```

ShopSimulator、Lucene、Pyserini、Java、spaCy、Catalog runtime 不在本地复制。Remote service 绑定 `127.0.0.1`，默认端口为 `5100`；本地 tunnel 默认端口为 `5500`。当前 remote environment version 为 `task-scoped-v3-multisession`：一个 service process 共享完整 Catalog-Fine/Lucene runtime，通过 external session UUID → internal slot 绑定多个轻量 `WebAgentTextEnv` session；reset/step/release 在 global `RLock` 下串行执行，昂贵的 teacher HTTP waits 留在 local worker threads 中并行。task-scoped reset 只 materialize 当前 TRAIN task 的 goal，但 search 仍使用完整 23,421-document Catalog-Fine Lucene index，不使用 gold lookup 或小 catalog 替代正式 search universe。

## 2. Teacher relay 配置

复制 `.env.teacher.example` 为 `.env.teacher`，仅在本地填写 `TEACHER_API_URL`、`TEACHER_API_KEY`、`TEACHER_API_MODEL` 和 `TEACHER_API_STYLE`。支持 `chat_completions` 与 `responses` 两种 OpenAI-compatible adapter；relay model identifier 不写死在代码中。`.env.teacher` 被 Git 忽略，key 不进入异常、日志、manifest 或测试 fixture。

默认 runtime engineering 参数：connect timeout 15 秒、read timeout 300 秒、初始请求之外最多 3 次 retry。408/429/500/502/503/504、连接错误和 timeout 使用 2/5/10 秒级 backoff（含少量 jitter），优先遵守 `Retry-After`。400/401/403/404/422 属于 request/config failure，不盲目重试。正常 HTTP 响应但 malformed action 或最终失败属于 teacher/model failure，才计 teacher attempt。

HTTP 200 但 invalid JSON、缺少 expected protocol structure 或没有可解析 visible text 的
provider response 统一归类为 `provider_protocol_error`：按 transient policy retry，耗尽后
停止整个 profiler、保存状态且不消耗 teacher attempt。协议正常但 visible output 为空则是
`teacher_empty_response`，属于 teacher/model failure。provider response body 不写入异常、日志
或 artifacts。

retry 耗尽时，collector 必须 flush 当前状态、标记 `infrastructure_interrupted` 并停止整个 collection；不消耗 task attempt quota。终端提示已保留数据和 `--resume` 命令。

## 3. Durable storage / resume

正式 canonical working copy 为本地 `data/teacher_raw/<scenario>/<run_id>/`：

```text
run_manifest.json  # immutable run identity/config
state.sqlite       # attempts 与 accepted ledger
accepted/          # 仅成功且 accepted 的 trajectory
attempts/          # 所有 attempt，包括失败和 partial
logs/
```

accepted JSON 先写入临时文件并 flush/fsync，再用原子 rename 发布，之后才在 SQLite transaction 中标记 accepted。partial、invalid-action、max-steps 和 infrastructure interruption 仅进入 attempts，永远不能直接进入 SFT。`--resume` 保留旧 artifact，不覆盖 accepted 数据；immutable config（scenario、teacher model、API style、reasoning、prompt/config hash、ShopSimulator fingerprint、environment/reward version）不一致时拒绝 resume。SIGINT 视为 graceful stop：保存当前 partial attempt、提交 SQLite、flush 日志并打印 resume 命令；不尝试恢复远程环境的内存 step，而是下次 reset 后开启新 attempt。

`TeacherLedger` 的 SQLite connection 使用 WAL、`check_same_thread=False` 和实例级 `RLock`，每个短 transaction/JSON publish 都在同一边界内完成；并发 scheduler 必须在最终 close 前 join workers。

## 4. Environment smoke

```bash
bash scripts/teacher_env_up.sh
python scripts/test_teacher_env.py
bash scripts/teacher_env_down.sh
```

脚本只管理自己记录的 tunnel/service PID，不使用 broad `pgrep`/kill。每次 `teacher_env_up.sh`
都会同时检查远程 `/health` 与本地 tunnel；PID 尚存但 forwarding channel stale，或远程
service PID 尚存但 `/health` 不通时，只回收匹配本项目命令行的 PID 并重建。运行中若网络
瞬断，profiler 仍按 infrastructure-interruption 规则停止并保存状态；恢复前重新执行
`teacher_env_up.sh`，然后用明确的 `--run-id ... --resume` 继续。`test_teacher_env.py`
不调用 teacher API，而是对 Single 与 Single&Pers 的固定 TRAIN task 走真实 `reset →
Thought/Action search → Lucene observation → click product → click option/SKU → Buy Now`，并
检查 terminal reward 与 `query_match=False`。这只是 infrastructure smoke，不产生 teacher
trajectory 或评测结果。

Remote provenance：ShopSimulator public snapshot 不包含 `shop_env/search_engine`；index 使用 pinned Princeton WebShop commit `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd` 的兼容 converter/indexing source。Catalog-Fine index build 命令和 source fingerprint 见 [P2_ENVIRONMENT_SETUP.md](P2_ENVIRONMENT_SETUP.md)。当前 remote spaCy 模型为 `core_web_sm 3.8.0`。

## 5. Frozen collection policy（P3b v1；本页只摘要）

每个 scenario hard target 为 6,000 条 accepted successful trajectories；约 3,000 unique 是
coverage target，接受范围为每 task 1–3 条、target 2。Default workers=8。Pass A first-success
cap=2 并使用 seed=1 deterministic same `(domain_zh, category)` reserve；Pass B 最多 2 次；
B+C post-first-success budget 合计 3 次；Pass C 做 proportional quota fill。Exact behavioral
duplicate hard reject，provisional near duplicate diagnostic-only。完整 attempt、acceptance、
infrastructure 与 logging contract 见 `P3B_COLLECTION_POLICY.md`。

未来 diversity fingerprint 至少包含 normalized action sequence、normalized search queries、clicked product IDs、selected options 和 trajectory length；Thought 措辞差异不视为策略差异。Teacher 只使用 visible `Thought: 简短 action rationale` 与 `Action:` protocol，不获取或保存 hidden chain-of-thought。Prompt 优先复用 upstream Single/Persona system prompt 与完整 visible conversation history；Persona 额外注入 `user_persona`，不添加改变 policy distribution 的“生成 SFT 数据”指令。

## 6. P3a metric definitions

Canonical summary 不改变任何 trajectory，只修正统计口径：

- `solved_tasks_attempts_to_success` 只对最终 solved task 统计首次 success ordinal；`first_phase_attempt_burden_per_acquired_success` 为所有 genuine first-success attempts 除以 acquired first successes，未把 unsolved task 的 retry cost 隐藏在 `attempts/success` 名称下。
- Diversity denominator 是同 task 内所有 successful trajectory pairs，且单列 first-second 与 second-second；Thought 文本不参与 behavioral duplicate。
- `api_call_latency` 是单次 API call，`api_total_time_per_attempt` 是一条 trajectory attempt 的 API wall time，`trajectory_attempt_wall_time` 是整条 attempt wall time。
- `success_only_lower_bound_*` 只使用成功 attempt 的 wall time；`observed_p3a_cost_aware_*` 使用 canonical P3a 所有 genuine attempts 的 wall cost / acquired successful trajectory。两者都是 rough baseline，不是 formal P3c prediction。
- `task_outcomes.profile_unsolved` 与 `attempt_outcomes.max_steps/terminal_unsuccessful/malformed_action/...` 分层报告。旧 artifacts 缺失完整 retry event 时报告 telemetry incomplete，不人工补猜；新请求保存 redacted retry event metadata。

## 7. Concurrency probe architecture

`src/rollout/concurrency.py` 提供 bounded `ThreadPoolExecutor`、`--workers N`（当前安全上限
32）和 global stop primitive。任一 worker 遇到 exhausted infrastructure/provider protocol
failure 后，不再让 queued task 发起新外部请求；in-flight request 可返回并落盘。每个 worker
使用独立 `TeacherClient`/HTTP client；probe artifacts 写到
`data/teacher_concurrency_probe/`，不进入 `teacher_profile`、SFT 或 canonical summary。

这是 concurrency probe infrastructure，不是正式 collection scheduler。2/8/10-worker probe
支持 P3b 冻结 formal default `N=8`；实际 worker count 仍可配置，但必须写入 immutable run
manifest。

远端 non-paid isolation stress 已验证 10 个混合 Single/Persona session：slot/session 唯一，
正常 search/product/option/Buy Now 路径互不串 task/goal，释放一个 session 不影响其余 session，
最终 active session 为 0；10-session 期间 process RSS/cgroup memory 未出现增长。该结果只证明
环境隔离与 runtime 复用，不是 teacher success-rate 或 throughput 结论。

## 8. Stages

```text
P3-0  API & collection infrastructure smoke       ← complete
P3a   Teacher profiling（canonical single worker） ← complete
P3a-C Concurrency extension/probe                 ← complete
P3b   Collection policy freeze                    ← complete
P3c   Formal collector implementation             ← next
P3d   Dataset freeze
```

P3a 的固定 profiling plan 为每个 scenario 24 条 TRAIN task（seed=1，来自冻结 primary SFT
manifest），first-success 每 task 最多 3 次尝试，成功后最多 2 次 second-demo exploration，
`max_action_steps=30`，单 worker。这些只是观察 teacher 行为的 operational limits，不是
P3c formal collection caps；P3b 才根据结果冻结 attempt cap、reserve 与 diversity threshold。
P3a 使用 `data/teacher_profile/<scenario>/<run_id>/` 的 `trajectories/`，与正式的
`data/teacher_raw/` 物理隔离；profiling artifacts 永远不会直接进入 SFT。

Prompt/rollout 使用 pinned `single_eval/configs/{standard,persona}/qwen3_235b.yaml` 的
scenario system prompt/source/hash、当前 persona 和 canonical `policy_observation`；每次 API
请求发送完整 visible history，teacher 只返回可见 `Thought:`/`Action:` protocol。remote
service 同时返回 raw observation 与 pre/post action diagnostics，但 collector 不把它们直接喂给
teacher。第二次 rollout fresh reset 且不提供第一条 trajectory，不做 diversity-conditioned
prompting，不保存 hidden chain-of-thought。

本轮明确不执行真实 teacher request、批量 trajectory、SFT formatter/training、GRPO、GPU 或模型 inference；
用户完成两个 scenario 后，再用 `scripts/summarize_teacher_profile.py` 生成真实报告供 P3b 分析。

## 9. Backup

`bash scripts/sync_teacher_data.sh [--dry-run]` 使用 rsync 从本地 `data/teacher_raw/` 增量同步至 `rtx-pro-6000-3:/root/data/shopsim/teacher_raw/`，不使用 `--delete`，不传输 `.env.teacher`，失败不会修改本地 canonical data。

## 10. P3a user-triggered profiling

完成本地 `.env.teacher` 配置并确认 relay 后，由用户手动执行：

```bash
bash scripts/teacher_env_up.sh
python3 scripts/profile_teacher.py --scenario single
python3 scripts/profile_teacher.py --scenario single_persona
python3 scripts/summarize_teacher_profile.py
bash scripts/teacher_env_down.sh
```

如果 infrastructure interruption，已完成的 profiling artifact 会保留；检查 API 后仅
对相应 scenario 使用 `--resume`。命令会按当前 API model/style/reasoning 与冻结 task
列表筛选兼容 run，并自动选择 `created_at` 最新的未完成 run；终端会打印所选 run 的
状态、已触及 task 数、terminal task 数、attempt 数和成功数。已完成 run 不参与选择。
被标记为 `invalidated_by_implementation_bug` 的旧 run 永远不会被 resume；修复协议后必须
创建新的 run_id。
如需恢复更早的兼容 run，可用 `--run-id` 显式覆盖自动选择。例如：

```bash
python3 scripts/profile_teacher.py \
  --scenario single \
  --run-id p3a-20260907T124356Z-920dc25e \
  --resume
```

`--run-id` 必须来自同一 scenario 且配置兼容的 run；不会删除、覆盖或合并其他 run。
如果没有兼容的未完成 run，命令会拒绝 resume 并要求恢复原配置或新建 run。P3a 不自动
调用另一个 API smoke、不做 concurrent workers，也不开始 P3b。

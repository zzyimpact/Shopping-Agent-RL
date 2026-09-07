# P3 Teacher Collection（P3-0 基础设施）

**状态**：P3-0 完成；P3a profiler 已实现并等待用户手动执行真实 profiling。本文不表示
P3a 结果或正式 collection 已完成。

## 1. 架构

Teacher controller 在用户本地运行，ShopSimulator 只在 `rtx-pro-6000-3:/root/shopping-agent-rl` 的 CPU 环境运行：

```text
local controller → teacher relay API
local controller → SSH local port forward → remote ShopSimulator HTTP service
```

ShopSimulator、Lucene、Pyserini、Java、spaCy、Catalog runtime 不在本地复制。Remote service 绑定 `127.0.0.1`，默认端口为 `5100`；本地 tunnel 默认端口为 `5500`。task-scoped reset 只 materialize 当前 TRAIN task 的 goal，但 search 仍使用完整 23,421-document Catalog-Fine Lucene index，不使用 gold lookup 或小 catalog 替代正式 search universe。

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

## 4. Environment smoke

```bash
bash scripts/teacher_env_up.sh
python scripts/test_teacher_env.py
bash scripts/teacher_env_down.sh
```

脚本只管理自己记录的 tunnel/service PID，不使用 broad `pgrep`/kill。`test_teacher_env.py` 不调用 teacher API，而是对 Single 与 Single&Pers 的固定 TRAIN task 走真实 `reset → Thought/Action search → Lucene observation → click product → click option/SKU → Buy Now`，并检查 terminal reward 与 `query_match=False`。这只是 infrastructure smoke，不产生 teacher trajectory 或评测结果。

Remote provenance：ShopSimulator public snapshot 不包含 `shop_env/search_engine`；index 使用 pinned Princeton WebShop commit `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd` 的兼容 converter/indexing source。Catalog-Fine index build 命令和 source fingerprint 见 [P2_ENVIRONMENT_SETUP.md](P2_ENVIRONMENT_SETUP.md)。当前 remote spaCy 模型为 `core_web_sm 3.8.0`。

## 5. Approved collection policy（本轮只记录，不执行）

每个 scenario 的目标是 6,000 条 accepted successful trajectories，约覆盖 3,000 个 unique tasks；2 demos/task 是默认 target 而非 hard constraint，接受范围为每 task 1–3 条。优先级为 `Success > behavior quality > task coverage > diversity`，采用 coverage-first：Pass A 获取第一条 success，Pass B 尝试第二条自然不同 success，Pass C 在最多 3 demos/task 内做 quota redistribution。独立 stochastic sampling 后做基于行为的 post-hoc diversity filtering；不通过 prompt 强造绕路，简单 task 可标记 `diversity_saturated` 只保留一条。困难 task 使用 bounded retry；第一条始终失败时，从相同 stratum 使用 deterministic reserve replacement；第二条失败或重复时保留第一条并由其他 task 补 quota。具体 attempt cap 和 similarity threshold 留到 P3a profiling 后冻结。

未来 diversity fingerprint 至少包含 normalized action sequence、normalized search queries、clicked product IDs、selected options 和 trajectory length；Thought 措辞差异不视为策略差异。Teacher 只使用 visible `Thought: 简短 action rationale` 与 `Action:` protocol，不获取或保存 hidden chain-of-thought。Prompt 优先复用 upstream Single/Persona system prompt 与完整 visible conversation history；Persona 额外注入 `user_persona`，不添加改变 policy distribution 的“生成 SFT 数据”指令。

## 6. Stages

```text
P3-0  API & collection infrastructure smoke       ← 本轮
P3a   Teacher profiling（小规模、单 worker）
P3b   Collection policy freeze
P3c   Formal collection
P3d   Dataset freeze
```

P3a 的固定 profiling plan 为每个 scenario 24 条 TRAIN task（seed=1，来自冻结 primary SFT
manifest），first-success 每 task 最多 3 次尝试，成功后最多 2 次 second-demo exploration，
`max_action_steps=30`，单 worker。这些只是观察 teacher 行为的 operational limits，不是
P3c formal collection caps；P3b 才根据结果冻结 attempt cap、reserve 与 diversity threshold。
P3a 使用 `data/teacher_profile/<scenario>/<run_id>/` 的 `trajectories/`，与正式的
`data/teacher_raw/` 物理隔离；profiling artifacts 永远不会直接进入 SFT。

Prompt/rollout 使用 remote `web_agent_text_env.py` 返回的 exact upstream prompt/source/hash、
当前 persona 和 observation；每次 API 请求发送完整 visible history，teacher 只返回可见
`Thought:`/`Action:` protocol。第二次 rollout fresh reset 且不提供第一条 trajectory，不做
diversity-conditioned prompting，不保存 hidden chain-of-thought。

本轮明确不执行真实 teacher request、批量 trajectory、SFT formatter/training、GRPO、GPU 或模型 inference；
用户完成两个 scenario 后，再用 `scripts/summarize_teacher_profile.py` 生成真实报告供 P3b 分析。

## 7. Backup

`bash scripts/sync_teacher_data.sh [--dry-run]` 使用 rsync 从本地 `data/teacher_raw/` 增量同步至 `rtx-pro-6000-3:/root/data/shopsim/teacher_raw/`，不使用 `--delete`，不传输 `.env.teacher`，失败不会修改本地 canonical data。

## 8. P3a user-triggered profiling

完成本地 `.env.teacher` 配置并确认 relay 后，由用户手动执行：

```bash
bash scripts/teacher_env_up.sh
python3 scripts/profile_teacher.py --scenario single
python3 scripts/profile_teacher.py --scenario single_persona
python3 scripts/summarize_teacher_profile.py
bash scripts/teacher_env_down.sh
```

如果 infrastructure interruption，已完成的 profiling artifact 会保留；检查 API 后仅
对相应 scenario 使用 `python3 scripts/profile_teacher.py --scenario <scenario> --resume`。
P3a 不自动调用另一个 API smoke、不做 concurrent workers，也不开始 P3b。

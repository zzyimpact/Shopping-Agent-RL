# P3c Formal Collector

P3c implements the formal teacher collector against frozen policy
`p3b-v1.1` (`a177c99f190f1f04ca20f079e6a4b42a9fa77cb741194db517ee494157ffca3c`).
The only P3b policy transport delta is `chat_completions`; the v1.1 delta is
recorded in `docs/P3B_COLLECTION_POLICY_V1_1.md`.

## Architecture

`scripts/collect_teacher.py` performs all no-paid preflight checks before it
constructs a `TeacherClient`. The main scheduler owns deterministic work
selection and serialized acceptance/state updates. A long-lived ThreadPoolExecutor with completion-driven refill
runs independent rollouts with separate remote sessions; the existing SQLite
WAL ledger and atomic JSON writer provide durable attempts and accepted
artifacts. The legacy default worker count is 8; worker capacity is now a
runtime throughput parameter and may change on resume (see runtime update below).
Relay URL, key, and API style are runtime transport settings and may change
between resume sessions; model `gpt-5.6-sol`, reasoning `high`, and all visible
prompt/protocol semantics remain fixed. Each new attempt records the API style
actually used for provenance.

## Passes

- Pass A gives each frozen primary coverage slot up to two genuine attempts.
  An unsolved task is replaced from the deterministic unused reserve queue for
  the same `(domain_zh, category)`; allocation is persisted before dispatch.
- Pass B starts only after A ends. Every first-success task gets B1 and, after
  failure, hygiene rejection, or an exact duplicate, at most one B2 recovery.
- Pass C starts only after B ends. It prioritizes tasks with one accepted demo,
  then two, while following largest-remainder stratum deficits. B+C share the
  three-attempt post-success budget and accepted demos never exceed three.
  Dispatch is capped by remaining accepted quota, so collection stops exactly
  at 6,000; exhausted capacity produces `quota_unmet` without policy relaxation.

No task can have two attempts in flight. Acceptance checks infrastructure
integrity, terminal success and all eight rewards, narrow no-progress hygiene,
then the existing behavioral fingerprint. Exact same-task duplicates are
rejected; provisional near duplicates are diagnostic only.

## Durability and resume

Runs live under `data/teacher_raw/<scenario>/<run_id>/`. The ledger and attempt
artifacts reconstruct the pass, coverage slots, attempt budgets, reserve
allocations, accepted counts, and deficits after a crash. `--resume` selects
the sole compatible incomplete run; if several exist, use the reported
`--run-id`. Ctrl+C stops new dispatch, lets in-flight work persist, releases
sessions, closes SQLite, and prints the exact resume command.

## Logging and inspection

The only aggregate log is append-only `data/teacher_raw/collector.log`. It
contains sanitized lifecycle events and no API URL, key, headers, or provider
body. Inspect a run with:

```bash
python3 scripts/inspect_teacher_run.py --scenario single --latest
python3 scripts/inspect_teacher_run.py --scenario single --latest --last 50
```

## Tiny formal pipeline smoke

The user-run paid smoke uses the real formal engine, two known-short TRAIN
tasks, and two workers. It is bounded to one Pass-A attempt per task and writes
only to `data/teacher_formal_smoke/`:

```bash
bash scripts/teacher_env_up.sh
python3 scripts/test_formal_collection.py
bash scripts/teacher_env_down.sh
```

The smoke does not validate transport compatibility and never enters formal,
P3a, or SFT data. Run it before starting a 6,000-trajectory scenario.

## Formal startup (after smoke acceptance)

```bash
python3 scripts/collect_teacher.py --scenario single
python3 scripts/collect_teacher.py --scenario single_persona
python3 scripts/collect_teacher.py --scenario single --resume
```

## Rolling scheduler correction (2026-09-09)

此前审计 HEAD `304511e7aa9fabe7841be4d769632342ef8b29af` 使用 fixed <=8
micro-batches、submission-order future consumption 和整批 refill barrier。
现在使用 `completion-driven-rolling-v1`：executor 在 engine 生命周期内复用，
`FIRST_COMPLETED → main-thread reconcile → recompute eligibility → refill`。
最多 configured workers 条在途，不预提交整个 Pass；默认仍为 8。
A/B/C 只在当前 Pass 无 eligible work **且无在途 future** 时转换。

每次 dispatch 前保存 reserve allocation、active task IDs 和可选 `in_flight_work`
reservation（task/pass/ordinal）。worker 清理结束、future 返回后才移除 reservation；
main-thread acceptance 不处理仍在清理的 candidate。Pass C 每条在途 attempt 都预留
一个潜在成功：`accepted + in_flight <= target`。失败/duplicate 才释放相应 quota，
不靠事后丢弃成功修复 overshoot。所有预算、same-stratum reserve、duplicate/hygiene、
prompt/reward 和 `p3b-v1.1` policy/hash 保持原值。Completion timing 不保证旧/新
scheduler 的未来样本逐条一致，但 reserve 仍按当前 durable state 确定性分配。

Implementation commit: `d9fade8320a8632e7e9b51a32f4a3c28db302dd2` (`fix: use rolling teacher collection scheduler`).
本轮定向 no-paid regression：42 passed（formal collector、concurrency、storage、runtime、collection policy）。

### Existing-run boundary / provenance

两个原 run 保留；本次静态检查时都是 stopped / Pass A / workers=8，无 active task：

| Scenario | Run ID | Accepted / unique successes | Genuine attempts | All attempt rows |
| --- | --- | ---: | ---: | ---: |
| Single | formal-20260908T105913Z-4eec9efd | 583 | 830 | 1016 |
| Single&Persona | formal-20260909T035508Z-17af7151 | 309 | 523 | 560 |

原始 manifest/attempts/accepted 不改写、不作废、不重新采集。Single 最后 durable
update 为 `2026-09-09T03:45:04.569769+00:00`；Persona 为
`2026-09-09T10:42:56.294026+00:00`。原 manifest creation commits 分别为
`e4285a4e714c690ad08b42d31e2f296bf19664fb` 和
`131e10e429030e21414b201243f8efecce073b3a`，它们不是所有后续采集的实现版本。

用户下次手动 resume 时，在既有 `formal_state.scheduler_history` 追加 scheduler
version、**实际执行的 git commit**、UTC resume 时间、policy、workers、当前 Pass、
既有 attempts 的最大 rowid、accepted/genuine counts；同一边界写入现有 collector.log
的 `SCHEDULER_START`。历史 manifest 不修改，新增 optional state 字段无需 migration，
旧 state 可直接恢复。未来审计按 rowid boundary 划分执行区间，不能声称整个 run
来自一个 implementation commit。本次尚未执行真实 resume，故没有伪造切换时间。

静态 gate 对两个真实 manifest/SQLite 的临时副本执行 resume 配置路径，remote 用
已保存 prompt/hash 与 fingerprint 的 stub，TeacherClient 被禁止构造，engine 不执行。
两种 API style 均通过；实时 tunnel/environment 可用性仍由用户启动时 preflight 检查。
Relay URL/key/style hotfix 保留；没有 automatic resume，infrastructure stop 后人工恢复。

正常输出应为初始最多 8 条 start，随后某条 finished 后尽快出现新的 start；
Pass 边界、无 eligible work、quota reservations 已满或 global stop 时不补位。
Rolling 会增加实际平均 API 并发和请求速率；观察 429/retry、provider latency、
terminal_unsuccessful 和 occupancy。若同时启动两个 scenario，每个上限 8，合计可能
达到 16 条在途；本次没有新增跨进程 throttle，也没有变更 workers policy。

## Runtime update: multiple API profiles

Early formal collection used fixed workers=8. The user subsequently reclassified
worker count and per-API-profile concurrency as runtime throughput parameters.
The p3b-v1.1 YAML/hash and earlier immutable-workers notes remain historical
records; formal resume no longer compares workers. All dataset identity,
attempt budgets, acceptance and Pass A/B/C rules remain unchanged. Existing
accepted data stay valid; resume the original run IDs without migration.

`--api-workers 1:8,2:12` uses numbered `.env.teacher` entries
`TEACHER_API_URL_1`, `TEACHER_API_KEY_1`, etc. `TEACHER_API_STYLE_1` is optional
when the shared `TEACHER_API_STYLE` is present; an explicit numbered style wins.
Model/reasoning remain shared `TEACHER_API_MODEL` / `TEACHER_REASONING_EFFORT`.
Profile IDs are positive integers. `--api-workers` and `--workers` are mutually
exclusive; without the former, unnumbered credentials and `--workers N` still
work. The existing engineering guard remains 1–32 total workers per collector.

One scenario still has one collector, ledger and central rolling scheduler.
Each profile has a fixed capacity counter; a future releases only its own
profile slot. One independent TeacherClient is bound to the entire trajectory.
No per-step switching, provider failover, adaptive routing or cross-process
key registry is introduced. The user assigns a key to only one scenario.
Any existing fatal infrastructure stop still stops the whole collector.

Start/finished lines show `[single][api=1]`, etc. This mapping is in memory only:
raw/accepted trajectory schemas and collector.log event layout are unchanged.
Crash-recovered historical completions may have no profile label. The existing
scheduler_history records only profile IDs/capacities (no keys/URLs), and inspect
and the final resume command retain the current runtime capacity arguments.
The initial run manifest remains historical and is not rewritten.

Configure credentials before running. Start the shared environment once with
`bash scripts/teacher_env_up.sh` if needed; do not launch two collectors for the
same scenario. Gracefully stop an existing scenario process before resuming it
with new capacity. Example allocations (not optimal-concurrency decisions):

```bash
python3 scripts/collect_teacher.py --scenario single --api-workers 1:8,2:12 --run-id formal-20260908T105913Z-4eec9efd --resume
python3 scripts/collect_teacher.py --scenario single_persona --api-workers 3:6 --run-id formal-20260909T035508Z-17af7151 --resume
```

This extension received static diff/syntax checks only, per user request;
no tests, fake runs, environment startup or API calls were executed.

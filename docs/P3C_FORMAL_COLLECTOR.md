# P3c Formal Collector

P3c implements the formal teacher collector against frozen policy
`p3b-v1.1` (`a177c99f190f1f04ca20f079e6a4b42a9fa77cb741194db517ee494157ffca3c`).
The only P3b policy transport delta is `chat_completions`; the v1.1 delta is
recorded in `docs/P3B_COLLECTION_POLICY_V1_1.md`.

## Architecture

`scripts/collect_teacher.py` performs all no-paid preflight checks before it
constructs a `TeacherClient`. The main scheduler owns deterministic work
selection and serialized acceptance/state updates. Existing bounded workers
run independent rollouts with separate remote sessions; the existing SQLite
WAL ledger and atomic JSON writer provide durable attempts and accepted
artifacts. The default worker count is 8 and is immutable on resume.

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

No task can appear twice in an active batch. Acceptance checks infrastructure
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

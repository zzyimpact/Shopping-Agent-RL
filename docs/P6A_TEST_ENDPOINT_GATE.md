# P6A TEST endpoint gate

## Outcome and resource evidence

TEST_ENDPOINT_GATE: PASS. BASE_128_COMMAND_READY: YES means user-controlled A -> B -> C,
not that a formal evaluation has been started. Starting HEAD: b93675e.

The previous FAIL was correct **in no-GPU mode**: cgroup 0.5 CPU / 2 GiB, with roughly
1.7 GiB ShopEnv RSS. It does not apply to GPU mode. On 2026-09-11, rtx-4 reported:

- cgroup CPU `2200000 100000`: **22 CPU**, despite `nproc=208` host visibility.
- cgroup memory.max `118111600640`: **110 GiB**; current `438378496` (~418 MiB),
  remaining quota ~109.59 GiB before smoke. Host `free -h` (1 TiB) is not the quota.
- RTX PRO 6000 Blackwell, 97887 MiB VRAM, driver 580.95.05; no model loaded this round.
- SECOND_EVAL_SERVICE_SAFE: YES. Real TEST runtime after two resets: RSS
  `1398198272` (~1.30 GiB), cgroup current `1469739008` (~1.37 GiB).
- Port 5200 was free. One temporary TEST service was started, checked, and stopped by
  exact PID 2125. A final health-only lifecycle check used PID 2550, proved repeated start
  does not restart, and stopped that PID too. No persistent service was left running by Codex.

No collector lifecycle assumptions/checks are required for this GPU-first path.

## Split and health contract

The original remote service was intentionally TRAIN-only for collection. D2b used a TRAIN
task; that PASS never established held-out admission. The same server implementation now
accepts `--task-split train|test`, default **train**, with no union mode. Startup reads the
four full P2 official pools `train/test_{single,single_persona}.json`, validates scenario,
official_split, nonempty unique IDs, and train/test disjointness for each scenario.
These are P2 exports of the official tag/index splits, not sampled authorization lists.
No manifests are rewritten. Request-provided split fields cannot change server admission.

Unknown, other-split, and scenario-inappropriate IDs return HTTP 400 before simulator work.
IDs shared by the official Single and Persona pools are legitimately valid in both scenarios.
The official fixed-128 manifests only choose evaluation order/subset. Reset/step/reward,
prompt, parser and session implementation are unchanged.

Health adds `task_split` and retains status, runtime_loaded, active_sessions, shared_runtime,
and protocol versions. `eval_policy.py` now requires healthy explicit TEST health **before
loading Qwen**; legacy/unknown/TRAIN endpoints fail closed. Service source_fingerprint is
`unknown` in this direct deployment (no new upstream full-tree fingerprint claimed).

## Runtime and ownership

- Checkout: `/root/autodl-tmp/shop-rl-eval/code`, synchronized through Git.
- Lifecycle/admission helper Python: `/root/autodl-tmp/shop-rl-preflight/.venv/bin/python`.
- Actual simulator child Python: `/root/miniconda3/bin/python`, the existing Flask/ShopEnv
  runtime; no training package installation or environment changes.
- Helper always launches the same `scripts/remote_teacher_env_service.py` with explicit
  `--host 127.0.0.1 --port 5200 --task-split test`, `UPSTREAM_ROOT=/root/ShopSimulator`.
- Catalog/search: existing `/root/data/shopsim/catalog-fine-runtime.json` and `search_engine`.
- Log: `/root/autodl-tmp/shop-rl-eval/logs/shop_env_test_5200.log`.
- PID record: `/root/autodl-tmp/shop-rl-eval/run/shop_env_test_5200.pid` (JSON PID/starttime/argv).

`eval_env.py` only manages 5200; fixed paths, exclusive lifecycle lock, occupied-port refusal,
no automatic restart. Stop validates PID, process starttime and exact command, then signals
through a Linux pidfd. Active sessions block stop. Stale/foreign PID records fail safely,
without deleting them or signaling another process. No broad kill or 5100 management.
Only stop after evaluation has ended/released its sessions; after an ungraceful kill, an
orphan session needs explicit investigation rather than force-stopping a busy service.

An infrastructure smoke found the existing Miniconda Python lacks `os.pidfd_open` even
though its version is 3.12.3. The validated training Python 3.10.12 supports both pidfd APIs.
The helper now checks support before startup; **use the helper interpreter shown below**.
The service child still uses the existing simulator Python. No `bin/activate` is needed.

## A: user starts TEST service on rtx-4

```bash
(
set -euo pipefail
cd /root/autodl-tmp/shop-rl-eval/code
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python scripts/eval_env.py start
)
```

The helper creates log/run directories, detaches the child with stdout/stderr logging,
writes its exact identity, and waits for TEST health. A healthy owned start is idempotent.
Health before first reset normally has runtime_loaded=false (lazy initialization).

## B: user verifies admission, no Qwen

```bash
(
set -euo pipefail
cd /root/autodl-tmp/shop-rl-eval/code
curl --fail --max-time 5 http://127.0.0.1:5200/health
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python scripts/eval_env.py check
)
```

Expect task_split=test. The check performs one reset/release per scenario using the first
frozen TEST ID, then requires rejection of one official TRAIN ID and an unknown ID. It
prints only IDs/status/health, creates no formal evaluation artifact, and never calls a
teacher API. Initial reset loads the shared catalog/search runtime; allow up to 120 seconds.
Observed both scenarios: TEST `934644943723` accepted/released; TRAIN `834368861472`
and unknown ID rejected. Final active_sessions=0, runtime_loaded=true.

## C: user runs Single fixed-128, only after A/B PASS

```bash
(
set -euo pipefail
cd /root/autodl-tmp/shop-rl-eval/code
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p /root/autodl-tmp/shop-rl-eval/logs
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python -u scripts/eval_policy.py \
  --scenario single --model-path /root/autodl-tmp/Qwen3-8B \
  --manifest /root/data/shopsim/manifests/eval_128_single.json \
  --endpoint http://127.0.0.1:5200 \
  --output-dir /root/autodl-tmp/shop-rl-eval/runs/p6a-base-128-single-seed1 \
  --fixed-128 --seed 1 --no-do-sample --max-new-tokens 512 --reward strict \
  2>&1 | tee -a /root/autodl-tmp/shop-rl-eval/logs/p6a-base-128-single-seed1.log
)
```

Unchanged candidate contract: Base/no adapter, greedy, 512 new tokens, 32768 context,
30 action steps, seed=1, strict scalar and all eight metrics. Native thinking unchanged.
Review the real pilot's token-cap/malformed counts before deciding whether 512 is appropriate.
Terminal: task_start index/128, turn tokens/cap, task_complete actions/outcome/rewards/timing,
summary and exception traceback. Second terminal: `watch -n 2 nvidia-smi` and optionally `top`.

## Single resume and service stop

Ctrl+C once attempts current session release, preserves completed episode rows and updates
summary. Same commit/config/output required; interrupted task resets, completed prefix skips.
No crash/forced-kill recovery guarantee. Never run two writers on the same output directory.

```bash
(
set -euo pipefail
cd /root/autodl-tmp/shop-rl-eval/code
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p /root/autodl-tmp/shop-rl-eval/logs
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python -u scripts/eval_policy.py --resume \
  --scenario single --model-path /root/autodl-tmp/Qwen3-8B \
  --manifest /root/data/shopsim/manifests/eval_128_single.json \
  --endpoint http://127.0.0.1:5200 \
  --output-dir /root/autodl-tmp/shop-rl-eval/runs/p6a-base-128-single-seed1 \
  --fixed-128 --seed 1 --no-do-sample --max-new-tokens 512 --reward strict \
  2>&1 | tee -a /root/autodl-tmp/shop-rl-eval/logs/p6a-base-128-single-seed1.log
)
```

D: stop only the owned idle TEST service:

```bash
(
set -euo pipefail
cd /root/autodl-tmp/shop-rl-eval/code
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python scripts/eval_env.py stop
)
```

## Persona: DO NOT RUN UNTIL SINGLE-128 REVIEW

```bash
(
set -euo pipefail
cd /root/autodl-tmp/shop-rl-eval/code
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p /root/autodl-tmp/shop-rl-eval/logs
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python -u scripts/eval_policy.py \
  --scenario single_persona --model-path /root/autodl-tmp/Qwen3-8B \
  --manifest /root/data/shopsim/manifests/eval_128_single_persona.json \
  --endpoint http://127.0.0.1:5200 \
  --output-dir /root/autodl-tmp/shop-rl-eval/runs/p6a-base-128-single_persona-seed1 \
  --fixed-128 --seed 1 --no-do-sample --max-new-tokens 512 --reward strict \
  2>&1 | tee -a /root/autodl-tmp/shop-rl-eval/logs/p6a-base-128-single_persona-seed1.log
)
```

Return Single run path, terminal summary and `eval/summary.json`, plus errors.jsonl if present.
GPU-first plan only (not launched): TEST endpoint -> Single-128 -> review -> Persona-128 ->
Full Base Single/Persona -> Training Config Freeze -> Direct GRPO Strict. No SFT dataset
freeze prerequisite; future teacher collection is user-controlled and not this path's blocker.

## Tests and scope

- Local: `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest
  tests/env/test_eval_admission.py tests/training -q`: **58 passed, 12 skipped**;
  Flask/opt-in training dependencies account for explicit local skips.
- Remote real Flask: `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  /root/miniconda3/bin/python -m pytest tests/env/test_eval_admission.py
  tests/env/test_remote_teacher_env_service.py -q`: **14 passed**, no skips.
- Both real fixed manifests pass remote `--fixed-128 --dry-run` with endpoint 5200;
  neither formal output directory exists. No model or environment call in dry-run.
- Real helper start -> check -> stop PASS, and final start -> idempotent start -> stop PASS.
  Shell syntax, Python compile, diff whitespace and changed-file secret checks PASS.

Deterministic Flask tests cover both admission directions, both scenarios, unknown/wrong
scenario, ignored client split fields, disjoint manifest validation, default train, union
CLI rejection, health split, independent sessions/release and unchanged reward passthrough.
Lifecycle tests cover exact PID ownership and missing pidfd fail-before-start. One older
fake clickable fixture was corrected to match upstream lowercase button keys; no parser or
simulator behavior was changed. Evaluator regression rejects TRAIN/legacy health before load.

Remote bounded smoke used real Catalog-Fine/ShopSimulator but no Qwen, no env.step, no policy
evaluation. Teacher API calls NONE; formal Base/SFT/GRPO started by Codex NO. No model, venv,
checkpoint, smoke outputs, frozen manifests or teacher artifacts are committed.

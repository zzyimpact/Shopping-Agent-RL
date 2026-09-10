# P6A-Prep — Base fixed-128 command preparation

Starting HEAD/origin/main: `3014a9a`. **BASE_128_COMMAND_READY: NO**.
The commands below match the actual CLI and frozen manifests, but must NOT be launched yet:
current shared ShopEnv implementation permits only TRAIN IDs; its reset handler rejects TEST IDs
with HTTP 400 (source inspection, no TEST reset executed).
D2b exercised a TRAIN task, so its PASS does not establish TEST admission. Earlier in this audit the
CPU-only host returned connection refused on 5100. After the user brought the host back online,
health returned `status=ok`, `task-scoped-v3-multisession`, and zero active sessions; the running
process is `/root/miniconda3/bin/python /root/shopping-agent-rl/scripts/remote_teacher_env_service.py --port 5100`.
Remote deployed source still explicitly checks `settings["train_ids"][scenario]` before reset.
No service was started, replaced, or restarted by Codex.
A healthy TEST-capable endpoint compatible with TeacherEnvClient is required before either command
can be considered ready. Extending/reloading the shared service is outside this prep round's permitted
changes; no TRAIN substitution, split bypass, or second environment implementation was introduced.

GPU evaluation/SFT/GRPO is exclusively user-started from this round onward. Codex ran only read-only
remote checks, CLI dry-runs, fake/local tests, and tokenizer-only CPU checks. No weights loaded.

## Resolved files and decoding

- Host: `rtx-4`; Python: `/root/autodl-tmp/shop-rl-preflight/.venv/bin/python` (Python 3.10.12).
- Stack still torch 2.8.0+cu128, transformers 4.57.6, TRL 1.12.0, PEFT 0.19.1, accelerate 1.15.0,
  datasets 4.8.5; no package install/upgrade.
- The venv Python works with the pinned packages, but `bin/activate` is absent. Invoke the absolute
  Python path below directly; no activation or reinstall is required. The original command drafts
  incorrectly assumed an activation script existed; only the Python executable had been validated.
- Model: `/root/autodl-tmp/Qwen3-8B`; small config/tokenizer/index files exist.
- Dedicated prepared code: `/root/autodl-tmp/shop-rl-eval/code`. Existing `/root/shopping-agent-rl`
  is an older checkout with untracked deployed service files; it is left unchanged.
- Results: `/root/autodl-tmp/shop-rl-eval/runs/p6a-base-128-{single,single_persona}-seed1`.
- Existing client URL convention: `http://127.0.0.1:5100`; currently NOT verified TEST-capable.

Remote manifests were copied read-only to ignored `.cache/p6a-prep/` and checked against P2:

| Manifest under `/root/data/shopsim/manifests/` | Count/split | Ordered IDs SHA256 |
| --- | --- | --- |
| eval_128_single.json | 128 / test | 2bddabe94e2367fff581c771b2bee9d602906e5aa07e83be51a8c69624bd9155 |
| eval_128_single_persona.json | 128 / test | 68db6d538a6c06efe0e181957499bef470be73b01ae77ad2eb48174864ba3410 |

File SHA256 respectively `3d4ddb3dc26cb9616e685e20997ed31a1f940c3bc52467b23ce6cd0cc62bf1a4`
and `0d9bfa893baead87362c01acf7c54e7fac4e452e95ff3b7f11f1f53ecafa6845`.
No new IDs sampled; metadata never goes to policy messages.

Existing evaluator contract retained: **greedy (`do_sample=false`), max_new_tokens=512,
max_context_tokens=32768, max_action_steps=30, seed=1**, base without adapter. Native Qwen chat template
is retained, including its default thinking-enabled generation prefix; D2b `enable_thinking=false`
is not copied into formal evaluation. Native EOS IDs are 151645/151643. Temperature/top_p/top_k
have no sampling effect in greedy mode. Prompt/parser/reward/AgentRollout semantics remain unchanged.
Eight reward metrics always reported; `--reward strict` selects alpha=1 only for the extra scalar.

512 is the existing bounded budget, not a claim that every response fits. Token-cap and malformed counts
are exposed for pilot review; no hidden budget tuning or 128-episode probe occurred. Context overflow
raises explicitly; no silent truncation. Decoding is a project default, not a newly inferred paper setting.

## Commands — blocked pending TEST endpoint readiness

The user must enable their GPU and resolve the endpoint gate first. Run the scenarios separately;
there is no automatic full evaluation continuation. The blocks below are command drafts, not permission
for Codex to launch. Adding `--dry-run` validates files/config without torch/model/ShopEnv calls and
without creating the output directory. A dry-run PASS does not verify endpoint admission or GPU.

### Single

```bash
cd /root/autodl-tmp/shop-rl-eval/code
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
set -o pipefail
mkdir -p /root/autodl-tmp/shop-rl-eval/logs
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python -u scripts/eval_policy.py \
  --scenario single \
  --model-path /root/autodl-tmp/Qwen3-8B \
  --manifest /root/data/shopsim/manifests/eval_128_single.json \
  --endpoint http://127.0.0.1:5100 \
  --output-dir /root/autodl-tmp/shop-rl-eval/runs/p6a-base-128-single-seed1 \
  --fixed-128 --seed 1 --no-do-sample --max-new-tokens 512 --reward strict \
  2>&1 | tee -a /root/autodl-tmp/shop-rl-eval/logs/p6a-base-128-single-seed1.log
```

### Single & Persona

```bash
cd /root/autodl-tmp/shop-rl-eval/code
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
set -o pipefail
mkdir -p /root/autodl-tmp/shop-rl-eval/logs
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python -u scripts/eval_policy.py \
  --scenario single_persona \
  --model-path /root/autodl-tmp/Qwen3-8B \
  --manifest /root/data/shopsim/manifests/eval_128_single_persona.json \
  --endpoint http://127.0.0.1:5100 \
  --output-dir /root/autodl-tmp/shop-rl-eval/runs/p6a-base-128-single_persona-seed1 \
  --fixed-128 --seed 1 --no-do-sample --max-new-tokens 512 --reward strict \
  2>&1 | tee -a /root/autodl-tmp/shop-rl-eval/logs/p6a-base-128-single_persona-seed1.log
```

## Progress, stop, resume, metrics

Terminal JSON events: `environment_health`, `model_loading`, `task_start` (index/128), `turn`
(token count/cap), `task_complete` (actions, eight metrics, status, timing), `summary`; exceptions retain
stderr traceback. Second terminal: `watch -n 2 nvidia-smi` once the user has enabled GPU.

Each completed episode is immediately appended to `eval/episodes.jsonl`; `eval/summary.json` is
atomically replaced after each task and on interruption. `eval/errors.jsonl` records interrupted/error
attempts separately from reward denominators. Infrastructure errors propagate, never become zero reward.
`run_manifest.json` includes config, git commit, package versions, task-manifest and model metadata hashes;
`config.json` is the resolved config snapshot; root `metrics.jsonl` records invocation/final summary.

**Ctrl+C once**: current session release is attempted by AgentRollout finally; prior rows remain. With the
same command/commit/config/output directory, add **`--resume` immediately after `scripts/eval_policy.py`**.
Completed ordered prefix is skipped, interrupted task gets a fresh reset. No environment-memory replay.
Resume validates config/input hashes/commit and only supports greedy decoding; it also works if interrupted
before the first episode was saved. Corrupt/truncated JSONL fails explicitly rather than silently dropping
results; hard-kill/power-loss recovery is not promised. Avoid concurrent writers to one run directory.
Remote environment randomness is not snapshotted, so a retried incomplete task need not reproduce its prior state.

Profiling definitions (per-task and mean/total summary):

- `generated_tokens`: all actual backend generated token counts across turns, including thinking/EOS.
- `input_tokens`: cumulative model input tokens, counting repeated history as actual generation work.
- `full_trajectory_tokens`: final generation's native-rendered input length + its sampled output length;
  excludes post-final observation. It is context size, not cumulative compute or GRPO token provenance.
- `observation_header_tokens`: tokens outside historical assistant content/EOS in that final rendered
  input, including system, user observations, role headers, whitespace and generation prefix. Native
  rendering can remove historical thinking; no attempt is made to call it an append-only sampled stream.
- `environment_wait_s`: reset/step/release client wall time, measured separately from generation.
- `trajectories_per_hour`: completed episodes / sum of their wall time; excludes model load, pauses,
  failed attempts and logging. `generation_tokens_per_second`: all completed generated tokens /
  generation wall time (not kernel-only time). Resume does not divide all results by only new invocation time.
- Malformed/invalid/max_steps/finish/success counts plus infrastructure error and generation-cap counts.

Return both run paths, terminal summary, `eval/summary.json`, root metrics.jsonl and any errors.jsonl for
review. There is no need to send model/checkpoint files. Full 1459/1343 evaluation: **NOT STARTED;
WAIT FOR USER REVIEW OF FIXED-128 PILOT**.

## Validation and scope

`PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/training -q`:
56 passed, 6 skipped. Tokenizer-only opt-in:
`QWEN_TOKENIZER_PATH=../models/Qwen3-8B HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .cache/training-preflight-venv/bin/python -m pytest tests/training/test_eval.py -q`:
7 passed. Real tokenizer test loads only AutoTokenizer; no model/forward/backward.
Both real fixed manifests pass CLI --fixed-128 --dry-run, including count/split/hash/order.
Light tests cover partial run + Ctrl+C/infrastructure error + resume, no duplicate episodes, identity mismatch,
summary denominator across invocations, and unchanged 8 metrics. Full diff/compile/secret checks performed.

Teacher API calls: NONE. Teacher collection/config/artifact/SQLite untouched. Shared service code/lifecycle
untouched. Formal Base evaluation/SFT/GRPO started by Codex: NO. GPU/model weights used this round: NO.

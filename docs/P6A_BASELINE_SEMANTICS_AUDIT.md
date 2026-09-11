# P6A Base behavior audit

## Final decision: formal sampling and task-local resume

POLICY_OBSERVATION_PARITY: PASS. EVAL_DECODING: FROZEN.
EVALUATOR_TASK_LOCAL_RNG: PASS. End-to-end SAMPLED_EVAL_RESUME_SAFE: NO pending
the independently randomized TEST environment described below. The v3 command is
prepared but HOLD: do not start it until that strict-reproducibility blocker is resolved.
The audit/preparation sections below are historical; their REOPENED/candidate labels
describe the earlier stage, not the current decision. Do not rerun those commands.

The user completed Mode B: 6 episodes, 4 finishes, 1 success, 2 max_steps,
0 invalid/malformed/caps/infrastructure errors; mean steps 15.6667.
Rloose=.5317460317, Rstrict=.1666666667, Rsucc=.1666666667,
Rfinish=.6666666667, Rcategory=.6666666667, Rattribute=.5833333333,
Roption=.1666666667, Rprice=.6666666667.
679786107726 changed from greedy max_steps to success in 8 actions;
727130683779 changed from invalid action to terminal finish in 9 actions.
715265851689 and 713324276879 still reached max_steps. Sampling materially changes
and improves the pathological greedy distribution; it does not eliminate all loops.
Six selected cases are diagnostic evidence, not an estimate of formal Base performance.

`configs/training/eval_formal.yaml` now freezes checkpoint evaluation:
enable_thinking=false, do_sample=true, temperature=.7, top_p=.8, top_k=20,
min_p=0, use_model_defaults=false, max_new_tokens=512, max_context_tokens=32768,
max_action_steps=30, seed=base_seed=1, reward_alpha=1 (strict).
This is [PROJECT-FIXED] Qwen3 non-thinking recommended sampling, a project
implementation choice, not paper-disclosed exact serving configuration.
The unified policy already passes use_model_defaults=false. No policy, prompt,
parser, environment, reward, SFT formatting or GRPO sampling changes are made here.
GRPO remains temperature=1/top_p=1/top_k=0/G=8, independent stochastic exploration.

The evaluator alone uses `sampling_seed_strategy=per_episode_manifest_index_v1`.
`manifest_index` is zero-based in the original complete manifest;
`episode_seed=base_seed+manifest_index` (Single-128 seeds 1 through 128).
Transformers set_seed seeds Python, NumPy, torch CPU and all CUDA devices once
before each episode, never before each turn. An interrupted episode resets and
restarts with the same seed; completed episodes are skipped without changing indices.
Both episodes.jsonl and responses.jsonl record index and episode_seed. Incomplete
response attempts remain append-only evidence; repeated task/turn rows after resume
are not extra completed episodes. Timestamps/timing/errors are not expected to match.

Resume requires exact config, manifest file hash and ordered-ID hash, base seed,
seed strategy, model/tokenizer metadata, adapter path and project commit. Keep weights,
adapter contents, environment data, software and hardware unchanged. Task-local seeding
removes dependence on prior tasks' draw counts; it is not a claim of bitwise equality
across GPU architectures/kernels or changed remote observations. No CUDA RNG binary
checkpoint or GRPO-wide reseeding is introduced.

A read-only check found an additional, pre-existing end-to-end limitation in the
deployed upstream: `engine/goal.py:get_existed_goals` samples `price_upper` on each
reset (line 77), and `get_reward` uses that threshold for r_price (lines 234-236).
`engine/engine.py:generate_product_prices` samples multi-price products at load
(line 182); special `search[<r>]` also samples products (line 148).
The service calls get_goals again for each reset. None of these remote-process RNGs
is controlled by evaluator set_seed. Thus matching policy draws alone cannot prove
identical complete observations/rewards, even on identical hardware. Existing parity
PASS is not contradicted: service and upstream share those semantics.
No environment/price/reward fix has been attempted in this evaluator-only change.
Minimal TEST-environment RNG control needs an explicit follow-up scope decision;
do not claim strict end-to-end resume safety or start v3 before resolving it.

Artifacts retained unchanged: v1 has INVALIDATED_BY_GENERATION_CONTRACT; v2 has
PAUSED_FOR_BASELINE_SEMANTICS_AUDIT; both DO NOT RESUME. The Mode B run remains
diagnostic-only and must never be resumed/merged as formal. Only the new v3 directory
below is eligible for formal resume. The evaluator rejects invalidation/audit markers.

### Current user commands: A/B, then C

Prepared commands, ON HOLD pending the environment RNG blocker above.
Run on rtx-4 after resolving it and manually selecting GPU mode. At preparation, 5200 was stopped,
port free, and stale PID 1868 was absent. Only that dead pidfile was archived to
`run/shop_env_test_5200.pid.stale-1868`; no process was signaled. The existing helper
starts one TEST-only instance, never restarts a healthy owned instance, logs to
`logs/shop_env_test_5200.log`, and pins PID ownership with starttime/argv and pidfd.
An ownership mismatch fails safely; never broad-kill or remove a live PID record.

A/B (service uses /root/miniconda3/bin/python; helper uses GPU environment Python):

```bash
(
set -euo pipefail
cd /root/autodl-tmp/shop-rl-eval/code
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python scripts/eval_env.py start
curl --fail --silent --show-error http://127.0.0.1:5200/health
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python scripts/eval_env.py check
)
```

Require task_split=test and TEST-reset/release plus TRAIN/unknown rejection PASS.
No model is loaded by A/B. C, user-run Single fixed-128 v3, no adapter:

```bash
(
set -euo pipefail
cd /root/autodl-tmp/shop-rl-eval/code
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p /root/autodl-tmp/shop-rl-eval/logs
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python -u scripts/eval_policy.py \
  --config configs/training/eval_formal.yaml \
  --scenario single --model-path /root/autodl-tmp/Qwen3-8B \
  --manifest /root/data/shopsim/manifests/eval_128_single.json \
  --endpoint http://127.0.0.1:5200 \
  --output-dir /root/autodl-tmp/shop-rl-eval/runs/p6a-base-128-single-seed1-v3 \
  --fixed-128 --seed 1 --reward strict \
  2>&1 | tee -a /root/autodl-tmp/shop-rl-eval/logs/p6a-base-128-single-seed1-v3.log
)
```

Ctrl+C once releases the active session, preserves completed rows and writes summary.
Do not kill -9. Keep the same deployed commit/config/files. Exact v3 resume:

```bash
(
set -euo pipefail
cd /root/autodl-tmp/shop-rl-eval/code
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p /root/autodl-tmp/shop-rl-eval/logs
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python -u scripts/eval_policy.py \
  --config configs/training/eval_formal.yaml \
  --scenario single --model-path /root/autodl-tmp/Qwen3-8B \
  --manifest /root/data/shopsim/manifests/eval_128_single.json \
  --endpoint http://127.0.0.1:5200 \
  --output-dir /root/autodl-tmp/shop-rl-eval/runs/p6a-base-128-single-seed1-v3 \
  --fixed-128 --seed 1 --reward strict --resume \
  2>&1 | tee -a /root/autodl-tmp/shop-rl-eval/logs/p6a-base-128-single-seed1-v3.log
)
```

Stop the TEST service only after evaluation has exited and released sessions:

```bash
(
set -euo pipefail
cd /root/autodl-tmp/shop-rl-eval/code
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python scripts/eval_env.py stop
)
```

No new probe or automatic Single-to-Persona chain. Normal max_steps,
terminal_unsuccessful and occasional loops are valid Base outcomes, not stop gates.
Stop/review systemic malformed/caps, endpoint failures or context overflow.
Optional second terminal: `watch -n 2 nvidia-smi`.
After 128, return terminal summary, v3/eval/summary.json and eval/errors.jsonl if present.
Review all eight metrics, finishes/successes, max_steps, context max/p95 from saved
episodes, caps/malformed/invalid, trajectories/hour and model/environment wall time.
Compare paper numbers directionally only. Persona is HELD UNTIL SINGLE-128 v3 REVIEW.
Full evaluation, SFT and GRPO remain user-controlled and were not started by Codex.

### Validation for this freeze only

CPU-only `tests/training/test_eval.py`: 9 passed, 1 optional real-tokenizer profiling
test skipped (template/policy unchanged). Includes real Python/NumPy/torch CPU draws,
two-turn fake sampled episodes, uninterrupted tasks 1/2/3 vs interrupt after task 2
consumed RNG then resume 2/3, seed/index metadata, old-marker rejection, exact identity
and commit guards, formal config resolution and model-free CLI dry-run.
No full training tests, model loading, GPU preflight or service/replay run.
Compile, diff whitespace and changed-file secret checks pass;
the deployed formal fixed-128 command is validated with --dry-run only.

## Verdict and scope

**POLICY_OBSERVATION_PARITY_PASS_DECODING_IS_MAIN_VARIABLE** for the bounded audit.
**EVAL_DECODING: REOPENED**, not newly frozen. No protocol bug was confirmed; no
environment, prompt, action parser, reward, context or step-limit fix was made.
Sampling is a user-run diagnostic hypothesis, not a proven cure for loops or a way to
match paper scores. Native thinking remains false. No model was loaded or called.

The v2 run `/root/autodl-tmp/shop-rl-eval/runs/p6a-base-128-single-seed1-v2` has an
additive `audit_status.json`: **PAUSED_FOR_BASELINE_SEMANTICS_AUDIT**, DO NOT RESUME.
It is not invalidated as an implementation bug. All original artifacts remain intact.
61 completed episodes; next task 919591747915 interrupted by the user. Existing summary:
33 max_steps, 19 terminal_unsuccessful, 6 success, 2 invalid, 1 malformed/cap;
25 finishes, 6 successes, no infrastructure errors. No task beyond the seven user-selected
representatives was inspected in depth. The aggregate comes from the existing summary.

Attachment log.md, continuous completed indices 28-61: 34 tasks, 24 max_steps,
8 terminal_unsuccessful, 1 success, 1 invalid; Rfinish=9/34, Rsucc=1/34,
Rloose=.1688375, Rstrict=.0352941. This diagnostic slice is not a formal 128-task estimate.
All 61 summary means (also not a full evaluation): Rloose=.2571038, Rstrict=.1098361,
Rsucc=.0983607, Rfinish=.4098361. Selecting the later slice gives a different distribution.
Neither establishes that the paper difference is an implementation bug.

## Raw responses and invalid action

- 715265851689 turns 6/10/30 explicitly repeat that product features are needed to
  confirm the requirements. Turns 5/7/9 say the page lacks information and select prev.
  Turn 8 repeats the request to check description. All are genuine saved responses,
  ending with im_end, not generation truncation.
- 679786107726 turns 4/6/28 explicitly choose Features to confirm panda/threaded
  neck/spoon requirements even after selecting the matching option. Return turns say
  there is insufficient information. No raw observation history was saved for this task.
- 713324276879 turns 3/7 choose Features, turn 4 selects back to search, turn 5
  reissues the same query, turn 6 selects the same product. Raw Thoughts deliberately
  repeat the request to find a suitable refrigerator; no cap is involved.
- 879763900818 selects the 30g option and buys at turn 4; all eight metrics are 1.
- 590860568914 and 918830854883 finish by Buy Now after inspecting pages, without an
  option-selection action. Their saved metrics have Roption=0, Rfinish=1. This is a
  successful terminal/reward path, not evidence of a broken buy operation.
- 727130683779 selects product **709488631478**, returns from Features via prev,
  then emits `click[black+送挂件]`. That product's actual canonical option is
  **`黑色+送挂件`**. The model translated part of an exact-match option. This is a
  policy argument error, not the same visible button being rejected. The product's
  options were inspected from the already-loaded runtime, without a third reset/replay.
  No historic observation was stored, so the exact original screen cannot be quoted.

## Two-task no-model replay

Exactly two tasks, 715265851689 (30 actions) and 879763900818 (4 actions), fresh reset.
The existing service Flask test client was compared against pinned upstream
WebAgentTextEnv plus shop_agent._handle_reset_action/_handle_interact_action.
Both sides used independent goals/browser/session state with the same read-only
Catalog-Fine/product index/Lucene objects. This avoids two catalog loads; it is not
an independent data/index audit. No persistent server or paid API was started.

Every step matched: policy observation byte-for-byte, clickables, page type, selected
options, search keywords/page/ASIN state, done and reward. AgentRollout consumed the
stored responses through a RecordedPolicy (no model). Each subsequent message list
was exactly prior history + assistant response + corresponding user observation;
roles alternated correctly, without duplicate insertion or dropped turns.

For 715265851689, turns 3/5/7/9/29 have the identical product observation SHA256
`d64a7e1e347dc85bf5bd8812893fcf53d600d8ca5cc8af607b8bd02c5a5b8776`.
The selected solar/electric option persists. Product observations expose description,
features, reviews, buy now, and all 16 product options. Description is a short title;
Features is **empty in upstream too**, with only back-to-search/prev available.
Prev correctly restores the item page. The model sees the same state plus longer
history and chooses to inspect it again. No anti-loop or forced-purchase change is justified.

Historical observations/full message arrays were not persisted in v2; this is a real
evidence limitation. Replay is fresh evidence, not a claim to have recovered original
byte strings. A tokenizer-only cross-check nevertheless reproduces original aggregate
input lengths **exactly**: 292239 / 8231, and final trajectory lengths 17658 / 3387,
for loop/success respectively. All replay generation prefixes close an empty native
think block. This supports history/prefix correctness, but token counts alone are not
cryptographic proof of historic content identity. The other two loop tasks were not replayed.

Replay reports/logs remain outside Git under `/root/autodl-tmp/shop-rl-eval/audit-v2/`
and `audit_replay.log`; local copies are ignored `.cache/p6a-prep/`. At audit time the
machine had returned to 0.5 CPU / 2 GiB mode. One temporary runtime completed within
that quota and exited; no GPU, no Qwen, no collector lifecycle inspection.

## Comparability and performance

Pinned single_eval/agent.py run_task uses `while True`, while upstream shop_agent.py
sets `over` when `len(env.history)>42` (Single starts with 2 entries and appends one
assistant entry per action). This is not a matching 30-step cap. Upstream API calls
also use temperature=0 and do not reveal the paper deployment's actual backend/template
settings. The project is not an exact paper serving/evaluation reproduction.
Keep max_action_steps=30 for fair Base/SFT/Direct-GRPO/SFT-init comparisons; do not
chase reported paper numbers by changing protocol or reward.

The attached 34-task slice reaches 28701 full-trajectory tokens (other tasks 28K/25K/20K+).
32K is a real constraint, but this slice shows no overflow. Leave 32768 unchanged.
Measured from the slice: 72.16 trajectories/hour, 38.91 generated tokens/s, environment
wait 1.076% of episode wall time. Whole saved summary: 89.40 trajectories/hour,
39.31 tokens/s. Generation/prefill dominates; no performance architecture work this round.

## Mode B preparation, not a protocol fix

The only runtime plumbing addition is optional top_k/min_p in GenerationConfig and
sampled QwenPolicy.generate kwargs, so the candidate does not depend on inherited artifact
defaults. Greedy remains unchanged. GRPO sample retains top_k=0/min_p disabled and rejects
accidental use of the evaluation-only filters. Direct/SFT-init GRPO parameters are not changed.

`configs/training/eval_mode_b_diagnostic.yaml`: enable_thinking=false, do_sample=true,
temperature=.7, top_p=.8, top_k=20, min_p=0; seed=1, max_new_tokens=512, context=32768,
max_action_steps=30. Qwen's staged README recommends this non-thinking sampling setting.
No repetition/presence penalty or anti-loop logic is added. TRL/GPU preflight not rerun.

`configs/eval/diagnostic_6_single.json` contains the six user-specified existing IDs in
frozen-128 order: 715265851689, 679786107726, 590860568914, 879763900818, 713324276879,
727130683779. Source file SHA256 is recorded and verified against the remote frozen file.
No random selection, no frozen manifest mutation. diagnostic_only=true and
merge_into_formal_results=false are recorded in both config/manifest. Not a formal score.

## User command, after manually selecting GPU mode

5200 is stopped. Its stale PID 2697 record (process absent, port free) was archived as
`shop_env_test_5200.pid.stale-2697`, with no signal sent. Start/check using the existing
TEST-only helper; no Qwen involved:

```bash
(
set -euo pipefail
cd /root/autodl-tmp/shop-rl-eval/code
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python scripts/eval_env.py start
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python scripts/eval_env.py check
)
```

Only the user runs the following **six-task diagnostic**, not fixed-128:

```bash
(
set -euo pipefail
cd /root/autodl-tmp/shop-rl-eval/code
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p /root/autodl-tmp/shop-rl-eval/logs
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python -u scripts/eval_policy.py \
  --config configs/training/eval_mode_b_diagnostic.yaml \
  --scenario single --model-path /root/autodl-tmp/Qwen3-8B \
  --manifest configs/eval/diagnostic_6_single.json \
  --endpoint http://127.0.0.1:5200 \
  --output-dir /root/autodl-tmp/shop-rl-eval/runs/p6a-decoding-diagnostic-nonthinking-sampling-seed1 \
  --seed 1 --do-sample --max-new-tokens 512 --reward strict \
  2>&1 | tee -a /root/autodl-tmp/shop-rl-eval/logs/p6a-decoding-diagnostic-nonthinking-sampling-seed1.log
)
```

No --fixed-128 and no --resume: this is sampled diagnostic, not a formal/greedy resumed run.
Artifacts: config/run manifest/commit, task hashes, responses.jsonl, episodes.jsonl,
summary.json and errors if any. Never merge with formal results. Inspect changed product/
option/Buy Now choices on former loops, finishes/successes, mean steps, max_steps,
invalid/malformed/caps/context lengths. Six cases do not estimate a reliable baseline score.
If cycles persist, review prompt/model behavior next without expanding this diagnostic.

Existing v2: DO NOT RESUME. No Mode A rerun, fixed-128, Persona, full eval, SFT or GRPO
started by Codex. Sampling candidate is not frozen. No teacher API calls.

## Minimal validation

Exactly two real no-model replays (30 + 4 actions per path), followed by tokenizer-only
history reconstruction. Targeted tests, not the full training suite:

```bash
QWEN_TOKENIZER_PATH=../models/Qwen3-8B HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.cache/training-preflight-venv/bin/python -m pytest \
tests/training/test_policy.py tests/training/test_generation_contract.py tests/training/test_grpo.py -q
```

30 passed. Real Transformers 4.57.6 builds temperature/top-k/top-p/min-p processors;
min_p=0 is checked to leave already-filtered scores unchanged. No forward or optimizer.
Compile, shell syntax, changed-file secret patterns and git diff --check pass.

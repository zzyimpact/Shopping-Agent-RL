# P6A Base behavior audit

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

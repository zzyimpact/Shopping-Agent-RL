# P6A generation contract fix

## Invalidated pilot and evidence limits

Starting HEAD: fa9516a. The user-started run
`/root/autodl-tmp/shop-rl-eval/runs/p6a-base-128-single-seed1` is now
**INVALIDATED_BY_GENERATION_CONTRACT**. DO NOT RESUME, include in Base metrics, or merge
with a corrected run. A separate `invalidation.json` records the reason, evidence limits,
counts and original hashes. Existing episodes.jsonl, run_manifest.json and summary.json
are unchanged (hashes verified after marker creation). No artifacts were deleted.

The stopped run contains **31 completed episodes**, all malformed and all with a cap;
13 fail on the first generation and 18 on a later generation. Across completed episodes:
55 generations, 24 parsed/executed actions, 31 caps, zero infrastructure errors. The next
task `717980338959` was interrupted by KeyboardInterrupt. These are invalidation diagnostics,
not usable Base scores. The user's pasted log was incomplete around task 29.

Six actual saved episode examples (token totals are across the episode, not just last turn):

| task_id | generation count | generated tokens | cap count | parsed actions before failure |
| --- | ---: | ---: | ---: | --- |
| 934644943723 | 1 | 512 | 1 | none |
| 553564918071 | 1 | 512 | 1 | none |
| 941683615646 | 1 | 512 | 1 | none |
| 851382463433 | 2 | 957 | 1 | search, then capped malformed |
| 906111580499 | 2 | 983 | 1 | search, then capped malformed |
| 827409641965 | 2 | 733 | 1 | search, then capped malformed |

**Raw-response diagnosis cannot be completed retrospectively.** The old evaluator removed
`visible_responses` and `messages`, retaining only `response_characters`; generation IDs
were not persisted. The run directory and terminal log contain no decoded assistant text
or final token IDs. Consequently there is no honest way to quote six raw responses, locate
their closing think tags/EOS, or prove their repetition patterns without another generation.
No model was re-run to fabricate this missing evidence. No `<think>` markers in the log
means **not recorded**, not evidence that the model used non-thinking mode.

NATIVE_THINKING_CONFIRMED: YES **for configured runtime mode**; generated native-think
content in this old run is NOT RETROSPECTIVELY VERIFIABLE. Saved config has greedy decoding
(`do_sample=false`), 512 tokens, and no template override. Real tokenizer/config
checks show that omission selects the documented thinking mode. The configuration conflict
is confirmed; native reasoning consuming the cap is a strongly supported causal explanation,
not a raw-token trace proven from this run. The invalidation reason requested by the project is:
`qwen-native-thinking + greedy decoding caused systematic generation-cap truncation before parseable action`.

## Official contract and project decision

The user-provided Qwen3-8B model's README.md, lines 115-145 and 327-328, states:
thinking defaults on; thinking generates `<think>...</think>` before final response;
**DO NOT use greedy decoding** in thinking mode because of degradation/endless repetitions.
It documents `enable_thinking=False` as the hard switch. Source: the locally staged official
model README, also published at https://huggingface.co/Qwen/Qwen3-8B#switching-between-thinking-and-non-thinking-mode.

The actual staged tokenizer renders these generation suffixes:

```text
default: <|im_start|>assistant\n
false:   <|im_start|>assistant\n<think>\n\n</think>\n\n
```

In false mode the empty, already-closed think block is **inserted conditioning context**,
not model-generated reasoning. In GRPO it is owned by the environment/prefix (loss mask=0),
through the existing token bridge. It is not a new SFT target. ShopSimulator's visible
`Thought: ... Action: ...` is a different protocol and remains unchanged.

```ini
[PROJECT-FIXED v1]
qwen_native_thinking = false
```

This is a project implementation choice, not a disclosed paper-author setting. Shared
GenerationConfig always resolves `chat_template_kwargs.enable_thinking=false` and rejects
true overrides. Defaults YAML records the field explicitly for both evaluation and GRPO.
QwenPolicy.generate, prompt_token_ids, and observation_token_ids all consume that setting,
so Base/SFT/GRPO checkpoint evaluation and Direct/SFT-init GRPO share the prefix contract.

Greedy evaluation remains deterministic, 512 new tokens, 32768 context, 30 action steps,
seed=1, strict scalar plus all eight diagnostics. GRPO sampling remains temperature=1,
top_p=1, top_k=0, use_model_defaults=False. No loss/advantage/reward/parser changes.
512 is not raised: old cap evidence was confounded by native thinking. The corrected
user-started pilot is needed to assess the bounded non-thinking action budget.

SFT formatter, assistant-only labels, visible Thought/Action targets, terminal observation
omission, accepted artifacts and datasets are untouched. Tokenizer regression verifies
trainable target remains visible content + im_end, while inference after any adapter uses
the same explicit non-thinking prefix. This does not claim training/inference prefixes
are byte-identical; the closed empty think prefix is an inference template control.

## Minimal runtime and evidence changes

- Greedy generate already omitted sampling kwargs, but inherited model generation_config
  contained temperature=.6/top_p=.95/top_k=20. Copy and neutralize those ignored settings
  (HF greedy-neutral 1/1/50), preserve EOS IDs/pad/stopping, pass use_model_defaults=False.
  Sampling-only kwargs are still not passed for greedy. The model's config is not mutated.
- Model-generated IDs are retained directly, without decode/re-tokenize. Each v2 generation
  now appends `eval/responses.jsonl`: task/turn/time, visible response, raw special-token-aware
  decode, backend IDs, EOS IDs/ended_with_eos, cap status and native-thinking flag. This
  captures a generation before parsing/env.step, including a later interrupted attempt.
  No evaluator-only reset payloads or gold metadata are serialized there.
- Eval CLI refuses any output directory containing `invalidation.json`, before model load.
  New v2 identity includes the resolved false flag in config.json and run_manifest.json.

## TEST service and commands on rtx-4

5200 was stopped when inspected. Its leftover pidfile referred to nonexistent PID 2774;
after validating its identity and the free port, that record was moved to
`/root/autodl-tmp/shop-rl-eval/run/shop_env_test_5200.pid.stale-2774`. No process was signaled.
The service implementation, split admission and lifecycle helper are unchanged this round.

A/B: start the same TEST service (no restart if already healthy), then verify admission:

```bash
(
set -euo pipefail
cd /root/autodl-tmp/shop-rl-eval/code
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python scripts/eval_env.py start
curl --fail --max-time 5 http://127.0.0.1:5200/health
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python scripts/eval_env.py check
)
```

C: only after A/B PASS, user starts corrected Single fixed-128:

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
  --output-dir /root/autodl-tmp/shop-rl-eval/runs/p6a-base-128-single-seed1-v2 \
  --fixed-128 --seed 1 --no-do-sample --max-new-tokens 512 --reward strict \
  2>&1 | tee -a /root/autodl-tmp/shop-rl-eval/logs/p6a-base-128-single-seed1-v2.log
)
```

No extra thinking flag is needed: it is shared project config/runtime. User observes the
first five tasks of this same fixed-128 run; no new mini-manifest. If systematic cap/no-action
malformed persists, Ctrl+C once and review responses.jsonl. Otherwise continue the pilot;
parseable/multi-step/mixed outcomes are expected sanity signals, not guaranteed results.

Resume **v2 only**, at the same commit/config/output after ordinary interruption:

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
  --output-dir /root/autodl-tmp/shop-rl-eval/runs/p6a-base-128-single-seed1-v2 \
  --fixed-128 --seed 1 --no-do-sample --max-new-tokens 512 --reward strict \
  2>&1 | tee -a /root/autodl-tmp/shop-rl-eval/logs/p6a-base-128-single-seed1-v2.log
)
```

Ctrl+C once: current session release attempted, completed episodes preserved, summary updated.
After evaluation ends, stop only the owned idle TEST service:

```bash
cd /root/autodl-tmp/shop-rl-eval/code
/root/autodl-tmp/shop-rl-preflight/.venv/bin/python scripts/eval_env.py stop
```

Old v1: DO NOT RESUME. Persona: NOT STARTED, WAIT FOR CORRECTED SINGLE-128 REVIEW.
No formal evaluation/SFT/GRPO, model load, GPU preflight or benchmark was run by Codex.

## Validation

- Lightweight `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/training -q`:
  60 passed, 7 opt-in skips.
- Full tests/training in the isolated env with QWEN_TOKENIZER_PATH set and
  RUN_TRAINING_PREFLIGHT=0: **62 passed, 5 training/GPU preflight skips**.
- Real tokenizer/config, no model forward: `QWEN_TOKENIZER_PATH=../models/Qwen3-8B
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  .cache/training-preflight-venv/bin/python -m pytest tests/training/test_generation_contract.py
  tests/training/test_eval.py tests/training/test_policy.py tests/training/test_grpo.py -q`:
  32 passed. Includes actual HF config strict validation with preserved artifact EOS,
  actual tokenizer prefix for Base/SFT adapter inference and GRPO turns, untouched SFT labels,
  and fake backend IDs/config assertions including sampling 1/1/0/use_model_defaults=False.

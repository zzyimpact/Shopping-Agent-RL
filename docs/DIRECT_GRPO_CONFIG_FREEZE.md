# Direct GRPO Training Config Freeze

This freeze follows completion of the Single and Single&Persona fixed-128 Base
calibration. It is a project implementation contract, not a claim about an
undisclosed paper serving configuration.

## Frozen contract

- Qwen3-8B BF16, one process and one GPU, fresh PEFT LoRA for Direct GRPO.
- `G=8`, `task_groups_per_update=32`, `256` trajectories per optimizer update,
  `200` optimizer updates, `num_iterations=1`.
- Strict scalar reward (`alpha=1`), `loss_type=grpo`, `beta=0`, grouped reward
  scaling, learning rate `1e-6`, AdamW Torch, linear scheduler, zero warmup,
  grad-norm clip `1.0`, GRPO clip epsilon `0.2`.
- LoRA `r=8`, `alpha=16`, `q_proj/k_proj/v_proj/o_proj`, dropout `0.0`.
- Microbatch `1`; accumulation is `256` (`G * task_groups_per_update`).
- Rollout sampling is non-thinking stochastic sampling: `temperature=1`,
  `top_p=1`, `top_k=0`, `min_p=0`, `use_model_defaults=false`.
- `max_new_tokens=512`, `max_context_tokens=32768`, `max_action_steps=30`.
- Checkpoint every `20` optimizer updates; log every update. Scenario runs and
  output directories are always separate for `single` and `single_persona`.

The formal evaluation sampler (`temperature=0.7`, `top_p=0.8`, `top_k=20`)
is evaluation-only and is rejected by the GRPO CLI contract.

## Split and provenance guards

`train_grpo.py` requires `official_split=train` and the declared scenario when
loading the task manifest. Before loading the model it checks the endpoint
health and requires `task_split=train` and the pinned environment version.
The run manifest records the train-manifest hash, schedule hash, config hash,
model/tokenizer metadata, TRL version contract, endpoint split, seed strategy,
and the git commit. Resume requires an HF checkpoint in the same run and exact
identity equality.

The training rollout uses the same `AgentRollout` protocol and strict reward
normalization as evaluation. Malformed, invalid, max-step, and context-limit
episodes are zero-reward policy outcomes; infrastructure errors propagate and
abort the update. A context-window-capped partial sample is retained in the
token trace but is never parsed or sent to ShopEnv.

## GPU admission (user-run only)

The admission is deliberately two updates with one task group per update and
still uses `G=8`; it is not a formal training run. Stop after `checkpoint-1`
is written, then resume with the identical command to prove checkpoint,
optimizer, scheduler, adapter, and schedule restoration.

The run must show rollout/update metrics, finite loss and grad norm, a changed
LoRA adapter hash, a saved `trainer_state.json` plus `optimizer.pt` and
`scheduler.pt`, and a resumed run whose first rollout uses the next schedule
index. Record wall time, rollout/generation time, environment wait,
generated-token throughput, trajectories/hour, peak VRAM, and update wall time.

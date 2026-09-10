# Engineering Round C — Online GRPO integration

实现范围：TRL 1.12.0 + Transformers generation + PEFT；一个 scenario、一个进程、一个 GPU。
本轮只做 source audit 和 fake/local tests；没有安装训练依赖、加载 Qwen、调用远程环境或训练。
teacher collection 与数据格式完全不变。vLLM 不启用。

## Pinned source contract

只读核对 [PyPI TRL 1.12.0 source archive](https://pypi.org/project/trl/1.12.0/#files)：

```text
trl-1.12.0.tar.gz SHA256
494ec6cfd07097bb8116cc7bec299b067dd765d1c19ed3eca2d874a29979f229
```

源文件 `trl/trainer/grpo_trainer.py` / `utils.py` 的实际路径：

| Surface | 本项目依赖的行为 |
| --- | --- |
| `_get_train_sampler` → `RepeatSampler` | map-style dataset 的 task 已连续重复 G 次；不是 callback 再复制 G 次 |
| `_generate_and_score_completions(inputs)` | 每个 input row 保留 non-policy metadata；交给 `_generate(prompts)` |
| `rollout_func(prompts, trainer)` | 返回 `prompt_ids`, `completion_ids`, `logprobs`；一行对应一条 completion |
| `_generate` | 将返回的 `env_mask` 取出为内部 `tool_mask`；其余 per-row list 传给 reward function |
| `_calculate_rewards` | callback 得到 `prompts`, `completions`, `completion_ids` 和其余字段的列表 |
| `_generate_and_score_completions` | `rewards.view(-1, num_generations)` 计算 group baseline/advantage |
| `_prepare_inputs` | 先生成/评分完整 group batch，再 shuffle/split 为梯度累积 microbatches |
| `_compute_loss` | attention 包含 observation；policy loss mask 为 `completion_mask * tool_mask` |

源码 callback docstring 的 “no duplication” 描述与实际 sampler 路径不一致；实现以以上实际
调用路径为准。没有复制/修改 TRL sampler、generation、advantage 或 loss 实现。

训练依赖 proposal 保持：`torch==2.8.0`, `transformers>=4.56.2,<5.0`, `trl==1.12.0`,
`peft>=0.13,<0.20`, `accelerate>=1.4,<2`, `datasets>=4.7,<5`。运行入口检查 TRL 精确版本；
依赖本轮未安装。下述 resume source 核对 Transformers v4.56.2；最终解析出的其他版本仍需 preflight。

## Metadata bridge 与统一 rollout

dataset row 只有 transport placeholder prompt 和独立的 `task_id/scenario/schedule_index`。
placeholder 是空 user message，**不送入 tokenizer**。实际 initial policy prompt 始终来自
AgentRollout 的 `env.reset`、现有 prompt 构造函数与 policy-visible observation。

`ShopGRPOTrainer` 只覆盖一个 **private** method：`_generate_and_score_completions`。
在调用 parent 前校验并暂存当前 batch 的三个 metadata 字段，在 `finally` 清除。
parent 收到浅拷贝的 row dict，避免其合并 reward extra fields 时修改 Dataset batch。
callback 按当前 row 顺序读 metadata，不做 prompt-text lookup、不添加 task ID/gold 到 prompt。
升级 TRL 必须重新审查这个小的 private surface。

```text
task schedule → TRL RepeatSampler → temporary metadata bridge
                                          ↓
                            current Trainer model → QwenPolicy
                                          ↓
                             AgentRollout (one env loop)
                                          ↓
                            token trace + terminal metrics
                                          ↓
                             native TRL GRPO loss/update
```

Base/SFT evaluator 仍调用 `generate(messages) -> str`。GRPO 通过同一个 AgentRollout 的
`run(task_id, sampling=...)` 开启 capture；reset、prompt、parser、step、reward、release/close
仍只有一套。callback 每批从 `trainer.model_wrapped` unwrap 当前 policy，不持有过期模型副本；
暂时 eval 以 generation，结束/异常后恢复 training 状态。

## Token contract

`QwenPolicy.sample(input_ids, sampling=...) -> PolicySample(text, token_ids, logprobs)` 使用
Transformers 实际生成的 IDs 和 `compute_transition_scores(..., normalize_logits=True)`。
不会对 decoded assistant text 再 tokenize 来替代 sampled IDs；text 仅给既有 parser/env。

```text
prompt_ids: initial native Qwen prompt + first assistant generation prefix

completion_ids:
  sampled assistant A1 (+ sampled EOS)    env_mask=1, actual sampling logprob
  inserted closure if A1 lacked EOS      env_mask=0, logprob=0.0
  newline + native user observation
    + next assistant generation prefix  env_mask=0, logprob=0.0
  sampled assistant A2 (+ sampled EOS)    env_mask=1, actual sampling logprob
```

每行 `len(completion_ids) == len(logprobs) == len(env_mask)`。程序插入的所有 ChatML header、
observation delimiter、可选 non-thinking prefix 均为 0；sampled EOS 为 1。
初始 prefix 属于 prompt，本来就没有 completion loss。TRL 将外部 logprob 的 0.0 占位符
作为普通有限数值读入；loss 中由 `tool_mask` 排除，无需 NaN/None。
TRL 对完整 supplied completion 建 attention mask，不会在中间 assistant EOS 切断。

后续 generation 的 conditioning 是累计的原始 IDs。只对新 observation 与插入前缀应用
native Qwen template/tokenizer。这避免 whole-history Qwen rendering 可能删除历史 `<think>`
或改变 token boundaries。现有 text-only evaluator 不改；训练端精确保留实际 sampled history。
真实 tokenizer suffix / forward-conditioning 对齐属于 training-env/GPU preflight。

只有确定还要生成下一 assistant turn 时才 append observation token。terminal、invalid、
max steps 或 context-limit 后的 observation 留在 rollout messages/diagnostics，不进入尾部
completion。超长初始 prompt 报错；后续空间不足则结束为 `context_limit`，保留已生成 token，
不截断历史。malformed/invalid/max_steps/未成功 terminal 使用原有 policy-failure metrics；
基础设施异常直接传播并中断训练，不转成 reward=0，不新增 retry framework。

## Groups、更新与 resume

正式 defaults 显式为 `loss_type="grpo"`, `G=8`, `beta=0.0`, `scale_rewards="group"`,
`learning_rate=1e-6`。每个 update 32 个 task groups、200 updates 来自 project spec。
microbatch/LoRA rank/target modules 必须显式提供，尚待 GPU-PREFLIGHT。

单进程 `generation_batch_size = G * task_groups_per_update`，
`steps_per_generation = gradient_accumulation_steps = generation_batch_size / microbatch`，
`num_iterations=1`。每个 optimizer update 使用一批新 online trajectories，下一批读取更新后的模型。
batch 大小必须整除 microbatch；完整组校验失败直接报错。

例如两任务 G=3：sampler 输入顺序 `A,A,A,B,B,B`；callback 同序执行六次
`env_factory → reset → episode → release → close`，不再次 G duplication，不共享 episode state。
该顺序在 TRL group reward 计算前不变；之后 TRL 可以自由 shuffle microbatches。

只有 task ID 顺序是预先构造的：独立 `random.Random(seed)` 打乱训练 task 列表并按需重复，
生成 `groups * max_steps` 个 schedule rows，hash 写入 run manifest。
`shuffle_dataset=False` 避免 TRL RepeatSampler 的私有 RNG cursor；它只连续重复 task index。
这是 task scheduling，不是预生成 response/trajectory dataset。

Transformers v4.56.2 Trainer 的 resume 会从 `trainer_state.global_step` 推导已消费的
`global_step * gradient_accumulation_steps` 个 dataloader batches，并使用 `skip_first_batches`。
TRL RepeatSampler 恰好把每个 generation batch 重复 `steps_per_generation` 次。
因此 optimizer checkpoint 边界续跑落在下一完整 generation batch；TRL 初始化为空的 buffer
会重新生成。HF 负责 adapter、optimizer/scheduler、Trainer state 和 RNG 保存/恢复；项目仅保存
config/manifest/schedule identity 并透传 `resume_from_checkpoint`，不自建 cursor/checkpoint。
中断中的未提交 update 从上一个 checkpoint 重跑。环境非确定性不保证逐 token replay。

## Reward、lineage、logging

唯一 scalar：`R_alpha = alpha * Rstrict + (1-alpha) * Rloose`。
CLI `--reward strict` → 1，`--reward loose` → 0，或 `--alpha 0.5`，二者互斥。
内部只保留 `reward_alpha`。每条 trajectory 仍由现有 wrapper 得到全部八 metrics。
custom rollout 返回 `rollout_reward/rollout_metrics/rollout_status`；reward callback 只返回
预计算的 scalar 列表，不 decode 推断 purchase、不再次访问环境/scorer。

同一 `train_grpo.py`：Direct = BF16 base + fresh LoRA；`init: sft_adapter` = base +
`PeftModel.from_pretrained(..., is_trainable=True)`，继续该 adapter，不叠第二个 adapter。
后者要求 `sft_run_dir` 和其 `checkpoints/` 下的 `adapter_path`，校验 SFT run scenario、
base/tokenizer 路径，记录 source manifest/config hash。RL checkpoint 写入新的 GRPO run，
不会覆写 SFT run。单个 run 不混合 scenario。

`run_manifest.json`, `metrics.jsonl`, `checkpoints/`, `eval/` 复用 Round B helpers。
记录每 update 的 task IDs/schedule indices、reward mean/std、八 metrics、group std、
zero-variance group fraction、steps、invalid/malformed/context-limit rate、policy token count、
rollout/update wall time。Trainer native logs 追加 grad norm 等；实际 GPU peak allocation 由
runtime callback 读取，本轮 fake tests 不伪造 GPU 指标。评估继续独立使用现有 Evaluator。

## Runtime boundary / 下一轮

fake/local tests 覆盖 token ownership、logprob 对齐、无 re-tokenization、末尾省略、G grouping、
独立 resets、metadata 不泄漏、reward/CLI、infra error、当前模型引用与 lineage/resume 透传。
**Source-level contract verified; runtime integration = TRAINING-ENV/GPU PREFLIGHT REQUIRED.**

- Training-env CPU：在独立环境解析上述依赖；用实际 TRL/HF tiny CPU model 验证 subclass、
  reward extra fields、sampler/buffer/resume；真实 Qwen tokenizer 检查 suffix IDs 与 SFT EOS labels。
- GPU：Qwen3-8B BF16 + fresh/SFT LoRA、stochastic generation、真实 sampling logprobs 与
  differentiable likelihood 对齐、完整 masked multi-turn batch、one GRPO optimizer update、
  下一 rollout 使用更新权重、checkpoint save/reload 与 task 顺序、显存/吞吐。
- 真实 ShopEnv episode 只在用户允许增加 remote load 的后续 preflight 执行。
- 初始 temperature=1/top_p=1/top_k=0、AdamW、warmup=0、clip epsilon=.2、grad norm=1
  是 project choices；sampling 可配置且始终 stochastic。TRL 在 training logits 上使用
  temperature；top-p 截断不进入它的 differentiable likelihood。top_p<1 的 override 会采用
  TRL 标准行为，需要明确评估这一 sampling/loss distribution 差异。
- `num_iterations=1`、generation 与 accumulation 对齐时，非 vLLM 路径使用当前 detached
  likelihood 作为 old likelihood；返回的真实 sampling logprobs 不会自动替换它。保留 TRL
  该原生语义；`disable_dropout=True`, `mask_truncated_completions=False` 显式设置。

训练命令形状（占位路径与 GPU preflight 参数必须在下一轮确定；本轮不运行）：

```bash
python3 scripts/train_grpo.py \
  --config configs/experiments/single_direct_grpo_strict.yaml \
  --model-path /persistent/models/Qwen3-8B \
  --train-task-manifest /path/to/single_train.json \
  --output-dir /persistent/runs/single-direct-grpo \
  --per-device-train-batch-size <microbatch> \
  --lora-r <rank> --lora-alpha <lora-alpha> --target-modules <modules> \
  --reward strict
```

Persona 使用另一个小 override；SFT-init 额外指定 `--init sft_adapter --sft-run-dir ...
--adapter-path ...`。同一 run 恢复额外传 `--resume-from-checkpoint .../checkpoints/checkpoint-N`，
config/lineage/task schedule identity 必须相同。不需要重新格式化 teacher 数据来启动 Direct GRPO。

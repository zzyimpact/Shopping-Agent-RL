# Engineering Round D2 / P5 — Qwen3-8B GPU preflight

2026-09-10。起点 `main@1cd95a11c12f0bae6f061ca7e41dee8dbc2930a4`，初始工作区干净。

**GPU_PREFLIGHT: FAIL（尚未完成真实 ShopEnv 两项 smoke）**。
GPU/package/model/SFT/FakeEnv GRPO 的真实运行均 PASS；不是硬件故障。
`rtx-4` 的现有配置 endpoint `127.0.0.1:5000`、`:5100` health 均 connection refused，
本地既有 `:5500` 也不可用。已请求可复用 endpoint，未启动、重启或替换任何服务。
因此真实 ShopEnv AgentRollout、真实 ShopEnv GRPO batch **未执行**；不以 FakeEnv 代替验收。

**READY FOR FORMAL GPU EXPERIMENTS: NO**：唯一尚未完成的接线条件是可访问、兼容现有
TeacherEnvClient 的 ShopEnv endpoint，以完成上述两个有界 smoke。

## Environment / artifact

所有 SSH 操作使用 `rtx-4`。独立 code checkout、venv、输出位于：

```text
/root/autodl-tmp/shop-rl-preflight/code
/root/autodl-tmp/shop-rl-preflight/.venv
/root/autodl-tmp/shop-rl-preflight/results
```

未修改 `/root/miniconda3` 或 teacher environment。使用已有系统 Python 3.10.12 创建
venv，pip 仅 bootstrap 到该 venv；与 D1 Python 3.10.21 的 patch 版本差异明确保留。

| Runtime | 实测 |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Driver / compute capability | 595.58.03 / 12.0 |
| nvidia-smi VRAM / torch usable | 97887 MiB / 94.9724 GiB |
| Python / torch | 3.10.12 / **2.8.0+cu128**（Linux cp310 wheel） |
| torch CUDA / driver advertised CUDA | 12.8 / 13.2 |
| transformers / trl / peft | 4.57.6 / **1.12.0** / 0.19.1 |
| accelerate / datasets / tokenizers | 1.15.0 / 4.8.5 / 0.22.2 |
| PyYAML / httpx / pytest | 6.0.3 / 0.28.1 / 9.1.1 |
| pip check / package + project imports | PASS |
| CUDA allocation / BF16 matmul / real 8B forward | PASS |

API versions 保持 `requirements-training.txt`。初始 torch allocated/reserved 均 0；
未安装 FlashAttention、vLLM 或 generation server。

用户提供相对路径 `autodl-tmp/Qwen3-8b`；实际存在的大小写为
**`/root/autodl-tmp/Qwen3-8B`**。config、generation_config、tokenizer 文件、index、
SHA256SUMS.txt 全部存在，index 的全部 shard 引用都存在：

| Shard（model-N-of-00005.safetensors） | Bytes |
| --- | ---: |
| 00001 | 3,996,250,744 |
| 00002 | 3,993,160,032 |
| 00003 | 3,959,604,768 |
| 00004 | 3,187,841,392 |
| 00005 | 1,244,659,840 |
| Total | **16,381,516,776** |

**MODEL_ARTIFACT_READY: YES**，无需 staging。未重新 hash 全部权重，未下载/传输模型。
所有加载均 `local_files_only=True`，并设置 HF offline 环境变量。

## Base / sampling / LoRA

真实 `Qwen3ForCausalLM`：8,190,735,360 params，BF16、cuda:0、SDPA。
load 3.64 s，allocated 15.265 GiB / reserved 15.285 GiB。无 LoRA base forward finite。

修复后的 QwenPolicy.sample：prompt 363 tokens，generated **61** tokens（含 EOS 151645），
1.395 s，**43.72 tok/s**，peak 15.463 GiB。逐项比对 backend sequences 与 PolicySample IDs、
normalized transition scores 与 logprobs，全部对齐且 finite；没有 generated decode→retokenize。
实测有效 temperature=1、top_p=1、top_k=0、repetition_penalty=1。

全部 LoRA smoke 使用 r=8、alpha=16、q_proj/k_proj/v_proj/o_proj、dropout=0。
7,667,712 trainable params（相对 base 0.09361%），所有 requires_grad 参数名均为 lora_*。
直接 LoRA forward/backward/AdamW step：loss 5.06591、grad norm 3.00773，adapter hash 改变，
peak 15.416 GiB。base 参数冻结，optimizer 只包含可训练 adapter。

## SFT GPU update / resume

使用现有 tokenize_with_assistant_mask → Dataset → build_sft_trainer → train_sft，
没有另一套 SFT optimization 实现。两条合成 fixture，长度 **35 / 1530**，各 14 assistant
TRAIN tokens，其余 **21 / 1516 MASK**；collator 原样保留 labels，最终 token 为 TRAIN EOS，
trailing terminal observation 不进入 input。没有读取增长中的 accepted dataset。

microbatch=1、accumulation=1、gradient checkpointing=on；两条 fixture 共计划 2 steps，
首进程在 step 1 保存停止，第二个全新 model/Trainer 进程从 checkpoint-1 恢复到 step 2。
这是 **TEST ONLY**，不是正式 epoch。

| Actual step | Length | Loss | Grad norm | Fwd+bwd seconds | Peak allocated GiB | Reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1530 | 6.5036 | 5.18842 | 0.755 | 19.335 | 19.580 |
| 2（resume） | 35 | 6.1890 | 4.64790 | 0.675 | 18.865 | 18.928 |

每阶段 train + checkpoint/final adapter export 为 1.505 / 1.599 s（不含模型 load）。
两步 loss/grad finite，adapter 参数均改变。resume 的 on_train_begin 实测 global_step=1、
adapter hash 等于上一阶段输出、optimizer state 非空、scheduler.last_epoch=1，最终 step=2。
HF 原生 optimizer.pt/scheduler.pt/trainer_state.json 均存在，run identity 检查通过。

## Real 8B GRPO / FakeEnv

真实 ShopGRPOTrainer + TRL 1.12.0 + CUDA + Qwen3-8B LoRA；G=2、1 group/update、
microbatch=1、accumulation=2、num_iterations=1、use_vllm=false、GC=on。
FakeEnv 每 episode 两步；仅测试 generate 候选前缀受限，以确保 parsable multi-turn/EOS 和
非零 group variance。仍由真实模型采样 backend token IDs/logprobs，没有伪造 PolicySample。
该 fixture 不是自然购物行为或正式采样性能的测量。

- 实际 callback metadata：checkpoint 前 `[TEST_A, TEST_A] / indices [0,0]`；
  新实例 resume 后 `[TEST_B, TEST_B] / [1,1]`，group 未拆开、未从 index 0 重来。
- transport prompts 均为空 user message；真实 prompt 仍由 AgentRollout reset 构造，
  task metadata 只经过 private bridge，不加入 tokenizer-visible transport prompt。
- 每组 rewards `[1,0]`，advantages `[0.7070068,-0.7070068]`；reward extras 直接消费
  AgentRollout 已计算的八指标与 scalar。没有通过重建文字重复跑环境/scorer。
- 每 completion **45 tokens = 26 sampled + 19 inserted**；IDs/logprobs/env_mask 同长，
  sampled EOS 保留，中间 EOS 后 attention 未截断，terminal observation 被省略。
- 真实 TRL prepare 的 completion_mask 对全部 45 tokens 为 1，tool_mask 等于 env_mask。
  autograd hook 在实际 loss backward 验证 inserted token 的 per-token logprob 梯度严格为 0，
  sampled token 梯度非零，两阶段共 4 次 microbatch 检查通过。
- 真实 prepared inputs 不含 old_per_token_logps；aligned non-vLLM path 保持 current
  per-token likelihood.detach() 语义。sampling logprobs 是采样分布记录（本 fixture 有候选限制），
  **不作为 old-policy likelihood**。没有修改 TRL loss 或 likelihood 定义。

| Step | Loss | Grad norm | Rollout s | Generate s | Update s | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0 | 0.947499 | 2.113 | 2.063 | 0.750 | 15.572 |
| 2（resume） | 0.0 | 0.963286 | 2.085 | 2.037 | 0.689 | 15.630 |

对称 group 的平均 loss=0 是预期值，梯度非零且参数更新。train + checkpoint/export
3.481 / 3.629 s；update time 是现有 callback 的非-rollout wall time，包含 Trainer 开销。
FakeEnv 环境网络等待为 0，不能据此推断真实 ShopEnv wait。

```text
initial: 95be3c77600d597462b8fb601d2a9aaa24937182b518d8e3d828e033d5c3f11d
step 1:  9c0ec06f0756b715dfb2415668349acbd287ea826ff97686eaf88b7b260bfc52
step 2:  ba1aa02209c9b95531802b9dceb3927d1d277345b0aab05a57a9e0bc3ebc8d58
```

新实例 restore 时 adapter/global_step/optimizer/scheduler/run identity 全部检查通过。
下一次 rollout 的 current Trainer model hash 等于 step 1 hash，明确不是初始化模型。

## Correctness fix

真实 Qwen artifact 的 generation_config 提供 temperature=.6/top_p=.95。Transformers 4.57.6
默认把自定义 GenerationConfig 中**等于全局默认值**的字段替换成 model defaults；所以之前
显式请求 1/1 仍被覆盖。D1 tiny model 没有不同的 artifact defaults，未触发该条件。

最小修复：仅 QwenPolicy.sample 的 model.generate 增加 `use_model_defaults=False`。
保留既有 BOS/EOS/PAD 处理、evaluator 路径和 teacher prompt/parser。
CPU regression 将真实 tiny model 的 artifact defaults 设为 .6/.95/top_k20/repetition1.1，
检查实际 HF resolved config 仍为 1/1/0/1，并执行真实 generation/transition-score 对齐。
GPU base 重跑再次检查实际 resolved config 通过。原错误采样的 measurement 不用于上表。

## Preliminary recommendations — NOT FROZEN

| Parameter | 下一轮建议起点 |
| --- | --- |
| SFT / GRPO LoRA | r=8、alpha=16、q/k/v/o projections；正确性已验证，效果未验证 |
| SFT microbatch / accumulation | 1 / 32，维持现有 effective batch 32 |
| GRPO microbatch / accumulation | 1 / (G × groups_per_update)；现有 G=8、groups=32 时为 256 |
| Gradient checkpointing | 两者 on |
| Sampling | temperature=1、top_p=1、top_k=0 作为下一轮起点 |

所有以上选择均 **PRELIMINARY / NOT FROZEN**；未修改正式 experiment hyperparameters。
smoke 的 enable_thinking=False、max_action_steps=2、G=2 都只是 fixture；正式 action cap 30 不变。
短/长 SFT 测试来自不同冷启动进程，内存包含初始化工作区，不能从两点拟合长 context 显存。
本轮未测 32768-token 边界、正式完整 trajectory 长度或高 microbatch；冻结 batch 前仍需代表性长度测量。

**PERFORMANCE OPTIMIZATION RECOMMENDED: NO（当前证据不足以授权实现）**。
FakeEnv fixture 中 generation 占 rollout+update 约 72%，但候选限制/Python callback 和极短
completion 不能代表正式 workload，真实 environment wait 未测。
**PERFORMANCE-PROFILING REQUIRED**：真实 ShopEnv endpoint 可用后补有界实测，再决定
sequential rollout 是否值得优化。没有实现 vLLM、async、batched episodes 或分布式路径。

## Reproduction / validation

脚本为 `scripts/gpu_preflight.py`，调用现有 training core；每 stage 新进程。
在独立 code checkout 中使用以下命令（以下 loop 已执行前六个 stages）：

```bash
cd /root/autodl-tmp/shop-rl-preflight/code
../.venv/bin/python -m pip check
for stage in gate base sft sft-resume grpo grpo-resume; do
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
  ../.venv/bin/python scripts/gpu_preflight.py --stage "$stage" \
    --model-path /root/autodl-tmp/Qwen3-8B \
    --output-root /root/autodl-tmp/shop-rl-preflight/results
done
```

重新开始完整 preflight 应选择新的 output-root；现有训练输出只允许显式 resume。
`real-rollout` / `real-grpo` 还要求 `--endpoint` 和 `--task-id`，未执行。
endpoint 可用后只选一条 train task：1 个两步 rollout + G=2 的两步 batch，最多 3 episodes；
不为制造 reward variance 扩大批次，不重启服务。

本地 `.cache/d2-preflight/` 保存小型 measurement JSON；remote results 保存 TEST ONLY
checkpoints。模型/venv/cache/checkpoint/生成 rollout 均不提交。代码变更只含 sampling fix、
对应 regression、GPU preflight 脚本、machine host/model path 和本文档。

Teacher API calls: **NONE**；真实 ShopEnv reset/step: **NONE**（仅 health probes）。
Formal teacher collection interrupted: **NO**。Formal SFT/GRPO/full evaluation started: **NO**。
GPU used: **YES**，仅本轮授权 preflight。未修改任何 teacher 进程/config/artifact/ledger。

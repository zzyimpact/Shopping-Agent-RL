# Engineering Round D2b — ShopEnv integration completion gate

2026-09-10。起点 HEAD/origin/main 均为 `df5db540f54c4aaa3cea2a5895b33799cc736ed4`。
初始工作区已有用户修改：`scripts/teacher_env_up.sh` 默认 host 改为 rtx-4；保留，不纳入本轮提交。

```yaml
SHOPENV_SERVICE_READY: YES
REAL_QWEN_SHOPENV_ROLLOUT: PASS
REAL_SHOPENV_TOKEN_CONTRACT: PASS
REAL_SHOPENV_GRPO_INTEGRATION: PASS
REAL_SHOPENV_GRPO_UPDATE: INCONCLUSIVE_DUE_TO_REWARD_VARIANCE
GPU_PREFLIGHT_FINAL: PASS
READY FOR FORMAL GPU EXPERIMENTS: YES
```

这是 D2 唯一缺口的补验，不改写 [D2 当时的 FAIL](ENGINEERING_ROUND_D2_GPU_PREFLIGHT.md)。
未重新运行 base/LoRA/generation benchmark、SFT、FakeEnv GRPO 或 checkpoint/resume。

## Root cause / topology / safety

分类：**service 未运行 + 按需生命周期**，不是 ShopSimulator 功能损坏或改用其他 port。
D2 时 5000/5100 connection refused、没有 service process。现有 up 脚本通过 nohup 启动
`remote_teacher_env_service.py`，down 脚本按 ownership/PID 停止它；不是自动常驻系统服务。
5000 为较早 runtime service 配置，兼容 TeacherEnvClient 的 task-scoped service 预期端口是 5100。
无法从现存证据判定 D2 之前是谁/何时停过服务，不作推测。

D2b 开始时服务已在 **19:24:57（remote process 时间）** 启动，早于本轮操作：

```text
host: rtx-4
PID: 1748
/root/miniconda3/bin/python /root/shopping-agent-rl/scripts/remote_teacher_env_service.py
  --port 5100 --source-fingerprint 2c8373d721766f0c1c5c98292bc59bbea2f6bbaef139eb20ac00fb09fd5ef67b
training URL: http://127.0.0.1:5100
teacher URL: http://127.0.0.1:5500
existing local tunnel PID: 3632 (5500 -> rtx-4 localhost:5100)
```

本地 ps 未发现 formal collector/Python 进程，remote 相关 Python 为上述服务；没有声称
此前 collection 一直在运行。现有 tunnel 与两个 localhost health 均健康，active_sessions=0。
未读取 collection config/API profile、SQLite、accepted/raw；collector CLI 代码的默认 endpoint
为 5500。无需查 secret 或启动 collector。

决策：**REUSE EXISTING SERVICE**。未调用 up/down、未重启 PID、未新建 tunnel/服务。
service/protocol 文件小型 SHA256 与本地 HEAD 一致。health 确认：
`task-scoped-v3-multisession`、`single-eval-policy-v1`、`p3a-visible-action-v2`、shared_runtime=true、
runtime_loaded=true、max_sessions=32。共享 Catalog-Fine/Lucene/SimServer，独立 session。
仅使用已有加载资源，未重建 catalog/index，未暴露公网端口。

## Bounded real rollout

复用 `/root/autodl-tmp/Qwen3-8B` 和 `/root/autodl-tmp/shop-rl-preflight/.venv`，未安装包。
只读取 train_single.json 第一条 task ID **834368861472**，不基于 reward 挑选任务。
三条 episodes 全为 Single；每条最多 2 action steps，**TEST FIXTURE ONLY，正式上限仍 30**。

独立 AgentRollout 实际动作：

```text
search[金丝胡桃木 日式简约 屏风]
click[834368861472]
```

2 次 env.step，status=max_steps；八指标及 r_alpha 均 0。不是成功购买，符合 smoke 要求。
第二次生成确实使用第一次 step 返回的 2097-character observation。模型原协议/prompt/parser
全部不变；temperature=1/top_p=1/top_k=0、enable_thinking=false、max_new_tokens=256 为测试参数。

## Real token provenance

初始 prompt 304 tokens，第二次 input 1997 tokens；completion 1816 = **242 sampled + 1574 external**。
实际采样输入逐次等于累计 stream 前缀，采样 IDs/logprobs 原样匹配 completion 对应区间。
external span [119,1693) 精确等于原生模板对真实 observation/header 的渲染；前一 token 是 sampled EOS。
最终 token 属于 sampled turn，最后一次 step observation 未附加到 completion。

初始 prompt 与 reset 的 `policy_context + policy_observation` 白名单投影逐 token 相等；
记录器不保存完整 reset/step gold payload。metadata bridge 的 transport prompt 均为空 user message。
task/schedule bookkeeping 没有插入 prompt。商品 ID 可以合法出现在搜索结果和 click action 中，
这不是 task metadata 泄漏，不能用禁止一切商品 ID 的字符串检查替代 provenance。

真实 GRPO parent prepare 实测全部有效 completion tokens attention=1；tool_mask 等于 env_mask：
sampled=1，external=0。中间 EOS 后的 observation 和下一次 sampled turn 没被截断。
本轮没有 terminal purchase；验证了 max-step/malformed 后不附加无后续生成的 observation，
terminal 路径沿用 D1/D2 已通过的测试，没有声称本轮发生真实终局购买。

## Real GRPO integration

G=2、一个 task group、microbatch=1、accumulation=2、LoRA r8/a16/qkvo，单次 Trainer step。
独立 session：`a18255f9eda54a39a08d91d7c30360e9`、`e74a576bfacb4307aee25cfc14ffd5eb`。
每成员 reset 一次、成功 search step 一次、随后第二次 generation、release 一次。
总共（含独立 rollout）3 resets、4 steps、3 releases，未追加 trajectories。

| Member | Completion | Sampled | External | Status | Reward / advantage |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 1792 | 228 | 1564 | malformed_action | 0 / 0 |
| 2 | 1776 | 214 | 1562 | malformed_action | 0 / 0 |

第二次生成均到达 128-token 测试 budget，未形成 canonical action；不改 parser/prompt。
callback metadata 顺序为同 task、indices [0,0]；reward callback 仅调用一次，直接返回 [0,0]，
回调前后 env 请求数不变。TRL prepare、group advantage、loss/backward/optimizer step 正常走完，
global_step=1、loss=0、grad_norm=0、adapter hash 未变。
**INCONCLUSIVE_DUE_TO_REWARD_VARIANCE**，不是 integration failure；D2 已验证非零 advantage 的8B更新。
没有为了制造非零梯度修改 reward 或挑选额外任务。

## Bounded performance / parameters

| Measurement | Standalone rollout | G=2 real batch |
| --- | ---: | ---: |
| Generated tokens | 242 | 442 |
| Generation seconds | 5.501 | 14.287 |
| Reset + step wait seconds | 0.154 | 0.180 |
| All env wait incl. release | 0.162 | 0.187 |
| Episode wall seconds (sum for G=2) | 5.717 | 14.965 |
| Generation / episode fraction | 96.2% | 95.5% |
| Env / episode fraction | 2.8% | 1.2% |

Generation wall 包含框架开销，不等于 CUDA kernel-only compute。GRPO 非-rollout update wall
1.298 s；含 checkpoint/export 的 stage 16.874 s，peak allocated 20.005 GiB。

**REAL_SHOPENV_PERFORMANCE_OPTIMIZATION_RECOMMENDED: INSUFFICIENT_EVIDENCE**。
这个 warm-service、同一 task、短轨迹样本不支持正式吞吐推断；当前样本主要耗时在 generation，
不是 env wait。未实现任何吞吐优化。所有 LoRA/batch/generation 建议继续 PRELIMINARY / NOT FROZEN，
正式冻结前另做代表性 sequence-length/VRAM profiling；没有从短样本推断 microbatch 上限。

## Changes / reproduction / verification

无 service/core correctness 修复。仅扩展原 `gpu_preflight.py` 的真实环境诊断：
session/request timing、可见 projection、精确 inserted token audit、reward callback passthrough 计数。
新增纯本地 `test_gpu_gate.py`：拒绝错误 prompt/observation、trailing mask、无第二 turn、非法 mask。

实际执行（stage 分别为 real-rollout、real-grpo，各一次，未运行其他 stages）：

```bash
cd /root/autodl-tmp/shop-rl-preflight/code
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
../.venv/bin/python scripts/gpu_preflight.py --stage real-rollout \
  --model-path /root/autodl-tmp/Qwen3-8B \
  --output-root /root/autodl-tmp/shop-rl-preflight/d2b-results \
  --endpoint http://127.0.0.1:5100 --task-id 834368861472
```

执行 helper SHA256：`7aca9bef21dccb3ae21bdb23fec3a371d9d770a2e84a1a41a4aaebc08453e384`。
remote 独立 snapshot HEAD 仍为 D1，加 D2 policy fix；policy/service/protocol 均逐文件 hash 确认一致，
只同步本轮 helper；未更新活跃服务代码。结果仅在 ignored `.cache/d2b-preflight/` 和 remote d2b-results。

`PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/training -q`：
**52 passed, 5 skipped**（D1 opt-in CPU tests 不重复运行）。`git diff --check`、新增 diff secret scan 通过。
结束 health active_sessions=0、原 PID 不变；没有残留本轮 GPU model process。

Formal teacher collector interrupted: **NO**；Teacher API calls: **NONE**；teacher artifacts modified: **NO**。
未开始 Base evaluation、正式 SFT/GRPO。最终 PASS 仅关闭 runtime gate，不冻结实验参数或启动下一阶段。

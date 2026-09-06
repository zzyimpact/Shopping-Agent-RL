# P2 数据审计与 profiling

**状态**：完成；需要用户审阅：是。本文记录 remote 真实数据，并正式采用项目 actual persona train pool。

## 数据来源与 fingerprint

- Host：`rtx-pro-6000-3`；项目：`/root/shopping-agent-rl`。
- Catalog-Fine 实际文件：`/root/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz`。
- 文件 SHA-256：`f51c33217061479f9c95a1068621fcd38e4883ae3d2f6a1627037bea934f2125`。
- 解压后 23,421 条 product/task record；每条有一个 `instructions` 元素。
- record keys：`asin`、`tag`、`domain_zh`、`domain_en_short`、`domain_en_long`、`title`、`sub_title`、`shop_name`、`category`、`full_description`、`attribute`、`customization_options`、`images`、`pricing`、`instructions`、可选 `user_persona`。
- instruction keys：`asin`、`instruction`、`worker_id`、`options`、`instruction_options`、`attributes`、`instruction_simple`。

## Official split 结论

Standard split 直接使用数据中的 `tag`：`eval` 位于 source index `0..1458`，`train` 位于 `1459..23420`，计数正好为 1,459/21,962。upstream `run_envs/run_web_agent_text_env.py` 对 persona 使用固定范围：test=`0..1342`，train=`1459..4781`。因此 remote 实际 persona 计数为 **1,343/3,323**；论文事实仍为 **1,343/3,383**，项目正式采用实际可复现的 3,323。

这不是随机切分结果；脚本只恢复 upstream 明确写出的范围，并验证各 scenario train/test ID 交集为 0。项目不填补缺失的 60 条，也不改用比例切分；这是相对于论文报告数量的 reproducibility deviation。

## Manifest 与 hash

脚本：`scripts/p2_prepare_data.py`；输出目录：`/root/data/shopsim/manifests`。manifest 只保存 `task_id`、scenario、official split、source index、domain、category，不复制 raw catalog。

| manifest | task 数 | task_ids SHA-256 |
|---|---:|---|
| `train_single.json` | 21,962 | `4e4847c31fc4aabaac21fb65a9fa8a7d9e90de372886eb71f915f142bf3c1e03` |
| `test_single.json` | 1,459 | `39213eae61e5aafdcd4819c944f9c72a2e3d198868703ee199ffeb78a1d63e2d` |
| `train_single_persona.json` | 3,323 | `f39b49d7cb546a657d4c15808967d827dab483e313c1100dd991b669295eedb8` |
| `test_single_persona.json` | 1,343 | `5400fccd1319241d940368297d356ba36ab2c21e7d6d8a4698c8d5991195c4f1` |
| `sft_task_manifest_single.json` | 3,000 | `870c399fbb2034f58a0dc4b18b8563d4a8748a35d78a9e89de509db5ac2b969d` |
| `sft_task_manifest_single_persona.json` | 3,000 | `f8ca0e5d3982661d19343df337a9713da16211ac43a361074a2c5d9c35fb6521` |
| `eval_128_single.json` | 128 | `2bddabe94e2367fff581c771b2bee9d602906e5aa07e83be51a8c69624bd9155` |
| `eval_128_single_persona.json` | 128 | `68db6d538a6c06efe0e181957499bef470be73b01ae77ad2eb48174864ba3410` |

SFT sampling 使用 `seed=1`，按 `(domain_zh, category)` 分层，采用 largest-remainder proportional allocation，每层使用 `Random(seed + offset)`，最后按 source index 排序。覆盖率：Single `3000/21962=13.66%`；按当前 upstream persona pool `3000/3323=90.28%`。128 subset 覆盖率分别为 `8.77%` 和 `9.53%`。SFT sample 与各自 test subset 无 overlap。

## Profiling 摘要

以下长度使用已传到 `/root/data/models/Qwen3-8B-tokenizer` 的 tokenizer 文件，未加载模型权重。价格是 raw `pricing` 数值分布；upstream 的随机 `price_upper` 由 goal 逻辑运行时派生，尚未在数据中物化。

| scenario/pool | tasks | instruction tokens (mean/p50/p90/p99/max) | persona tokens (mean/p50/p90/p99/max) | attrs mean/p50 | options mean/p50 | price p50/p90/p99 |
|---|---:|---|---|---:|---:|---:|
| Single train | 21,962 | 49.76/48/68/91/147 | 559.92/560/583/605/641* | 4.54/4 | 1.28/1 | 140/2,199/22,500 |
| Single test | 1,459 | 50.17/49/69/91/148 | 630.60/631/665/695/733* | 4.31/4 | 1.27/1 | 159/2,394/22,312 |
| Single&Pers train | 3,323 | 50.07/49/69/90/114 | 559.92/560/583/605/641 | 4.66/4 | 1.23/1 | 132/2,325/27,425 |
| Single&Pers test | 1,343 | 50.40/49/69/91/148 | 630.60/631/665/695/733 | 4.35/4 | 1.26/1 | 156/2,373/19,999 |

`*` Single pool 的 persona 统计仅表示其中实际带有 `user_persona` 的记录；persona scenario 使用其完整子池。所有已检查字段缺失计数为 0，`asin` 与 manifest task_id 不一致数为 0；category 共 235 个，domain 共 9 个。

## Policy-visible / evaluator-only

原始 record 把两类信息放在同一对象中。后续 policy formatter 必须拆分：

- **Policy-visible**：Single 的 `instructions[0].instruction`；Single&Pers 的 `instructions[0].instruction_simple` 加 `user_persona`；环境 reset/step 返回的页面 observation 与可用 action。
- **Evaluator-only**：`asin`、`category`、`attribute`、`instructions[0].attributes`、`instructions[0].options`/`instruction_options`、`pricing`、目标商品及 reward metadata。商品页面只有在环境 action 后通过 upstream observation 暴露，不把 gold task 字段注入 policy prompt。

## 真实 TRAIN task examples

下面四个例子都来自 train，不是 test。字段值直接摘自 raw record；同一条 underlying record 可分别作为 Single 与 Single&Pers 的 scenario 输入，但 policy-visible 投影不同。

### Single（source index 1459）

```json
{
  "asin": "834368861472",
  "tag": "train",
  "domain_zh": "家居家装",
  "category": "住宅家具›屏风/花窗›挂屏",
  "title": "设计师新款藤编屏风客厅入户对门玄关遮挡木格栅隔断日式实木简约",
  "attribute": ["藤编", "玄关", "遮挡", "日式", "简约"],
  "customization_options": {"颜色分类": ["樟子松（元/平）", "白蜡木（元/平）", "金丝胡桃木（元/平）", "更多材质、颜色请洽客服"]},
  "pricing": [300.0, 950.0],
  "instructions": [{"asin": "834368861472", "instruction": "帮我看看有没有金丝胡桃木材质的日式简约屏风，用在门口做个缓冲区，带点藤条编织的元素，大概950一平米左右的那种。", "worker_id": "1", "options": ["金丝胡桃木（元/平）"], "instruction_options": ["金丝胡桃木（元/平）"], "attributes": ["藤编", "玄关", "遮挡", "日式", "简约"], "instruction_simple": "帮我找一款用于客厅、入户玄关的屏风，大概950元左右一平米。"}],
  "user_persona": "存在于 raw record；Single policy 不读取"
}
```

### Single（source index 1460）

```json
{
  "asin": "920921857638",
  "tag": "train",
  "domain_zh": "家用电器数码",
  "category": "大家电›商用冷链›商用冷柜",
  "title": "美的冰柜商用大容量519升-40度超低温双门冷柜冷冻柜速冻一级能效",
  "attribute": ["大容量", "速冻", "一级能效"],
  "customization_options": {"颜色分类": ["BD/BC-519DKEM", "BD/BC-719DKEM", "BD/BC-419DKEM"]},
  "pricing": [3699.0, 5100.0],
  "instructions": [{"asin": "920921857638", "instruction": "求推荐一款商用大容量冷冻设备，具备快速冷冻功能，支持长时间储存肉类食材，适合用于肉铺场景，能耗表现优秀，价格在5000元左右。", "worker_id": "1", "options": ["BD/BC-719DKEM"], "instruction_options": ["BD/BC-719DKEM"], "attributes": ["大容量", "速冻", "一级能效"], "instruction_simple": "求推荐一款快速冷冻柜，价格在5000元左右。"}],
  "user_persona": "存在于 raw record；Single policy 不读取"
}
```

### Single&Pers（source index 1459）

```yaml
raw_task:
  asin: 834368861472
  tag: train
  instruction_simple: 帮我找一款用于客厅、入户玄关的屏风，大概950元左右一平米。
  user_persona:
    用户ID: "U834368861472, "
    地区信息: {省份: 浙江省, 城市: 杭州市, 区县: 西湖区, 时区: "UTC+8, "}
    人口属性: {性别: 女, 年龄段: "35-44, ", 消费等级: 中, 会员等级: 黄金会员}
    兴趣偏好:
      类目偏好: {家居生活: 高, 服饰鞋包: 中, 美妆护肤: 低}
      商品属性偏好: {价格区间: {最小值: 800, 最大值: 2500}, 风格: [日式, 简约, 侘寂风], 材质: [胡桃木, 藤编, 亚麻], 功能: [隔断, 收纳, 装饰]}
evaluator_only:
  target_asin: 834368861472
  target_attributes: [藤编, 玄关, 遮挡, 日式, 简约]
  target_options: [金丝胡桃木（元/平）]
  pricing: [300.0, 950.0]
```

### Single&Pers（source index 1460）

```yaml
raw_task:
  asin: 920921857638
  tag: train
  instruction_simple: 求推荐一款快速冷冻柜，价格在5000元左右。
  user_persona:
    用户ID: "U920921857638, "
    地区信息: {省份: 广东, 城市: 广州, 区县: 天河区, 时区: "UTC+8, "}
    人口属性: {性别: 男, 年龄段: "35+", 消费等级: 中, 会员等级: 黄金会员}
    兴趣偏好:
      类目偏好: {商用电器: 高, 食品加工设备: 中, 厨房用具: 中, 生鲜食材: 低}
      商品属性偏好: {价格区间: {最小值: 3000, 最大值: 8000}, 风格: [商用, 实用], 材质: [金属, 不锈钢], 功能: [节能, 快速冷冻]}
evaluator_only:
  target_asin: 920921857638
  target_attributes: [大容量, 速冻, 一级能效]
  target_options: [BD/BC-719DKEM]
  pricing: [3699.0, 5100.0]
```

## 数据异常与未完成项

1. 论文报告 3,383，而当前 public/upstream snapshot 为 3,323；项目 actual configuration 已正式冻结为 3,323。
2. 当前 snapshot 没有 `query`、`reason_key`、`instruction_sample` 等 upstream goal 代码会读取的字段；source trace 见 `P2_ENVIRONMENT_SETUP.md`。其中 `reason_key` 可安全为空，`instruction_sample` 有 single_eval 的 persona 投影证据可由 `instruction_simple` 兼容，但 `query` 没有官方 deterministic 生成逻辑，不能自行发明。
3. 官方 ShopSimulator 从未提交 `search_engine`；已按 pinned Princeton WebShop source 恢复 indexing infrastructure，并成功构建 23,421-doc Catalog-Fine index。真实 environment 仍因 `query` data contract 缺失未启动。
4. `zh_core_web_sm` 已验证为 `core_web_sm` 3.8.0；tokenizer profiling 已完成；没有加载 Qwen3-8B 权重。
5. 因 `query` 缺口，本轮没有把 gold-aware smoke trajectory 写入训练数据，也没有宣称 environment/reward end-to-end parity 已通过。

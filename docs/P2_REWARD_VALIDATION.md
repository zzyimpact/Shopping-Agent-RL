# P2 Reward validation

**状态**：完成；需要用户审阅：否。本轮没有修改 upstream reward matching semantics；仅通过项目 compatibility layer 对 public snapshot 的缺失 query 做显式兼容。

## 计划复用的 upstream scorer

- `/root/ShopSimulator/shop_env/web_agent_site/engine/goal.py`：`get_type_reward`、`get_attribute_reward`、`get_option_reward`、`get_reward`。
- `/root/ShopSimulator/get_score.py`：`calculate_metrics` 对 `reward_detail` 的字段映射。
- `/root/ShopSimulator/shop_env/web_agent_site/engine/normalize.py`：`normalize_color`。
- 项目 wrapper：`src/rewards/shopsim_reward.py`；测试：`tests/rewards/test_shopsim_reward.py`。

upstream `get_reward()` 实际实现的是 loose 风格总分：type/category factor ×（attribute 命中 + option 命中 + price indicator）/ 分母；`get_score.py` 再从输出 detail 计算 hard/success/diagnostic averages。项目要求的 `r_loose/r_strict/r_succ/r_finish/r_category/r_attribute/r_option/r_price` 统一接口仍需在 upstream 可运行后做薄包装和 parity 测试。

当前 wrapper 已完成：`r_loose` 直接来自 upstream，`r_strict = r_category * r_attribute * r_option * r_price`，`r_succ` 为 strict 全 1，`r_finish` 表示 terminal purchase，`r_alpha` 实现 alpha endpoint。fuzzy matching 未重写。

## Upstream facts 与 public-data mismatch

- `zh_core_web_sm` 已解决：remote load 成功，`core_web_sm` 版本 `3.8.0`。
- `search_engine` provenance 与 Catalog-Fine index 已解决：23,421 documents indexed，真实 Lucene query 可返回 gold train 商品。
- 当前 data snapshot 缺少 `query`、`instruction_sample` 等 goal 运行字段；source trace 确认没有官方生成逻辑。
- 项目 compatibility layer：缺失 query 时 `query=None`、`query_available=False`，`query_match` 无条件为 `False`；不从 title、category、instruction 或 search action 推断 query。合法 query 存在时保留 upstream equality semantics。
- persona 缺少 `instruction_sample` 时沿用 upstream `single_eval/env.py` 已明确使用的 `instruction_simple` projection；`reason_key` 保持 `None`。

## 本轮验证状态

| 检查 | 结果 |
|---|---|
| deterministic reward unit cases | 通过，12 tests（含 missing-query、empty-string、existing-query compatibility 与 category fallback） |
| upstream scorer vs project wrapper parity | component/aggregate parity 通过 |
| official public output sanity check | 未运行，仓库没有公开 output artifact |
| 8 metrics 统一计算 | wrapper 已实现并通过 unit tests；4 条真实 smoke episode 端到端通过 |

## Environment terminal parity

使用两个 TRAIN task（`834368861472`、`920921857638`），Single 与 Single&Pers 各运行一条真实 search → click → option → Buy Now episode。环境 reward 分别为 `1.0` 与 `0.8`；项目 wrapper 对同一 purchase/goal/options/price 重算完全一致，component detail 一致，且四条 episode 的 `query_match=False`。该 smoke 是 gold-aware correctness test，不是 policy/teacher trajectory，不进入 SFT 或正式 evaluation。

## Project deviation

`query_match=False` 是为保证 public snapshot 可运行和可复现而固定的 compatibility decision，不声称它等同于论文原始运行时行为。它可能影响 `r_type/Rcategory` 及下游 loose/strict reward 的边界 case；正式实验必须报告该 deviation。

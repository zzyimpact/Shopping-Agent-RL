# P2 Reward validation

**状态**：阻塞；需要用户审阅：是（与 upstream snapshot / split 冲突一起审阅）。本轮没有修改 upstream reward semantics，也没有把未验证的近似实现当作正式 scorer。

## 计划复用的 upstream scorer

- `/root/ShopSimulator/shop_env/web_agent_site/engine/goal.py`：`get_type_reward`、`get_attribute_reward`、`get_option_reward`、`get_reward`。
- `/root/ShopSimulator/get_score.py`：`calculate_metrics` 对 `reward_detail` 的字段映射。
- `/root/ShopSimulator/shop_env/web_agent_site/engine/normalize.py`：`normalize_color`。
- 项目 wrapper：`src/rewards/shopsim_reward.py`；测试：`tests/rewards/test_shopsim_reward.py`。

upstream `get_reward()` 实际实现的是 loose 风格总分：type/category factor ×（attribute 命中 + option 命中 + price indicator）/ 分母；`get_score.py` 再从输出 detail 计算 hard/success/diagnostic averages。项目要求的 `r_loose/r_strict/r_succ/r_finish/r_category/r_attribute/r_option/r_price` 统一接口仍需在 upstream 可运行后做薄包装和 parity 测试。

当前 wrapper 已完成：`r_loose` 直接来自 upstream，`r_strict = r_category * r_attribute * r_option * r_price`，`r_succ` 为 strict 全 1，`r_finish` 表示 terminal purchase，`r_alpha` 实现 alpha endpoint。fuzzy matching 未重写。

## 阻塞证据

- `zh_core_web_sm` 已解决：remote load 成功，`core_web_sm` 版本 `3.8.0`。
- `search_engine` provenance 与 Catalog-Fine index 已解决：23,421 documents indexed，真实 Lucene query 可返回 gold train 商品。
- 当前 data snapshot 缺少 `query`、`instruction_sample` 等 goal 运行字段；source trace 确认没有官方生成逻辑。`instruction_sample` 有 persona API 投影证据可兼容为 `instruction_simple`，但 `query` 无法在不发明 reward semantics 的情况下恢复。
- 因 `query` contract 缺失，不能安全启动真实 reset/search/click/buy episode；也不能声称 terminal reward cross-check 完成。

## 本轮验证状态

| 检查 | 结果 |
|---|---|
| deterministic reward unit cases | 通过，8 tests |
| upstream scorer vs project wrapper parity | component/aggregate parity 通过；end-to-end 未完成 |
| official public output sanity check | 未运行，仓库没有公开 output artifact |
| 8 metrics 统一计算 | wrapper 已实现并通过 unit tests；真实环境端到端仍阻塞 |

因此本文件不报告伪造的 terminal episode 数值，也不把 P2 标为通过。当前唯一剩余 blocker 是 upstream data contract 缺少无官方生成逻辑的 `query` 字段；在不发明该字段语义的前提下，无法完成真实 environment terminal reward parity。

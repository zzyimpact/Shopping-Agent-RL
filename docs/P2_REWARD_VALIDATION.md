# P2 Reward validation

**状态**：阻塞；需要用户审阅：是（与 upstream snapshot / split 冲突一起审阅）。本轮没有修改 upstream reward semantics，也没有把未验证的近似实现当作正式 scorer。

## 计划复用的 upstream scorer

- `/root/ShopSimulator/shop_env/web_agent_site/engine/goal.py`：`get_type_reward`、`get_attribute_reward`、`get_option_reward`、`get_reward`。
- `/root/ShopSimulator/get_score.py`：`calculate_metrics` 对 `reward_detail` 的字段映射。
- `/root/ShopSimulator/shop_env/web_agent_site/engine/normalize.py`：`normalize_color`。

upstream `get_reward()` 实际实现的是 loose 风格总分：type/category factor ×（attribute 命中 + option 命中 + price indicator）/ 分母；`get_score.py` 再从输出 detail 计算 hard/success/diagnostic averages。项目要求的 `r_loose/r_strict/r_succ/r_finish/r_category/r_attribute/r_option/r_price` 统一接口仍需在 upstream 可运行后做薄包装和 parity 测试。

## 阻塞证据

- 导入 `goal.py` 会执行 `spacy.load("zh_core_web_sm")`；模型尚未安装，不能导入 scorer。
- 当前 upstream checkout 缺少 `shop_env/search_engine` 与 generated index，无法完成真实 product search、click、option、buy、terminal reward episode。
- 当前 data snapshot 缺少 `query`、`instruction_sample` 等 goal 派生字段；不能通过 dummy tokenizer 或 `spacy.blank("zh")` 绕过，否则会改变官方语义。

## 本轮验证状态

| 检查 | 结果 |
|---|---|
| deterministic reward unit cases | 未运行，依赖 scorer import blocker |
| upstream scorer vs project wrapper parity | 未运行，尚未创建未验证 wrapper |
| official public output sanity check | 未运行，仓库没有公开 output artifact |
| 8 metrics 统一计算 | 未完成 |

因此本文件不报告伪造的 reward 数值，也不把该状态标为通过。待用户确认 persona 数据版本并补齐与此 snapshot 匹配的 upstream data/index/model 依赖后，再继续 reward wrapper、deterministic tests 与 environment smoke。

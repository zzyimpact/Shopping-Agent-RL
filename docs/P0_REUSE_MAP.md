# P0 Upstream Reuse Map

## Upstream 定位与版本

- 本地 checkout：`/Users/zzy/Desktop/Coding/shop-rl/ShopSimulator-main`
- P1 初次检查未发现目录；随后在确认目标路径为空且无远程 GitHub 认证后，将已推送项目快照恢复到 `/root/shopping-agent-rl`，并将 upstream 源码快照恢复到 `/root/ShopSimulator` 供 remote-only CPU validation。
- 本地目录没有 `.git` 元数据，因此无法验证 commit、branch 或 clean 状态；本报告不伪造 commit hash。正式实施前必须补齐可验证的 upstream commit pin。
- 上游 README 声明的公开仓库：`https://github.com/ShopAgent-Team/ShopSimulator`。
- 审计时通过 GitHub API 读取到 `main` 的参考 commit：`51bb26012cee31aea7ac26177c5ffe807026ac07`（2026-09-06 查询）；这是远程参考值，不等同于本地 checkout 的已验证 commit。
- 远程 upstream 路径：`/root/ShopSimulator`；目录内容对应 GitHub main 参考 commit `51bb26012cee31aea7ac26177c5ffe807026ac07`。远程目录没有 upstream `.git` 元数据，因此该 hash 是已审计的上游参考 pin，不等同于本地 checkout 的可验证 Git commit。

## 相关架构（Single / Single&Pers）

上游把商品加载、goal 构造、搜索、页面渲染和 reward 放在 `shop_env/web_agent_site`；`single_eval/env.py` 通过 HTTP `/api/shop_agent` 调用服务，`single_eval/agent.py` 负责单任务对话循环和 JSON 输出。persona 模式由 reset 返回的 `user_persona` 注入 system message，并在后续 observation 中恢复 instruction。Multi-Turn 代码存在，但不属于本项目 v1 场景。

## 复用决策

| 能力 | 决策 | 依据 / 新代码边界 |
|---|---|---|
| Shop environment | REUSE DIRECTLY | `shop_env/web_agent_site/envs/web_agent_text_env.py` 与 `shop_env/shop_env/shop_agent.py` 已提供 reset/step/release 生命周期和 HTTP 服务协议。 |
| Search/index | REUSE DIRECTLY | `engine.init_search_engine()`、`get_top_n_product_from_keywords()` 和上游 Lucene index 约定已定义搜索语义；不重写。 |
| Single-turn environment logic | THIN WRAPPER | 复用 `single_eval/env.ShopEnv` 的请求协议；新增代码只负责 rollout 所需的结构化 trajectory 适配、超步数控制和离线接口。 |
| Single-turn personalization | REUSE DIRECTLY | `ShopEnv.reset()` / `interact()` 的 persona 分支及 `shop_agent._handle_reset_action()` 已处理 persona 注入。 |
| Agent system prompt | REUSE DIRECTLY | 优先采用 `single_eval/configs/standard/qwen3_235b.yaml` 与 `persona/qwen3_235b.yaml` 的中文 prompt；只在后续训练配置中参数化。 |
| Action parser | REUSE DIRECTLY | `engine.parse_action()` 与 `shop_agent._extract_action_from_response()` 已覆盖 `search[...]` / `click[...]` 约定。 |
| Environment observation format | REUSE DIRECTLY | `shop_agent._format_available_actions()` 和 `ShopEnv.interact()` 已形成 observation + 可用 action 文本协议。 |
| Reward/scoring | THIN WRAPPER | `engine.goal.get_reward()` 是官方 loose 风格 scorer，但项目还需 strict endpoint、8 指标统一命名和 deterministic parity tests；包装而非替换商品匹配逻辑。 |
| Evaluation/result serialization | THIN WRAPPER | `single_eval/agent.Agent.save_to_json()` 与 `get_score.calculate_metrics()` 可保持兼容；新增 wrapper 统一 `Rloose/Rstrict/Rsucc/Rfinish/Rcategory/Rattribute/Roption/Rprice` 字段。 |
| Teacher rollout | NEW PROJECT CODE | 上游 `single_eval/agent.py` 绑定 OpenAI 评测循环，没有可恢复、分片、coverage-first 的成功轨迹采集器。 |
| SFT dataset formatting | NEW PROJECT CODE | 上游没有 post-training trajectory schema、成功过滤或 task/demo manifest。 |
| SFT training | NEW PROJECT CODE | README 与顶层代码未提供 SFT trainer；按 DESIGN.md 使用 Qwen3-8B + LoRA/PEFT 的项目模块。 |
| Online GRPO integration | NEW PROJECT CODE | 上游没有 ROLL/GRPO training integration；需把环境 rollout、group reward 和更新器连接起来。 |
| Logging/metrics | NEW PROJECT CODE | 上游结果统计面向静态 JSON，缺少训练 step、group variance、checkpoint 和 run metadata。 |

## 可直接复用的文件 / 函数 / 类

- `shop_env/web_agent_site/envs/web_agent_text_env.py`：`WebAgentTextEnv.reset()`、`step()`、`get_available_actions()`；`SimServer.done()`、`receive()`；商品、session 和页面状态机。
- `shop_env/web_agent_site/engine/engine.py`：`parse_action()`、`init_search_engine()`、`get_top_n_product_from_keywords()`、`get_product_per_page()`、`load_products()`、`map_action_to_html()`。
- `shop_env/web_agent_site/engine/goal.py`：`get_goals()`、`get_existed_goals()`、`get_type_reward()`、`get_attribute_reward()`、`get_option_reward()`、`get_reward()`。
- `shop_env/web_agent_site/engine/normalize.py`：`normalize_color()` 及选项归一化常量。
- `shop_env/shop_env/shop_agent.py`：`_handle_reset_action()`、`_extract_action_from_response()`、`_format_available_actions()`、`_handle_interact_action()`、`shop_agent()`。
- `single_eval/env.py`：`ShopEnv.reset()`、`interact()`、`release()` 的 API client。
- `single_eval/agent.py`：`Agent.reset()`、`act()`、`save_to_json()` 的单任务消息/结果结构；不直接复用其 teacher API 循环作为正式 collector。
- `get_score.py`：`get_finished_task()`、`calculate_metrics()`、`save_metrics()` 的结果文件扫描与兼容性参考。

## 最小新代码职责

1. 固定 upstream 引用、运行时路径和 scenario/task manifest。
2. 将 upstream observation/action/status 映射为可序列化 trajectory，并保留 hidden goal 只供 evaluator 使用。
3. 实现 teacher collection、成功过滤、SFT record 格式和 resume/shard 元数据。
4. 包装官方 scorer，补充 strict reward 与 8 个统一指标的测试。
5. 提供 Qwen3-8B LoRA SFT、在线 GRPO、checkpoint 和最小训练日志接口；本阶段不实现这些模块。

## 风险与歧义

- 当前 upstream checkout 无 commit 元数据，无法满足可复现实验的 pin 要求。
- 本地 checkout 缺少 `shop_env/search_engine` index 目录；只有 `shop_env/data/fine_items_eval_train_all.json.gz`，因此服务启动可行性尚未证明。
- `get_reward()` 只显式返回 loose 风格总分和部分 detail；论文 strict 公式、`Rfinish` 及统一命名需要 P2 parity tests 确认。
- 上游 goal 生成使用随机价格上界；正式 manifest / seed 与 upstream 随机性必须在 P2 冻结。
- `single_eval/env.py` 的 persona 分支会改写返回字典中的 `instruction`；wrapper 必须避免污染 evaluator hidden fields。

本阶段不对 upstream API 做重设计，也不复制 upstream 源码。

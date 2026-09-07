# P3a invalid_action debug audit

> 本报告由 `scripts/debug_p3a_invalid_action.py` 生成；只读审计旧 paid artifacts，不代表正式 P3a 统计。报告不包含 API key、URL 或 provider raw body。

## 1. 症状与审计范围

- scenario: `single`
- audited profiling runs: 5
- audited attempt artifacts/ledger rows: 42
- API requests recorded: 88; retries: 0
- 所有旧数据仅用于 debug/audit；不得进入最终 P3a report、SFT 或 evaluation。

### Run 状态

| run_id | manifest status | SQLite status | attempts |
|---|---|---|---:|
| `p3a-20260907T121022Z-7553f5e0` | `invalidated_by_implementation_bug` | `invalidated_by_implementation_bug` | 5 |
| `p3a-20260907T121831Z-7d8f1747` | `invalidated_by_implementation_bug` | `invalidated_by_implementation_bug` | 1 |
| `p3a-20260907T122340Z-3ab71e2f` | `invalidated_by_implementation_bug` | `invalidated_by_implementation_bug` | 1 |
| `p3a-20260907T122540Z-5b3bcf17` | `invalidated_by_implementation_bug` | `invalidated_by_implementation_bug` | 1 |
| `p3a-20260907T124356Z-920dc25e` | `invalidated_by_implementation_bug` | `invalidated_by_implementation_bug` | 34 |

## 2. 旧标签与 invalid step 分布

要求的终止标签统计：

| status | count |
|---|---:|
| `invalid_action` | 24 |
| `malformed_action` | 0 |
| `infrastructure_interrupted` | 5 |
| `teacher_empty_response` | 0 |
| `terminal_unsuccessful` | 0 |
| `success` | 0 |
| `max_steps` | 0 |
| `profile_unsolved` | 11 |
| `teacher_failure` | 0 |
| `provider_config_error` | 1 |
| `unknown` | 1 |

- 旧 invalid 标签相关 attempts（含后来被 relabel 为 `profile_unsolved` 且 `invalid_action_count>0`）：34
- artifact `malformed_action_count` 累计：1（其中可能已被旧 profiler relabel 为 `profile_unsolved`）
- inferred invalid step（1-based，来自最后一条已执行 action）：`2`: 23, `3`: 6, `4`: 3, `5`: 2
- 上述 step 是 artifact audit inference；不是把旧 label 当作 ground truth。

## 3. API calls / attempt

- total recorded API requests: 88
- API calls per attempt（全部 attempts）: `0`: 6, `1`: 2, `2`: 23, `3`: 6, `4`: 3, `5`: 2
- API calls per old-invalid attempt（含 relabelled profile_unsolved）: `2`: 23, `3`: 6, `4`: 3, `5`: 2
- HTTP status: `200`: 88
- returned model: `gpt-5.6-sol`: 88
- 诊断字段缺失时按 `N/A` 处理；本报告不读取 provider body。

## 4. Action 分布

- all recorded transitions: `Back/navigation`: 1, `Next >`: 1, `product ASIN/numeric click`: 33, `search`: 52
- old-invalid attempts 的最后 action: `Back/navigation`: 1, `product ASIN/numeric click`: 33
- legacy artifact 无 `status` 但 SQLite 有标签：1（按 artifact audit 计为 `unknown`）

## 5. Pre-step legality audit（archived observation heuristic）

- click transitions audited: 35
- argument present in archived pre-step observation: 35
- argument absent in archived pre-step observation: 0
- pre-step unknown: 0
- argument absent after the step (post-state mutation signal): 34
- argument still present after the step: 1
- post-step unknown: 0

### Re-audit classification

| audited_reason | count | interpretation |
|---|---:|---|
| `false_invalid_diagnostic` | 34 | pre-step 可见且执行后页面改变，符合 post-state validity bug |
| `genuinely_illegal_click_candidate` | 0 | pre-step archived observation 未找到参数（仍需环境 replay 定论） |
| `genuine_malformed_action` | 1 | 没有可执行 parsed action；不应归为 validity bug |
| `unresolved` | 0 | artifact 信息不足 |

> `argument present/absent` 是保守的文本审计信号；完整 pre-step legality 仍应由 non-paid remote replay regression 覆盖。

## 6. Teacher visible response 样本

### Sample 1: task `745584327849` (run `p3a-20260907T121022Z-7553f5e0`, status `invalid_action`)

- response step: 2 (terminal); extracted action: `click[745584327849]`
- normalized action: type=`click`, argument=`745584327849`, kind=`product ASIN/numeric click`
- audited reason: `false_invalid_diagnostic` (step=2; argument present pre-step and absent after state-changing step)

```text
Thought: 搜索结果中，745584327849 的商品明确标注“德国正品”“儿童防驼背”“小学生学习写字专用”，价格79元，最符合德国进口、矫姿、小学生专用和预算约80元的条件。先进入该商品详情确认颜色规格。
Action: click[745584327849]
```

### Sample 2: task `689095595880` (run `p3a-20260907T121022Z-7553f5e0`, status `invalid_action`)

- response step: 3 (terminal); extracted action: `click[565485386906]`
- normalized action: type=`click`, argument=`565485386906`, kind=`product ASIN/numeric click`
- audited reason: `false_invalid_diagnostic` (step=3; argument present pre-step and absent after state-changing step)

```text
Thought: 当前结果中，565485386906 是墙面壁挂式化妆品收纳架，定位和美容院使用场景最匹配，价格区间下限较低且最高仅略超预算，值得先进入商品详情核对尺寸、层数和具体规格。
Action: click[565485386906]
```

### Sample 3: task `930026158309` (run `p3a-20260907T124356Z-920dc25e`, status `invalid_action`)

- response step: 2 (terminal); extracted action: `click[930026158309]`
- normalized action: type=`click`, argument=`930026158309`, kind=`product ASIN/numeric click`
- audited reason: `false_invalid_diagnostic` (step=2; argument present pre-step and absent after state-changing step)

```text
Thought: 搜索结果中，930026158309同时明确包含吸盘、枪灰色、壁挂、补光、折叠、放大、双面和带灯等核心特征，且价格区间最高为240元，最符合需求。先查看该商品的规格选项。
Action: click[930026158309]
```

### Sample 4: task `38157104987` (run `p3a-20260907T124356Z-920dc25e`, status `invalid_action`)

- response step: 2 (terminal); extracted action: `click[38157104987]`
- normalized action: type=`click`, argument=`38157104987`, kind=`product ASIN/numeric click`
- audited reason: `false_invalid_diagnostic` (step=2; argument present pre-step and absent after state-changing step)

```text
Thought: 目前结果中，第一款“进口超级保温涂料”同时明确包含进口、纳米陶瓷、耐高温、金属、隔热防水、超薄和节能等核心特征，匹配度最高。虽然价格区间上限超过预算，但需要进入商品页确认18升规格的具体价格是否不超过2700元。
Action: click[38157104987]
```

### Sample 5: task `802546101141` (run `p3a-20260907T124356Z-920dc25e`, status `invalid_action`)

- response step: 2 (terminal); extracted action: `click[802546101141]`
- normalized action: type=`click`, argument=`802546101141`, kind=`product ASIN/numeric click`
- audited reason: `false_invalid_diagnostic` (step=2; argument present pre-step and absent after state-changing step)

```text
Thought: 搜索结果中第一款明确是“仿真秋海棠叶”，黑色、高端、小众，并适合插花与摄影道具；价格区间为13–25元，最接近预算15元。先进入该商品查看规格。
Action: click[802546101141]
```

### Malformed sample: task `802546101141` (run `p3a-20260907T124356Z-920dc25e`)

- extracted action: ``search[黑色 高端 仿真秋海棠叶 单支 摄影道具 插花]``; upstream parse 未产生可执行的 canonical action。

```text
Thought: 需求是小众、高端仿真秋海棠叶，黑色，适合摄影道具或插花，单支，预算约15元。先用包含品类、颜色和单支规格的关键词搜索，以减少同类商品干扰。
Action: `search[黑色 高端 仿真秋海棠叶 单支 摄影道具 插花]`
```

## 7. Parser trace

- visible responses checked: 88
- normalized upstream-extraction mismatches against recorded action: 0
- trailing whitespace/newline differences are normalized before mismatch counting.

## 8. Official protocol comparison / root-cause evidence

1. 当前 `scripts/remote_teacher_env_service.py` 在 `env.step(action)` 后才调用 `get_available_actions()`，并用已经刷新过的 `text_to_clickable` 计算 `action_valid`；这把 PRE-STEP 合法性判断变成了 POST-STEP 判断。
2. 当前 profiler 把 remote 返回的 raw `observation` 直接追加为下一条 user message；官方 Single formatter 还要附加“搜索功能是否可用”和“可点击的按钮”，因此必须以明确的 policy-visible observation contract 对齐。
3. 官方 Persona reset/interaction 使用 `instruction_simple` 投影；应确认 remote/profiler 不把 evaluator/full instruction 泄漏给 teacher。
4. upstream action extraction/parse 是 source of truth；本审计未把不同 Thought 文本视为不同 action。

### 本次审计结论

- artifact 证据支持 post-state validity 假阳性候选：34 条。
- 真正 malformed 候选：1 条。
- 在修复并通过 non-paid pre-step/official-observation regression 前，不应恢复旧 run 或把旧数据纳入统计。

## 9. Fix / regression contract

- 已修复：`action_valid` 针对 pre-step available actions 计算；step 后只返回 next-state available actions。
- 已修复：remote service 返回 canonical `policy_observation`，profiler 只消费该字段；raw observation 仅作 diagnostics。
- 已修复：Single/Persona system prompt 来自 pinned `single_eval` YAML；Persona 使用 `instruction_simple` projection，且不泄漏 evaluator-only fields。
- 已修复：action extraction/parser trace 复用 upstream 的精确 marker/regex，不修复 markdown、大小写或非法格式。
- 已加入 regression：合法 product/option/Buy Now、非法 click、Single/Persona observation formatter、upstream extraction、旧 run invalidation/resume guard。
- 本地非付费结果：targeted protocol/profiler/storage/prompt tests `46 passed`；remote service endpoint tests因本机未安装 Flask runtime 而跳过。
- `py_compile`、shell syntax 和 `git diff --check` 均通过；由于当前网络无法解析 remote host，未声称 live remote replay 已通过。
- 本机未执行 remote live replay：当前网络无法解析 `rtx-pro-6000-3`；remote replay 需由用户网络恢复后手动 smoke。

## 10. Old paid run handling

- 本次运行已执行显式 `--invalidate`；旧 artifacts 保留，run 状态为 `invalidated_by_implementation_bug`，禁止 `--resume`。
- 新修复已 bump immutable `environment_version=task-scoped-v2`、`policy_observation_version=single-eval-policy-v1`、`profiler_protocol_version=p3a-visible-action-v2`；必须创建全新 profiling run。

## 11. Warnings

- 跳过 04: run_manifest.json 缺失
- 跳过 1: run_manifest.json 缺失

生成时间：2026-09-07T15:29:00.436673+00:00

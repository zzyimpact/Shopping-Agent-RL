# Engineering Round B — SFT projection review

Round B 的代码开发与 teacher collection 并行。本轮没有冻结最终数据量、selected
manifest 或 SFT dataset，也没有启动 collector、远程环境请求、模型加载或训练。

## Projection contract

`load_selected_examples(selection_manifest, accepted_root=..., scenario=...)` 只读取 manifest
明确列出的 accepted JSON，不扫描目录、不访问 SQLite。selection 数量可为 3K/4K/6K，
代码没有 6000 gate。`train_sft.py` 另外检查所有 selected task 属于相同 scenario 的
TRAIN manifest；`eval_policy.py` 检查 TEST manifest，可直接传现有 eval_128 或 full test
（Single 1459 / Persona 1343），不重采样。

selection 例子（相对路径以该 run 的 `accepted/` 为根；hash 可填原文件 SHA256）：

```json
{
  "scenario": "single",
  "artifacts": [
    {"path": "c804954c89f74166a87584bdd2eb4535.json",
     "accepted_id": "c804954c89f74166a87584bdd2eb4535"}
  ]
}
```

ID/scenario 必须与源 artifact 相同；若给 `sha256` 则校验。训练 manifest 会额外记录
实际读取的 selected content hash（含顺序），并在 resume 时比较。

Projection 只构造每条 `messages` 的 `role` 与字符串 `content`：system、user observation、
assistant 原样保留。若 artifact 有 `visible_responses`，必须与 assistant turns 一致。
兼容没有新 diagnostics/`visible_responses` 字段的旧 accepted 数据，不重建历史 prompt，
不使用现阶段 reset prompt 替换历史消息。`write_sft_jsonl` 导出 messages 和 accepted/task
ID 供审计；送入 Trainer 的 Dataset **只有 input_ids 和 labels**。

`evaluator_only`、goal、query_match、reward_metrics、terminal_purchase、隐藏 scorer
字段从来不是 projection 输入源。不会把 artifact stringify 后再删字段。用户原始需求、
可见 Persona、搜索页商品 ID、可点击选项本来就属于 policy-visible 内容，必须保留；
不会按隐藏 target ASIN 的值去删合法可见文本。既有 leakage assertion 继续检查 messages。

## Assistant-only labels

```text
system / user / environment observation → -100 (MASK)
assistant visible response             → original token ID (TRAIN)
assistant <|im_end|>                    → TRAIN (learn to end the turn)
role headers / delimiters' whitespace / padding → MASK
```

一条 trajectory 的所有 assistant turns 共同训练，末尾 terminal user observation 保留但
全部 MASK。labels 不预先 shift，由 Transformers causal loss 完成 shift。

本地 `tokenizer_config.json` 的 Qwen3 template 没有 `{% generation %}`。它还可能随最后
user 的位置改变 assistant rendering，所以 **不使用 prefix token-length 差分**。实现：

1. 调用 native `apply_chat_template(..., tokenize=False, add_generation_prompt=False)`。
2. 对本项目 text-only、以 terminal user 结束的完整对话，验证 native rendering 与
   逐条 `role/content` ChatML 拼接完全一致；任何文本重写直接报 PREFLIGHT 错误。
3. fast tokenizer offsets 定位 assistant content + EOS，设置显式 labels；验证 native
   chat-template token IDs 相同、边界不跨 token、每个 assistant EOS 都被训练。
4. 超过 max_length 直接报错，交给数据选择审查，不静默截断或丢掉最后购买动作。

TRL 1.12.0 的 source 支持 pretokenized `input_ids/labels`，默认 collator 保留 labels
并以 -100 padding。这里设置 `dataset_kwargs={"skip_prepare_dataset": true}`，
`packing=false`、`padding_free=false`；`assistant_only_loss=false` **仅表示不再让 TRL
做第二次自动 masking**，实际 loss 已由我们的 assistant-only labels 限定。没有修改
Qwen tokenizer template、teacher prompt 或 action parser。

验证边界：本轮是 fake tokenizer/local tests + Qwen template/TRL source 阅读；当前环境
连 Jinja2/Transformers 都未安装，**未声称真实 tokenizer mask 已通过**。
真实 token offsets/EOS、TRL collator labels 保真属于 **TRAINING-ENV PREFLIGHT REQUIRED**
（可在 CPU 上验证，无需 GPU）。若未来 template 不再满足上述边界，应在独立 training env
解决，禁止静默回退为全 token loss。

## Four real projections

以下恰好四条均来自已完整落盘的 accepted 文件。没有读库或全量扫描；读取前后 source
SHA256 相同。所有角色/turn 都列出；长 system/observation 在文档中缩略，assistant
response 完整展示（文档显示去除行尾空格）。缩略/空格处理只用于文档，formatter 与完整
JSON 副本不修改原文。

每节提供 `.cache/round_b_review/` 下的完整 policy-visible JSON 供本机审查；这些是可删除
的 review 副本，不是正式 SFT freeze，不含 evaluator-only fields。

### Single — task 944750557106

Source: `data/teacher_raw/single/formal-20260908T105913Z-4eec9efd/accepted/c804954c89f74166a87584bdd2eb4535.json`

`accepted_id=c804954c89f74166a87584bdd2eb4535` · 12 messages · 5 assistant turns · accepted=true

Source SHA256: `fac627f182c9d7033fce9f6c50596a74b125b264119b5046bf1f61cf27a3357b`

[完整 projection（本机副本）](../.cache/round_b_review/single-944750557106.json)

1. `system` → **MASK**

```text
你正在进行网上购物。
我会给你一些关于需要做什么的指示。
你必须遵循这些指示。
每一轮，我会给你一个观察结果和一组可用的操作，你需要根据当前状态和指示来选择一个操作进行响应。
如果搜索功能可用，你可以使用搜索操作。
你可以点击可点击按钮列表中的某一个按钮。
操作应遵循以下结构：
search[关键词]
click[值]
如果操作无效，则不执行任何操作。
搜索中的关键词由你自行决定，但点击操作中的值必须是可用操作列表中的值。
一轮只能执行一个动作
注意，在点击购买前，确保至少点击了一个商品的规格。
请记住，你的搜索关键词应该精心设计。
你的响应应使用以下格
…（文档摘录；projection 保留全文）
```

2. `user` → **MASK**

```text
Instruction: 需要购买一款孕后女性坐月子用的披肩，是100%棉绉布做的，春夏季节都合适，夏天能挡挡太阳，还能当个小坎肩，冬天像斗篷似的帮产妇挡住凉气，外观上要有熊和花的元素，价格85元左右。

搜索功能是否可用: True

可点击的按钮: []
```

3. `assistant` → **TRAIN**

```text
Thought: 当前没有可点击商品，先搜索同时包含“月子披肩、100%棉绉布、熊和花、春夏适用”等关键要求的商品。价格约85元也应纳入搜索条件，以便优先找到符合材质、用途和外观要求的选项。该操作将直接帮助定位满足用户要求的商品，但目前尚未确认具体商品及规格，不能购买。
Action: search[孕妇月子披肩 100%棉绉布 熊 花 春夏 防晒 坎肩 85元]
```

4. `user` → **MASK**

```text
Instruction: [SEP] 需要购买一款孕后女性坐月子用的披肩，是100%棉绉布做的，春夏季节都合适，夏天能挡挡太阳，还能当个小坎肩，冬天像斗篷似的帮产妇挡住凉气，外观上要有熊和花的元素，价格85元左右。 [SEP] Back to Search [SEP] Page 1 (Total results: 150) [SEP] Next > [SEP] 944750557106 [SEP] 月子披肩产后春夏多款绉布纯棉月子披肩夏天防晒坎肩产妇防凉斗篷 [SEP] 86.0 [SEP] 934065177116 [SEP] 纯棉月子披肩春夏薄款产后
…（文档摘录；projection 保留全文）
```

5. `assistant` → **TRAIN**

```text
Thought: 当前列表中，944750557106“月子披肩产后春夏多款绉布纯棉月子披肩夏天防晒坎肩产妇防凉斗篷”与用户的用途、季节和功能要求最匹配，价格86元也接近85元。标题明确包含“纯棉”“绉布”“防晒坎肩”和“防凉斗篷”，但尚未确认是否为100%棉以及是否有熊和花元素，因此先进入商品详情核实，并在购买前选择具体规格。
Action: click[944750557106]
```

6. `user` → **MASK**

```text
Instruction: [SEP] 需要购买一款孕后女性坐月子用的披肩，是100%棉绉布做的，春夏季节都合适，夏天能挡挡太阳，还能当个小坎肩，冬天像斗篷似的帮产妇挡住凉气，外观上要有熊和花的元素，价格85元左右。 [SEP] Back to Search [SEP] < Prev [SEP] 尺码 [SEP] 均码 [SEP] 颜色分类 [SEP] zp10绉布五彩心披肩 [SEP] zp4绉布小熊花园披肩 [SEP] zp12绉布黑色波点披肩 [SEP] zp6绉布木槿花披肩 [SEP] zp16绉布圣诞麋鹿披肩 [SEP] zp9绉布紫点披肩 [S
…（文档摘录；projection 保留全文）
```

7. `assistant` → **TRAIN**

```text
Thought: 商品详情显示尺码为均码，价格86元，符合“85元左右”。颜色分类中“zp4绉布小熊花园披肩”同时包含小熊和花园元素，最符合外观要求；商品标题也明确写有绉布、纯棉、春夏、防晒、坎肩和防凉斗篷。虽然页面尚未显示“100%”棉的明确成分比例，但现有选项中该款与用户要求的匹配度最高。先选择包含熊和花元素的具体规格，再选择尺码，之后才能购买。
Action: click[zp4绉布小熊花园披肩]
```

8. `user` → **MASK**

```text
Instruction: [SEP] 需要购买一款孕后女性坐月子用的披肩，是100%棉绉布做的，春夏季节都合适，夏天能挡挡太阳，还能当个小坎肩，冬天像斗篷似的帮产妇挡住凉气，外观上要有熊和花的元素，价格85元左右。 [SEP] Back to Search [SEP] < Prev [SEP] 尺码 [SEP] 均码 [SEP] 颜色分类 [SEP] zp10绉布五彩心披肩 [SEP] zp4绉布小熊花园披肩 [SEP] zp12绉布黑色波点披肩 [SEP] zp6绉布木槿花披肩 [SEP] zp16绉布圣诞麋鹿披肩 [SEP] zp9绉布紫点披肩 [S
…（文档摘录；projection 保留全文）
```

9. `assistant` → **TRAIN**

```text
Thought: 已选择“zp4绉布小熊花园披肩”，该规格同时具备小熊和花元素；商品价格86元，接近预算。购买前还必须选择尺码，目前唯一尺码为“均码”，因此先点击均码完成规格选择。之后即可进入购买流程。
Action: click[均码]
```

10. `user` → **MASK**

```text
Instruction: [SEP] 需要购买一款孕后女性坐月子用的披肩，是100%棉绉布做的，春夏季节都合适，夏天能挡挡太阳，还能当个小坎肩，冬天像斗篷似的帮产妇挡住凉气，外观上要有熊和花的元素，价格85元左右。 [SEP] Back to Search [SEP] < Prev [SEP] 尺码 [SEP] 均码 [SEP] 颜色分类 [SEP] zp10绉布五彩心披肩 [SEP] zp4绉布小熊花园披肩 [SEP] zp12绉布黑色波点披肩 [SEP] zp6绉布木槿花披肩 [SEP] zp16绉布圣诞麋鹿披肩 [SEP] zp9绉布紫点披肩 [S
…（文档摘录；projection 保留全文）
```

11. `assistant` → **TRAIN**

```text
Thought: 已选择“zp4绉布小熊花园披肩”和“均码”，规格已完成；该商品价格86元，接近预算，且标题符合纯棉绉布、春夏防晒、坎肩和防凉斗篷等用途，图案包含小熊与花园元素。页面未明确展示“100%棉”成分，但当前没有更匹配的商品，现可进入购买流程。
Action: click[buy now]
```

12. `user` → **MASK**

```text
Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [SEP] asin [SEP] 944750557106 [SEP] options [SEP] {"\u5c3a\u7801": "\u5747\u7801", "\u989c\u8272\u5206\u7c7b": "zp4\u7ec9\u5e03\u5c0f\u718a\u82b1\u56ed\u62ab\u80a
…（文档摘录；projection 保留全文）
```

### Single — task 941565736582

Source: `data/teacher_raw/single/formal-20260908T105913Z-4eec9efd/accepted/73918a18ab8242e0920c00021e0f94bd.json`

`accepted_id=73918a18ab8242e0920c00021e0f94bd` · 12 messages · 5 assistant turns · accepted=true

Source SHA256: `3a0a970265a478946484d7e30e0280d5a58ec276f2c0b91300a42f04ff7d53f0`

[完整 projection（本机副本）](../.cache/round_b_review/single-941565736582.json)

1. `system` → **MASK**

```text
你正在进行网上购物。
我会给你一些关于需要做什么的指示。
你必须遵循这些指示。
每一轮，我会给你一个观察结果和一组可用的操作，你需要根据当前状态和指示来选择一个操作进行响应。
如果搜索功能可用，你可以使用搜索操作。
你可以点击可点击按钮列表中的某一个按钮。
操作应遵循以下结构：
search[关键词]
click[值]
如果操作无效，则不执行任何操作。
搜索中的关键词由你自行决定，但点击操作中的值必须是可用操作列表中的值。
一轮只能执行一个动作
注意，在点击购买前，确保至少点击了一个商品的规格。
请记住，你的搜索关键词应该精心设计。
你的响应应使用以下格
…（文档摘录；projection 保留全文）
```

2. `user` → **MASK**

```text
Instruction: 需要给家里小狗买个耐咬的牛筋骨头，三根17厘米左右的够它啃一阵子了，重量差不多一公斤，价格在三十五块上下。

搜索功能是否可用: True

可点击的按钮: []
```

3. `assistant` → **TRAIN**

```text
Thought: 当前没有可点击的商品，需要先搜索符合“耐咬、牛筋骨头、约17厘米、三根、约1公斤、35元左右”的商品。先用包含核心规格和价格的关键词搜索，以便找到匹配商品。
Action: search[耐咬牛筋骨头 狗狗 17厘米 3根 1公斤 35元]
```

4. `user` → **MASK**

```text
Instruction: [SEP] 需要给家里小狗买个耐咬的牛筋骨头，三根17厘米左右的够它啃一阵子了，重量差不多一公斤，价格在三十五块上下。 [SEP] Back to Search [SEP] Page 1 (Total results: 150) [SEP] Next > [SEP] 941505969474 [SEP] 狗磨牙棒牛皮骨头中小型犬耐咬胶牛皮卷零食训练奖励洁齿大型犬 [SEP] 28.0 to 182.0 [SEP] 681729532562 [SEP] 月亮狗狗零食牛棒骨宠物磨牙棒大小型犬耐咬清口气补钙洁齿骨头 [SEP] 12.
…（文档摘录；projection 保留全文）
```

5. `assistant` → **TRAIN**

```text
Thought: 当前结果中“狗狗磨牙玩具狗子磨牙棒小狗耐咬牛骨棒大骨头幼犬专用牛筋狗用品”明确包含牛筋、耐咬和小狗适用，价格区间18–47元，也最接近用户要求的35元左右。需要先进入商品详情核对是否有“三根、约17厘米、约1公斤”的规格，确认规格后再考虑购买。
Action: click[941565736582]
```

6. `user` → **MASK**

```text
Instruction: [SEP] 需要给家里小狗买个耐咬的牛筋骨头，三根17厘米左右的够它啃一阵子了，重量差不多一公斤，价格在三十五块上下。 [SEP] Back to Search [SEP] < Prev [SEP] 食品口味 [SEP] 小号（14-16cm）3支装 [SEP] max号牛后腿骨（800g) [SEP] 超大号（350g） [SEP] 中号（16-17cm）3支装 [SEP] 小号（14-16cm） [SEP] 多肉牛腿l号（≥150克） [SEP] 中号（16-17cm） [SEP] 超大号（350g）3支装 [SEP] 大号
…（文档摘录；projection 保留全文）
```

7. `assistant` → **TRAIN**

```text
Thought: 商品规格中有“中号（16-17cm）3支装”，与用户要求的三根、约17厘米高度匹配；另有“1kg”净含量选项，基本覆盖重量要求。先选择尺寸和数量最关键的规格，再选择净含量，确认最终价格是否接近35元后购买。
Action: click[中号（16-17cm）3支装]
```

8. `user` → **MASK**

```text
Instruction: [SEP] 需要给家里小狗买个耐咬的牛筋骨头，三根17厘米左右的够它啃一阵子了，重量差不多一公斤，价格在三十五块上下。 [SEP] Back to Search [SEP] < Prev [SEP] 食品口味 [SEP] 小号（14-16cm）3支装 [SEP] max号牛后腿骨（800g) [SEP] 超大号（350g） [SEP] 中号（16-17cm）3支装 [SEP] 小号（14-16cm） [SEP] 多肉牛腿l号（≥150克） [SEP] 中号（16-17cm） [SEP] 超大号（350g）3支装 [SEP] 大号
…（文档摘录；projection 保留全文）
```

9. `assistant` → **TRAIN**

```text
Thought: 当前已选中“中号（16-17cm）3支装”，价格为34元，符合三根、约17厘米及35元左右的要求；还需选择“1kg”净含量以满足重量要求。选择规格后再购买。
Action: click[1kg]
```

10. `user` → **MASK**

```text
Instruction: [SEP] 需要给家里小狗买个耐咬的牛筋骨头，三根17厘米左右的够它啃一阵子了，重量差不多一公斤，价格在三十五块上下。 [SEP] Back to Search [SEP] < Prev [SEP] 食品口味 [SEP] 小号（14-16cm）3支装 [SEP] max号牛后腿骨（800g) [SEP] 超大号（350g） [SEP] 中号（16-17cm）3支装 [SEP] 小号（14-16cm） [SEP] 多肉牛腿l号（≥150克） [SEP] 中号（16-17cm） [SEP] 超大号（350g）3支装 [SEP] 大号
…（文档摘录；projection 保留全文）
```

11. `assistant` → **TRAIN**

```text
Thought: 已选择“中号（16–17cm）3支装”和“1kg”净含量，完全覆盖三根、约17厘米、约1公斤的规格；当前价格47元，比35元上下高12元，但该商品是当前结果中规格匹配度最高的选项，且标题符合耐咬牛筋骨头需求。规格已确认，可以继续购买。
Action: click[buy now]
```

12. `user` → **MASK**

```text
Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [SEP] asin [SEP] 941565736582 [SEP] options [SEP] {"\u51c0\u542b\u91cf": "1kg", "\u98df\u54c1\u53e3\u5473": "\u4e2d\u53f7\uff0816-17cm\uff093\u652f\u88c5"} [SEP]
…（文档摘录；projection 保留全文）
```

### Single & Persona — task 922242361311

Source: `data/teacher_raw/single_persona/formal-20260909T035508Z-17af7151/accepted/af9100cf888243e3b3c8ed03f1fa8597.json`

`accepted_id=af9100cf888243e3b3c8ed03f1fa8597` · 10 messages · 4 assistant turns · accepted=true

Source SHA256: `7758af42b5ce948cf38f75df325cf7d9ee83d5e49ae801c32c27904b8a5bbdf6`

[完整 projection（本机副本）](../.cache/round_b_review/single_persona-922242361311.json)

1. `system` → **MASK**

```text
你正在进行一次网上购物模拟，目标是从商品库中选购最符合需求的商品。
我会提供他们的目标商品以及个人文档。请注意，商品库中存在大量同类商品，你必须通过合理操作，综合分析用户需求和当前页面信息，最终购买到最符合用户要求的商品。
购物流程采用多轮对话形式，每一轮我会提供当前页面的观察结
…（system 中间省略；下面是原文结尾的可见 Persona 摘录）…
0Z", "注册渠道": "App", "用户ID": "U922242361311, ", "用户标签": ["价格敏感型", "中老年男性", "个护基础需求", "促销关注者"], "行为特征": {"常用设备": "安卓手机", "日均浏览时长": 600, "最近14天关注店铺": [], "最近14天加购商品": [], "最近14天搜索关键词": ["500ml装的一洗上色植物精华染发膏"], "最近14天收藏商品": [], "活跃时间段": [19, 20, 21], "近7天访问次数": 3}}
```

2. `user` → **MASK**

```text
想入手一瓶自然黑色的染发膏，价格在20元左右。

搜索功能是否可用: True

可点击的按钮: []
```

3. `assistant` → **TRAIN**

```text
Thought: 用户需要自然黑色、价格约20元的染发膏，且近期搜索过“500ml装的一洗上色植物精华染发膏”。结合其偏好的霸王品牌、便捷上色功能和价格敏感特征，先搜索自然黑色染发膏及相关规格。
Action: search[霸王 自然黑色 一洗上色 植物精华染发膏 500ml 20元左右]
```

4. `user` → **MASK**

```text
Instruction: [SEP] 想入手一瓶自然黑色的染发膏，价格在20元左右。 [SEP] Back to Search [SEP] Page 1 (Total results: 150) [SEP] Next > [SEP] 931320955132 [SEP] 胖东来严选五贝子染发剂植物一洗正品黑纯自己在家染发膏男士专用 [SEP] 14.0 to 64.0 [SEP] 852283043101 [SEP] 润丝丹染发剂植物纯自然黑天然无刺激泡泡一梳黑正品品牌家染发膏 [SEP] 199.0 [SEP] 922242361311 [SEP] 南
…（文档摘录；projection 保留全文）
```

5. `assistant` → **TRAIN**

```text
Thought: 搜索结果中，南京同仁堂绿金家园款标价19元，接近预算，且明确包含植物、一洗上色和黑发，最符合“自然黑色、便捷上色、约20元”的需求。先查看该商品详情以确认颜色规格。
Action: click[922242361311]
```

6. `user` → **MASK**

```text
Instruction: [SEP] 想入手一瓶自然黑色的染发膏，价格在20元左右。 [SEP] Back to Search [SEP] < Prev [SEP] 颜色分类 [SEP] 葡萄紫  1瓶【500ml】 [SEP] 栗子色 1瓶 【500ml】 [SEP] 栗棕色 1瓶【500ml】 [SEP] 酒红色 1瓶 【500ml】 [SEP] 自然黑 1瓶 【500ml】 [SEP] 黑茶色 1瓶 【500ml】 [SEP] 棕黑色 1瓶【500ml】 [SEP] 南京同仁堂绿金家园一洗彩植然魅一支炫彩染发剂植物纯一洗黑发膏 [SEP] 价格:
…（文档摘录；projection 保留全文）
```

7. `assistant` → **TRAIN**

```text
Thought: 商品价格19元，符合约20元预算；规格中有“自然黑 1瓶【500ml】”，与用户需求完全匹配。先选择自然黑规格，再购买。
Action: click[自然黑 1瓶 【500ml】]
```

8. `user` → **MASK**

```text
Instruction: [SEP] 想入手一瓶自然黑色的染发膏，价格在20元左右。 [SEP] Back to Search [SEP] < Prev [SEP] 颜色分类 [SEP] 葡萄紫  1瓶【500ml】 [SEP] 栗子色 1瓶 【500ml】 [SEP] 栗棕色 1瓶【500ml】 [SEP] 酒红色 1瓶 【500ml】 [SEP] 自然黑 1瓶 【500ml】 [SEP] 黑茶色 1瓶 【500ml】 [SEP] 棕黑色 1瓶【500ml】 [SEP] 南京同仁堂绿金家园一洗彩植然魅一支炫彩染发剂植物纯一洗黑发膏 [SEP] 价格:
…（文档摘录；projection 保留全文）
```

9. `assistant` → **TRAIN**

```text
Thought: 已选择“自然黑 1瓶【500ml】”规格，商品价格为19元，符合用户对自然黑色和20元左右价格的要求，现在点击购买。
Action: click[buy now]
```

10. `user` → **MASK**

```text
Thank you for shopping with us! [SEP] 想入手一瓶自然黑色的染发膏，价格在20元左右。 [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [SEP] asin [SEP] 922242361311 [SEP] options [SEP] {"\u989c\u8272\u5206\u7c7b": "\u81ea\u7136\u9ed1 1\u74f6 \u3010500ml\u3011"} [SEP] attrs [SEP] None
…（文档摘录；projection 保留全文）
```

### Single & Persona — task 740656872880

Source: `data/teacher_raw/single_persona/formal-20260909T035508Z-17af7151/accepted/9e1b02b8a8ce4ea2a2de8b6cb014a622.json`

`accepted_id=9e1b02b8a8ce4ea2a2de8b6cb014a622` · 10 messages · 4 assistant turns · accepted=true

Source SHA256: `f2b79109b48c51c2e1caebb1d70da956e846d6d3a943ea3e6b33dc93a46af05a`

[完整 projection（本机副本）](../.cache/round_b_review/single_persona-740656872880.json)

1. `system` → **MASK**

```text
你正在进行一次网上购物模拟，目标是从商品库中选购最符合需求的商品。
我会提供他们的目标商品以及个人文档。请注意，商品库中存在大量同类商品，你必须通过合理操作，综合分析用户需求和当前页面信息，最终购买到最符合用户要求的商品。
购物流程采用多轮对话形式，每一轮我会提供当前页面的观察结
…（system 中间省略；下面是原文结尾的可见 Persona 摘录）…
": "U740656872880, ", "用户标签": ["羽毛球爱好者", "品质生活追求者", "品牌忠诚度高", "国羽同款装备迷", "运动达人"], "行为特征": {"常用设备": "iOS手机", "日均浏览时长": 1800, "最近14天关注店铺": [], "最近14天加购商品": [], "最近14天搜索关键词": ["大容量方形时尚羽毛球包", "能装下6支羽毛球拍的方包"], "最近14天收藏商品": [], "活跃时间段": [19, 20, 21], "近7天访问次数": 12}}
```

2. `user` → **MASK**

```text
分享一款国家队同款的李宁羽毛球包，要黑粉配色的，价格在600元左右。

搜索功能是否可用: True

可点击的按钮: []
```

3. `assistant` → **TRAIN**

```text
Thought: 用户明确需要李宁品牌、国家队同款、黑粉配色的羽毛球包，预算约600元；先用完整关键属性搜索，以提高匹配精度。
Action: search[李宁 国家队同款 羽毛球包 黑粉 600元]
```

4. `user` → **MASK**

```text
Instruction: [SEP] 分享一款国家队同款的李宁羽毛球包，要黑粉配色的，价格在600元左右。 [SEP] Back to Search [SEP] Page 1 (Total results: 150) [SEP] Next > [SEP] 740656872880 [SEP] 李宁（LI-NING）羽毛球包 时尚大容量羽毛球包6支装方包国家队系 [SEP] 499.0 to 899.0 [SEP] 700992061366 [SEP] yonex尤尼克斯网球羽毛球拍包BA02331WEX运动背包yy白蓝红男女款 [SEP] 680.0 t
…（文档摘录；projection 保留全文）
```

5. `assistant` → **TRAIN**

```text
Thought: 搜索结果中第一款是李宁品牌、明确标注“6支装方包”和“国家队系列”的羽毛球包，且价格区间499–899元，最接近用户约600元及大容量需求。先进入商品详情确认黑粉配色和具体规格。
Action: click[740656872880]
```

6. `user` → **MASK**

```text
Instruction: [SEP] 分享一款国家队同款的李宁羽毛球包，要黑粉配色的，价格在600元左右。 [SEP] Back to Search [SEP] < Prev [SEP] 颜色分类 [SEP] 方包abjt055-1(黑粉色） [SEP] absu399-1（黑色） [SEP] 方包 ablu067-2(白色) [SEP] 方包 ablu067-3(藏蓝色) [SEP] absu399-2（白 | 红色） [SEP] 方包abjt053-2(白蓝色） [SEP] 方包abjt055-2(蓝白色） [SEP] 方包abjt055-3(灰红色
…（文档摘录；projection 保留全文）
```

7. `assistant` → **TRAIN**

```text
Thought: 该商品明确为李宁国家队系列大容量6支装方包，并提供“方包abjt055-1（黑粉色）”规格，符合用户的品牌、配色、容量和功能需求；价格区间覆盖约600元。先选择黑粉色规格，再购买。
Action: click[方包abjt055-1(黑粉色）]
```

8. `user` → **MASK**

```text
Instruction: [SEP] 分享一款国家队同款的李宁羽毛球包，要黑粉配色的，价格在600元左右。 [SEP] Back to Search [SEP] < Prev [SEP] 颜色分类 [SEP] 方包abjt055-1(黑粉色） [SEP] absu399-1（黑色） [SEP] 方包 ablu067-2(白色) [SEP] 方包 ablu067-3(藏蓝色) [SEP] absu399-2（白 | 红色） [SEP] 方包abjt053-2(白蓝色） [SEP] 方包abjt055-2(蓝白色） [SEP] 方包abjt055-3(灰红色
…（文档摘录；projection 保留全文）
```

9. `assistant` → **TRAIN**

```text
Thought: 已选中“方包abjt055-1（黑粉色）”，对应价格599元；商品为李宁国家队系列、时尚大容量6支装方包，完整符合国家队同款、黑粉配色及600元左右预算，现可购买。
Action: click[buy now]
```

10. `user` → **MASK**

```text
Thank you for shopping with us! [SEP] 分享一款国家队同款的李宁羽毛球包，要黑粉配色的，价格在600元左右。 [SEP] None [SEP] (Paste it in your MTurk interface.) [SEP] Purchased [SEP] asin [SEP] 740656872880 [SEP] options [SEP] {"\u989c\u8272\u5206\u7c7b": "\u65b9\u5305abjt055-1(\u9ed1\u7c89\u8272\uff09"} [SEP] att
…（文档摘录；projection 保留全文）
```

## Implemented runtime and use boundaries

- `QwenPolicy.generate(messages) -> str`：Base + optional PEFT adapter；module import 不加载
  weights。默认 evaluator greedy，generation config/seed 记录到 run；历史不静默截断。
- `AgentRollout`：每 trajectory 一个 reset/session，复用现有 prompt/parser/TeacherEnvClient，
  默认 30 actions；malformed/invalid/max-steps 为失败，terminal 使用既有 upstream 分数。
  infrastructure errors 传播并释放 session，不混算为 policy failure。完整对话只在内存中保留
  一个 episode；evaluator 流式写每条完成结果，再聚合 8 metrics。
- `SFTTrainer + PEFT`：BF16、4 epochs、effective batch 32、LR 1e-5；microbatch、accumulation、
  LoRA r/alpha/target_modules 必须显式传入，缺失即拒绝，避免误走 full finetuning。
  当前最小路径限定一个可见 GPU。AdamW、warmup=0、max_grad_norm=1 是 project implementation
  choices；save/log cadence 可以小 config override。
- HF `Trainer.train(resume_from_checkpoint=...)` 恢复 checkpoint；项目只保存
  `run_manifest.json`、`metrics.jsonl`、`checkpoints/checkpoint-N`、`checkpoints/final`、`eval/`。
  `final` 是导出 adapter；训练恢复使用有 optimizer/scheduler/trainer state 的 `checkpoint-N`。
  记录 project commit、audited upstream reference pin、config、seed、input hash、依赖版本与 lineage。
  upstream reference pin 明确不伪装成本地无 `.git` archive 的可验证 commit。

共享参数位于 `configs/training/defaults.yaml`；project seed/30 steps 读取既有 spec。小 override
可以只写 scenario、model_path、output_dir 和实际要变的 `sft`/`generation` 字段。两个 scenario
使用独立 run/output 与 selection manifest。没有 Hydra、registry、teacher ledger 复用或 GRPO。

未来命令形状（**本轮未执行；保持 collection 运行时不要运行这些 eval/train 命令**）：

```bash
# --manifest 接现有固定 128 或 full TEST manifest；所有 checkpoint 复用同一 evaluator。
python3 scripts/eval_policy.py --config path/to/base_single.yaml \
  --manifest path/to/eval_128_single.json --reward strict

# config 必须填入已由 GPU preflight 选定的 LoRA 和 batch 参数。
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_sft.py --config path/to/sft_single.yaml \
  --selection-manifest path/to/selected_single.json \
  --accepted-root path/to/single/run/accepted \
  --train-task-manifest path/to/train_single.json
# 恢复时相同 config/selection，额外 --resume-from-checkpoint runs/<id>/checkpoints/checkpoint-N
```

## Remaining validation

**TRAINING-ENV PREFLIGHT（可 CPU）**：真实 Qwen template/token offsets/EOS labels、tokenizer/model
版本一致、TRL 1.12.0 Dataset/collator construction。依赖 proposal 不变，不在 teacher env 安装：
`torch==2.8.0; transformers>=4.56.2,<5.0; trl==1.12.0; peft>=0.13,<0.20;
accelerate>=1.4,<2; datasets>=4.7,<5`。

**GPU-PREFLIGHT**：Qwen3-8B BF16 load/generation、PEFT attach/trainable parameters、显存/context
成本、microbatch/accumulation/rank/modules、真实单条 ShopEnv episode（另约空闲环境）、SFT
forward/backward、checkpoint save/reload/optimizer resume。不能用本轮 fake tests 代替这些验证。

**等待最终数据**：最终 accepted 数量、selected manifest/hash、final SFT JSONL freeze、正式 SFT。

**Round C**：可继续实现 GRPO glue；本轮没有 GRPOTrainer、rollout_func、env_mask、logprob 或
G grouping。沿用 AgentRollout 的语义，token plumbing/runtime correctness 由 Round C 单独验证。

## Round B validation record

`PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/training -q`
只运行新增 fake/local tests；不运行历史 test suite。实际结果和 commit 见最终报告。

```text
Paid API calls: NONE
Remote ShopEnv calls: NONE
Qwen model loaded: NO
GPU used: NO
Training packages installed/upgraded: NO
Teacher collector changed: NO
Teacher artifact schema changed: NO
Existing accepted artifacts migrated: NO
Current collection interrupted: NO
```

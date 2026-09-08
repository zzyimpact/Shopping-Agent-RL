# P3b Collection Policy v1.1 delta

- Parent：`p3b-v1`
- Parent hash：`b502cd79e2e95e63680126595bd701b3108909169c5fa40ed6572fca6797a59f`
- Version：`p3b-v1.1`
- Hash：`a177c99f190f1f04ca20f079e6a4b42a9fa77cb741194db517ee494157ffca3c`
- 唯一 policy change：`teacher.api_style: responses → chat_completions`

原因：canonical P3a、2/8/10-worker concurrency probes 均实际使用
`chat_completions`，用户也已独立验证该 API path。P3c 因此不再要求 Responses transport
compatibility smoke；后续 tiny paid smoke 只验证 formal collector pipeline。

除 transport adapter 外，`p3b-v1` 的 workers、target、Pass A/B/C、attempt budgets、reserve、
acceptance、diversity、infrastructure、resume 与 logging policy 全部保持不变。历史
`formal_collection_p3b_v1.yaml` 及其 hash 原样保留。

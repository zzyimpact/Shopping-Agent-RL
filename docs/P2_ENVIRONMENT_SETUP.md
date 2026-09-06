# P2 ShopSimulator environment setup

**状态**：搜索基础设施已恢复；真实 environment 仍因 upstream data contract blocker 未完成。

## 缺失与 provenance

ShopSimulator public main（reference `51bb26012cee31aea7ac26177c5ffe807026ac07`）的 tree 和 history 都没有 `shop_env/search_engine`；该路径在 `shop_env/.gitignore` 中被排除。`commits?path=shop_env/search_engine` 返回 0，因此不是可从 ShopSimulator history 恢复的 tracked source。

ShopSimulator 的 `engine.py` 明确依赖 Pyserini `LuceneSearcher`，调用 `search(query, k)`、`doc(hit.docid).raw()`，并期待 raw JSON 的 `id`。与此接口和 indexing flow 兼容的 pinned external source 是 Princeton WebShop：

- repository：`https://github.com/princeton-nlp/WebShop`
- commit：`64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`
- source files：`search_engine/convert_product_file_format.py`、`search_engine/run_indexing.sh`、`search_engine/lucene_searcher.py`
- remote copy：`/root/data/shopsim/webshop-source`
- file SHA-256：`19dc136a6d828bd4016ec616ea0a565378ce7225b1ac60057b1929d67b4d0f03`、`96f0a2323bf6fda0eb0549fd66265c0cb05f988813f8181110761dc43b88ea0c`、`905be2081ffa4c9f93e1b106de4c012ac9677c8043dd264ae92a68dc2a287fa0`

没有改变 tokenization、BM25 参数、field weighting 或 query processing。项目脚本只按外部 converter 的字段投影为 JSONL，因为 Catalog-Fine raw schema 没有旧 WebShop 的 `small_description`；缺失 BulletPoints 按外部 converter 的空文本处理。没有加入 gold target、query 或 reward 字段。

## Catalog-Fine index

构建脚本：`scripts/build_catalog_fine_index.py`。

```bash
cd /root/shopping-agent-rl
/root/miniconda3/bin/python scripts/build_catalog_fine_index.py \
  --source /root/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz \
  --search-root /root/data/shopsim/search_engine
```

结果：

- resources：`/root/data/shopsim/search_engine/resources`，128MB
- index：`/root/data/shopsim/search_engine/indexes`，49MB（51,287,791 bytes）
- documents：23,421
- indexed：23,421；unindexable/empty/skipped/errors 均为 0
- source Catalog SHA-256：`f51c33217061479f9c95a1068621fcd38e4883ae3d2f6a1627037bea934f2125`

真实 query smoke：`金丝胡桃木 屏风` 返回目标 train ASIN `834368861472`，top-20 rank 12；`商用 大容量 冷冻` 返回目标 train ASIN `920921857638`，top-20 rank 7。两次查询均通过 Pyserini Lucene index，不是 gold lookup。

## Goal/data contract trace

- `single_eval/env.py` 的 persona 分支明确把 API reset 的 `instruction` 投影为 `instruction_simple`。
- `shop_agent._handle_reset_action()` 返回 `instruction`、`instruction_simple`、`goal_options`；`reason_key` 只在 persona 结果中转发，可安全保持 `None`。
- `WebAgentTextEnv`/`SimServer` 的旧 WebShop-compatible path 直接读取 `item['query']`、`item['reason_key']`，persona goal 读取 `product['instruction_sample']`。
- 当前 raw Catalog-Fine 只有 `instruction_simple`，没有 `instruction_sample`；persona API 的显式投影证明其是 policy-visible short instruction，但不能反推出 `query`。
- 当前 source 没有 query/instruction_sample preprocessing、conversion 或 runner injection。`load_products()` 先用 `p['query']`，再在 goal/reward 中比较 query；因此 missing query 是真实 runtime contract gap，不是可由现有 deterministic logic 恢复的字段。

## 当前结论

spaCy 已安装并验证为 `core_web_sm 3.8.0`；search/index 已可复现。仍不能启动官方 `WebAgentTextEnv`，因为这样会要求对 `query` 赋予未定义语义。没有修改 upstream、没有使用 `spacy.blank`、没有 gold lookup 替代 search，也没有生成训练数据。

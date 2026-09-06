# P2 ShopSimulator environment setup

**状态**：完成。搜索基础设施、Catalog-Fine compatibility 与 CPU smoke 已验证。

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
- 当前 source 没有 query/instruction_sample preprocessing、conversion 或 runner injection。`load_products()` 先读取 `p['query']`，再在 goal/reward 中比较 query；项目脚本 `/root/shopping-agent-rl/scripts/apply_shopsim_catalog_compat.py` 以显式 `query_available=False` 处理缺失 query，并让 `query_match=False`。

## 当前结论

spaCy 已安装并验证为 `core_web_sm 3.8.0`；search/index 已可复现。compatibility script 幂等运行后，完整 runtime catalog 可构造 Single/Persona goals；使用精简的两个 TRAIN record（完整 index 仍为 23,421 docs）运行 smoke：`python scripts/smoke_shopsim_cpu.py --catalog /root/data/shopsim/smoke-catalog.json --search-root /root/data/shopsim/search_engine --out /root/data/shopsim/smoke.json`。该 smoke 真实经过 search/click/option/Buy Now，未使用 gold lookup 替代 search。另以一个环境实例启动官方 `shop_env/shop_env/pack_api.py` Flask app 的 test client，沿同一 textual action parser 链完成 reset/search/click/option/Buy Now，terminal reward=1.0。没有下载模型、调用 teacher API 或生成训练数据。

## Compatibility patch fingerprints

当前远程 upstream source fingerprint（脚本 `fingerprint_upstream.sh`）：
`sha256=2c8373d721766f0c1c5c98292bc59bbea2f6bbaef139eb20ac00fb09fd5ef67b`。

- `engine.py`：`bb6fdac2b89143c6c69322bc5f6c4ef5b0f3964ff619a36eb2250f97a833352c`
- `goal.py`：`70770af2f7318f58d2f5db064c425fdbca8bead78cdb8b64e81f644cefde2045`
- `utils.py`：`d1412098ec10e37ed5cbe847747094e45b686936bbc038b91ab3046f0fad4631`
- compatibility script：`9fce02e7f8d57ed495a653a1dcb10dda04a0830ff0e0160c17fdd1e14eb35b85`

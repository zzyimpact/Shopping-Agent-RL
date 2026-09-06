#!/usr/bin/env python3
"""为当前 Catalog-Fine public snapshot 应用最小、可审计的兼容补丁。

补丁只处理 archive 路径、外置 search index 以及缺失 goal 字段：不生成 query，
并在 query 不可用时显式关闭 query matching。脚本对已应用状态幂等，对未知源码
状态拒绝修改。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> str:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return "already"
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    return "patched"


def apply(upstream: Path) -> None:
    engine = upstream / "shop_env/web_agent_site/engine/engine.py"
    goal = upstream / "shop_env/web_agent_site/engine/goal.py"
    utils = upstream / "shop_env/web_agent_site/utils.py"
    changes = []
    changes.append(replace_once(
        engine,
        "import random\nfrom collections import defaultdict",
        "import random\nimport gzip\nfrom collections import defaultdict",
    ))
    changes.append(replace_once(
        engine,
        "    with open(filepath) as f:\n        products = json.load(f)",
        "    opener = gzip.open if str(filepath).endswith('.gz') else open\n"
        "    with opener(filepath, 'rt', encoding='utf-8') as f:\n"
        "        products = json.load(f)",
    ))
    changes.append(replace_once(
        engine,
        "    search_engine = LuceneSearcher(os.path.join(BASE_DIR, f'../search_engine/{indexes}'))",
        "    search_root = os.environ.get('SHOPSIM_SEARCH_ROOT', os.path.join(BASE_DIR, '../search_engine'))\n"
        "    search_engine = LuceneSearcher(os.path.join(search_root, indexes))",
    ))
    changes.append(replace_once(
        engine,
        "        products[i]['query'] = p['query'].lower().strip()",
        "        # Catalog-Fine snapshot 没有 query；不推断其值，只记录可用性。\n"
        "        raw_query = p.get('query')\n"
        "        if isinstance(raw_query, str) and raw_query.strip():\n"
        "            products[i]['query'] = raw_query.lower().strip()\n"
        "            products[i]['query_available'] = True\n"
        "        else:\n"
        "            products[i]['query'] = None\n"
        "            products[i]['query_available'] = False",
    ))
    changes.append(replace_once(
        goal,
        "nlp = spacy.load(\"zh_core_web_sm\")",
        "nlp = None\n\n\ndef _get_nlp():\n    global nlp\n    if nlp is None:\n        nlp = spacy.load(\"zh_core_web_sm\")\n    return nlp",
    ))
    changes.append(replace_once(
        goal,
        "    purchased_type_parse = nlp(purchased_type)\n    desired_type_parse = nlp(desired_type)",
        "    parser = _get_nlp()\n    purchased_type_parse = parser(purchased_type)\n    desired_type_parse = parser(desired_type)",
    ))
    changes.append(replace_once(
        goal,
        "            reason_key = item['reason_key']",
        "            reason_key = item.get('reason_key')",
    ))
    changes.append(replace_once(
        goal,
        "                instruction_text = product['instruction_sample']",
        "                # single_eval/env.py 已将 persona API instruction 投影为\n"
        "                # instruction_simple；缺少 sample 时沿用该显式 upstream projection。\n"
        "                instruction_text = product.get(\n"
        "                    'instruction_sample',\n"
        "                    product.get('instruction_simple', product.get('instruction', '')),\n"
        "                )",
    ))
    changes.append(replace_once(
        goal,
        "    query_match = purchased_product['query'] == goal['query']",
        "    purchased_query = purchased_product.get('query')\n"
        "    goal_query = goal.get('query')\n"
        "    query_available = (\n"
        "        isinstance(purchased_query, str)\n"
        "        and bool(purchased_query.strip())\n"
        "        and isinstance(goal_query, str)\n"
        "        and bool(goal_query.strip())\n"
        "        and purchased_product.get('query_available', True)\n"
        "        and goal.get('query_available', True)\n"
        "    )\n"
        "    query_match = bool(query_available and purchased_query == goal_query)",
    ))
    changes.append(replace_once(
        utils,
        "import logging\nimport random",
        "import logging\nimport os\nimport random",
    ))
    changes.append(replace_once(
        utils,
        "DEFAULT_FILE_PATH = join(BASE_DIR, '../data/items_eval_train.json')",
        "DEFAULT_FILE_PATH = os.environ.get(\n"
        "    'SHOPSIM_CATALOG_PATH', join(BASE_DIR, '../data/items_eval_train.json')\n"
        ")",
    ))
    print(f"compatibility patch: {', '.join(changes)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream-root', default='/root/ShopSimulator', type=Path)
    args = parser.parse_args()
    apply(args.upstream_root)


if __name__ == '__main__':
    main()

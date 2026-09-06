#!/usr/bin/env bash
set -euo pipefail

# 对 upstream source/config 做稳定 fingerprint；数据、索引、日志和缓存不计入。
UPSTREAM_ROOT="${UPSTREAM_ROOT:-/root/ShopSimulator}"
cd "${UPSTREAM_ROOT}"
find . -type f \
  -not -path './shop_env/data/*' \
  -not -path './outputs/*' -not -path './*/outputs/*' \
  -not -path './cache/*' -not -path './*/cache/*' \
  -not -path './__pycache__/*' -not -path './*/__pycache__/*' \
  -not -path './shop_env/search_engine/indexes*/*' \
  -not -path './*/indexes/*' -not -name '*.pyc' \
  -not -path './shop_env/web_agent_site/models/logs/*' \
  -print0 | sort -z | xargs -0 sha256sum | sha256sum

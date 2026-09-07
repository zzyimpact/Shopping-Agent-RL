#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ROOT="${TEACHER_DATA_LOCAL_ROOT:-${PROJECT_ROOT}/data/teacher_raw}"
REMOTE_HOST="${TEACHER_DATA_REMOTE_HOST:-rtx-pro-6000-3}"
REMOTE_ROOT="${TEACHER_DATA_REMOTE_ROOT:-/root/data/shopsim/teacher_raw}"
DRY_RUN=()
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=(--dry-run)
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
mkdir -p "${LOCAL_ROOT}"
if ((${#DRY_RUN[@]} == 0)); then
  ssh "${REMOTE_HOST}" mkdir -p "${REMOTE_ROOT}"
else
  # dry-run 不创建远端目录；目录不存在时只报告将执行的动作并成功退出。
  if ! ssh "${REMOTE_HOST}" test -d "${REMOTE_ROOT}"; then
    echo "dry-run: would create remote destination ${REMOTE_HOST}:${REMOTE_ROOT}"
    exit 0
  fi
fi
# 无 --delete；remote 旧文件不会因本地缺失而删除。backup 失败也不修改 local canonical data。
rsync -az --partial "${DRY_RUN[@]}" \
  --exclude '.env.teacher' --exclude '*.tmp' \
  "${LOCAL_ROOT}/" "${REMOTE_HOST}:${REMOTE_ROOT}/"

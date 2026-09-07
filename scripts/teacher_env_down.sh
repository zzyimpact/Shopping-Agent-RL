#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${TEACHER_ENV_REMOTE_HOST:-rtx-pro-6000-3}"
REMOTE_PROJECT="${TEACHER_ENV_REMOTE_PROJECT:-/root/shopping-agent-rl}"
REMOTE_PORT="${TEACHER_ENV_REMOTE_PORT:-5100}"
LOCAL_PORT="${TEACHER_ENV_LOCAL_PORT:-5500}"
STATE_DIR="${PROJECT_ROOT}/.cache/teacher_env"
TUNNEL_PID_FILE="${STATE_DIR}/tunnel.pid"
REMOTE_OWNED_FILE="${STATE_DIR}/remote_service_owned"

if [[ -f "${TUNNEL_PID_FILE}" ]]; then
  pid="$(cat "${TUNNEL_PID_FILE}")"
  if kill -0 "${pid}" 2>/dev/null && ps -p "${pid}" -o command= | grep -Fq "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}"; then
    kill "${pid}"
    wait "${pid}" 2>/dev/null || true
    echo "Stopped tunnel PID ${pid}"
  fi
  rm -f "${TUNNEL_PID_FILE}"
fi

if [[ -f "${REMOTE_OWNED_FILE}" ]]; then
  ssh "${REMOTE_HOST}" bash -s -- "${REMOTE_PROJECT}" <<'REMOTE'
set -euo pipefail
project=$1
pid_file=/root/data/shopsim/teacher_env/service.pid
if [[ -f "${pid_file}" ]]; then
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null && tr '\0' ' ' < "/proc/${pid}/cmdline" | grep -Fq "${project}/scripts/remote_teacher_env_service.py"; then
    kill "${pid}"
    echo "Stopped remote service PID ${pid}"
  fi
  rm -f "${pid_file}"
fi
REMOTE
  rm -f "${REMOTE_OWNED_FILE}"
else
  echo "Remote service was not started by this local up invocation; left unchanged."
fi

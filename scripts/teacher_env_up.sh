#!/usr/bin/env bash
set -euo pipefail

# 本脚本只管理自己记录的 remote service/tunnel PID，不做 broad pgrep。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${TEACHER_ENV_REMOTE_HOST:-rtx-pro-6000-3}"
REMOTE_PROJECT="${TEACHER_ENV_REMOTE_PROJECT:-/root/shopping-agent-rl}"
REMOTE_PORT="${TEACHER_ENV_REMOTE_PORT:-5100}"
LOCAL_PORT="${TEACHER_ENV_LOCAL_PORT:-5500}"
STATE_DIR="${PROJECT_ROOT}/.cache/teacher_env"
TUNNEL_PID_FILE="${STATE_DIR}/tunnel.pid"
TUNNEL_LOG="${STATE_DIR}/tunnel.log"
REMOTE_OWNED_FILE="${STATE_DIR}/remote_service_owned"
mkdir -p "${STATE_DIR}"

# The service is project-owned (not ShopSimulator upstream).  Sync this one
# tracked file so a remote checkout that predates the current branch still
# exposes the policy_context contract required by the profiler.  No catalog,
# model, secret, or generated artifact is transferred.
scp -q "${PROJECT_ROOT}/scripts/remote_teacher_env_service.py" \
  "${REMOTE_HOST}:${REMOTE_PROJECT}/scripts/remote_teacher_env_service.py"

cleanup_failed_start() {
  if [[ -f "${TUNNEL_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${TUNNEL_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && \
       ps -p "${pid}" -o command= | grep -Fq "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}"; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
    rm -f "${TUNNEL_PID_FILE}"
  fi
}
trap cleanup_failed_start ERR

remote_result="$(ssh "${REMOTE_HOST}" bash -s -- "${REMOTE_PROJECT}" "${REMOTE_PORT}" <<'REMOTE'
set -euo pipefail
project=$1
port=$2
state=/root/data/shopsim/teacher_env
pid_file="${state}/service.pid"
mkdir -p "${state}"
if [[ -f "${pid_file}" ]]; then
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null && tr '\0' ' ' < "/proc/${pid}/cmdline" | grep -Fq "remote_teacher_env_service.py"; then
    echo "EXISTING ${pid}"
    exit 0
  fi
  rm -f "${pid_file}"
fi
fingerprint="$(UPSTREAM_ROOT=/root/ShopSimulator "${project}/scripts/fingerprint_upstream.sh" | awk '{print $1}')"
nohup /root/miniconda3/bin/python "${project}/scripts/remote_teacher_env_service.py" \
  --port "${port}" --source-fingerprint "${fingerprint}" \
  > "${state}/service.log" 2>&1 &
pid=$!
echo "${pid}" > "${pid_file}"
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; then
    echo "STARTED ${pid}"
    exit 0
  fi
  sleep 1
done
echo "remote service failed; see ${state}/service.log" >&2
exit 1
REMOTE
)"
echo "Remote service: ${remote_result}"
if grep -Eq '(^|[[:space:]])STARTED [0-9]+' <<<"${remote_result}"; then
  : > "${REMOTE_OWNED_FILE}"
elif ! grep -Eq '(^|[[:space:]])EXISTING [0-9]+' <<<"${remote_result}"; then
  rm -f "${REMOTE_OWNED_FILE}"
fi

if [[ -f "${TUNNEL_PID_FILE}" ]]; then
  tunnel_pid="$(cat "${TUNNEL_PID_FILE}")"
  if kill -0 "${tunnel_pid}" 2>/dev/null && ps -p "${tunnel_pid}" -o command= | grep -Fq "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}"; then
    echo "Tunnel already active: PID ${tunnel_pid}"
  else
    rm -f "${TUNNEL_PID_FILE}"
  fi
fi
if [[ ! -f "${TUNNEL_PID_FILE}" ]]; then
  # nohup 使 tunnel 脱离本次 shell；PID 只记录本脚本启动的 ssh，不使用 broad kill。
  nohup ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "${REMOTE_HOST}" \
    >"${TUNNEL_LOG}" 2>&1 </dev/null &
  tunnel_pid=$!
  echo "${tunnel_pid}" > "${TUNNEL_PID_FILE}"
fi
for _ in $(seq 1 20); do
  if curl -fsS "http://127.0.0.1:${LOCAL_PORT}/health" >/dev/null; then
    echo "Teacher environment ready: http://127.0.0.1:${LOCAL_PORT}"
    exit 0
  fi
  sleep 1
done
echo "SSH tunnel health check failed" >&2
exit 1

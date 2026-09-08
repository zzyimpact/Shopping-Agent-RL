#!/usr/bin/env bash
set -euo pipefail

# 本脚本只管理自己记录的 remote service/tunnel PID，不做 broad pgrep。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${TEACHER_ENV_REMOTE_HOST:-rtx-pro-6000-3}"
REMOTE_PROJECT="${TEACHER_ENV_REMOTE_PROJECT:-/root/shopping-agent-rl}"
REMOTE_PORT="${TEACHER_ENV_REMOTE_PORT:-5100}"
LOCAL_PORT="${TEACHER_ENV_LOCAL_PORT:-5500}"
EXPECTED_ENVIRONMENT_VERSION="task-scoped-v3-multisession"
EXPECTED_POLICY_OBSERVATION_VERSION="single-eval-policy-v1"
EXPECTED_PROFILER_PROTOCOL_VERSION="p3a-visible-action-v2"
STATE_DIR="${PROJECT_ROOT}/.cache/teacher_env"
TUNNEL_PID_FILE="${STATE_DIR}/tunnel.pid"
TUNNEL_LOG="${STATE_DIR}/tunnel.log"
REMOTE_OWNED_FILE="${STATE_DIR}/remote_service_owned"
TUNNEL_CREATED=0
mkdir -p "${STATE_DIR}"

tunnel_matches() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && \
    ps -p "${pid}" -o command= 2>/dev/null | grep -Fq -- \
      "-L 127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}"
}

stop_owned_tunnel() {
  local pid="${1:-}"
  if tunnel_matches "${pid}"; then
    kill "${pid}" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 0.2
    done
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${TUNNEL_PID_FILE}"
}

tunnel_health() {
  curl -fsS --connect-timeout 2 --max-time 5 \
    "http://127.0.0.1:${LOCAL_PORT}/health" >/dev/null 2>&1
}

# The service is project-owned (not ShopSimulator upstream).  Sync this one
# tracked file so a remote checkout that predates the current branch still
# exposes the policy_context contract required by the profiler.  No catalog,
# model, secret, or generated artifact is transferred.
scp -q "${PROJECT_ROOT}/scripts/remote_teacher_env_service.py" \
  "${REMOTE_HOST}:${REMOTE_PROJECT}/scripts/remote_teacher_env_service.py"
scp -q "${PROJECT_ROOT}/src/rollout/protocol.py" \
  "${REMOTE_HOST}:${REMOTE_PROJECT}/src/rollout/protocol.py"

cleanup_failed_start() {
  # Do not tear down a healthy tunnel that predated this invocation if a
  # remote scp/ssh step fails before we create a replacement.
  if [[ "${TUNNEL_CREATED}" == "1" && -f "${TUNNEL_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${TUNNEL_PID_FILE}" 2>/dev/null || true)"
    stop_owned_tunnel "${pid}"
  fi
}
trap cleanup_failed_start ERR

remote_result="$(ssh "${REMOTE_HOST}" bash -s -- "${REMOTE_PROJECT}" "${REMOTE_PORT}" \
  "${EXPECTED_ENVIRONMENT_VERSION}" "${EXPECTED_POLICY_OBSERVATION_VERSION}" \
  "${EXPECTED_PROFILER_PROTOCOL_VERSION}" <<'REMOTE'
set -euo pipefail
project=$1
port=$2
expected_environment_version=$3
expected_policy_observation_version=$4
expected_profiler_protocol_version=$5
state=/root/data/shopsim/teacher_env
pid_file="${state}/service.pid"
mkdir -p "${state}"
service_matches() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && \
    tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -Fq \
      "${project}/scripts/remote_teacher_env_service.py"
}
stop_service() {
  local pid="$1"
  if service_matches "${pid}"; then
    kill "${pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 0.2
    done
    kill -9 "${pid}" 2>/dev/null || true
  fi
}
health_matches() {
  curl -fsS --connect-timeout 2 --max-time 5 "http://127.0.0.1:${port}/health" \
    | /root/miniconda3/bin/python -c 'import json, sys; payload=json.load(sys.stdin); expected={"environment_version": sys.argv[1], "policy_observation_version": sys.argv[2], "profiler_protocol_version": sys.argv[3]}; sys.exit(0 if all(payload.get(k) == v for k, v in expected.items()) else 1)' \
      "${expected_environment_version}" "${expected_policy_observation_version}" \
      "${expected_profiler_protocol_version}"
}
if [[ -f "${pid_file}" ]]; then
  pid="$(cat "${pid_file}")"
  if service_matches "${pid}"; then
    if health_matches >/dev/null 2>&1; then
      echo "EXISTING ${pid}"
      exit 0
    fi
    echo "STALE_OR_UNHEALTHY ${pid}; restarting"
    stop_service "${pid}"
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
  if tunnel_matches "${tunnel_pid}"; then
    if tunnel_health; then
      echo "Tunnel already active: PID ${tunnel_pid}"
    else
      # The SSH process can survive while its forwarding channel is stale
      # after a laptop/network transition.  Recycle only this exact tunnel.
      echo "Existing tunnel PID ${tunnel_pid} is unhealthy; restarting it."
      stop_owned_tunnel "${tunnel_pid}"
      TUNNEL_CREATED=1
    fi
  else
    rm -f "${TUNNEL_PID_FILE}"
  fi
fi
if [[ ! -f "${TUNNEL_PID_FILE}" ]]; then
  # ssh 自身 daemonize，避免父 terminal/shell 退出时转发被回收。PID 只
  # 记录精确匹配这条 forward 的 ssh，不使用 broad kill。
  ssh -f -N -o ExitOnForwardFailure=yes -o ConnectTimeout=15 \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "${REMOTE_HOST}" \
    >"${TUNNEL_LOG}" 2>&1 </dev/null
  tunnel_pid=""
  for _ in $(seq 1 10); do
    tunnel_pid="$(ps -axo pid=,command= | awk -v needle="-L 127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" '$0 ~ needle {print $1; exit}')"
    [[ -n "${tunnel_pid}" ]] && break
    sleep 0.2
  done
  if [[ -z "${tunnel_pid}" ]]; then
    echo "SSH tunnel started but exact PID could not be resolved" >&2
    exit 1
  fi
  echo "${tunnel_pid}" > "${TUNNEL_PID_FILE}"
  TUNNEL_CREATED=1
fi
for _ in $(seq 1 20); do
  if tunnel_matches "$(cat "${TUNNEL_PID_FILE}" 2>/dev/null || true)" && tunnel_health; then
    echo "Teacher environment ready: http://127.0.0.1:${LOCAL_PORT}"
    exit 0
  fi
  sleep 1
done
echo "SSH tunnel health check failed" >&2
exit 1

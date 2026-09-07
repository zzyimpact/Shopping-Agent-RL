#!/usr/bin/env python3
"""Audit old P3a profiling attempts without contacting a teacher API.

This utility is intentionally read-only by default.  It reads the JSON attempt
artifacts and the SQLite ledger under ``data/teacher_profile`` and writes a
compact, secret-free debug report.  ``--invalidate`` is an explicit opt-in that
marks selected profiling runs as invalidated while preserving every artifact.

The audit is deliberately conservative: legality inferred from an archived
observation is labelled a *heuristic*.  A definitive pre-step replay can be
performed separately against the non-paid ShopSimulator smoke environment.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any, Iterable, Mapping
import uuid


INVALIDATION_CODE_VERSION = "p3a-invalid-action-debug-v1"
DEFAULT_OUTPUT = Path("docs/P3A_INVALID_ACTION_DEBUG.md")


@dataclass
class Attempt:
    run_id: str
    scenario: str
    run_path: Path
    artifact_path: Path | None
    record: dict[str, Any]
    ledger_status: str | None = None
    legacy: bool = False

    @property
    def status(self) -> str:
        # A legacy JSON artifact without a status is intentionally reported as
        # ``unknown`` even when an old ledger row contains a label.  This keeps
        # artifact-level audit counts honest and makes the discrepancy visible;
        # synthesized ledger-only placeholders still carry their ledger status.
        if self.legacy:
            return "unknown"
        value = self.record.get("status") or self.ledger_status or "unknown"
        return str(value)

    @property
    def task_id(self) -> str:
        return str(self.record.get("task_id") or "N/A")


@dataclass
class Run:
    run_id: str
    scenario: str
    path: Path
    manifest: dict[str, Any]
    attempts: list[Attempt] = field(default_factory=list)
    ledger_state: dict[str, str] = field(default_factory=dict)
    db_error: str | None = None


ACTION_RE = re.compile(r"(.+)\[(.+)\]", re.S)
VISIBLE_ACTION_RE = re.compile(r"(?:^|\n)\s*Action\s*:\s*(.*)", re.I)
PRODUCT_ID_RE = re.compile(r"^\d{8,16}$")
URL_RE = re.compile(r"https?://[^\s)]+", re.I)
SECRET_RE = re.compile(r"(?i)(?:sk|key|token|secret)[-_A-Za-z0-9]*\s*[:=]\s*[^\s,;]+")
BEARER_RE = re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(text: str) -> str:
    """Remove accidental transport/secret-looking strings from report text."""

    text = URL_RE.sub("<redacted-url>", text)
    text = BEARER_RE.sub("Bearer <redacted>", text)
    def replace_secret(match: re.Match[str]) -> str:
        token = match.group(0)
        separator = ":" if ":" in token else "="
        prefix = token.split(separator, 1)[0].rstrip()
        return prefix + separator + "<redacted>"

    text = SECRET_RE.sub(replace_secret, text)
    # Common API-key prefixes can occur without a key= label.
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "<redacted-key>", text)
    return text


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root is not an object")
    return value


def _safe_db_rows(path: Path) -> tuple[dict[str, str], dict[str, tuple[Any, ...]], str | None]:
    """Read the small ledger defensively; never mutate the database."""

    state: dict[str, str] = {}
    rows: dict[str, tuple[Any, ...]] = {}
    if not path.exists():
        return state, rows, "state.sqlite missing"
    db: sqlite3.Connection | None = None
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        for key, value in db.execute("SELECT key, value FROM run_state"):
            state[str(key)] = str(value)
        for row in db.execute(
            "SELECT attempt_id, task_id, scenario, status, artifact_path, started_at, "
            "finished_at, teacher_attempt FROM attempts ORDER BY rowid"
        ):
            rows[str(row[0])] = row
    except Exception as exc:  # audit should still report readable artifacts
        return state, rows, f"{type(exc).__name__}: {exc}"
    finally:
        if db is not None:
            db.close()
    return state, rows, None


def _placeholder_from_row(run: Run, row: tuple[Any, ...]) -> Attempt:
    attempt_id, task_id, scenario, status, artifact_path, started_at, finished_at, teacher_attempt = row
    return Attempt(
        run_id=run.run_id,
        scenario=run.scenario,
        run_path=run.path,
        artifact_path=(run.path / str(artifact_path)) if artifact_path else None,
        record={
            "attempt_id": str(attempt_id), "task_id": str(task_id),
            "scenario": str(scenario or run.scenario), "status": str(status),
            "started_at": started_at, "finished_at": finished_at,
            "teacher_attempt": teacher_attempt, "actions": [],
            "observations": [], "visible_responses": [], "api_diagnostics": [],
        },
        ledger_status=str(status),
    )


def load_runs(data_root: Path, scenario: str, run_id: str | None = None) -> tuple[list[Run], list[str]]:
    scenario_root = data_root / scenario
    warnings: list[str] = []
    if run_id:
        directories = [scenario_root / run_id]
    elif scenario_root.exists():
        directories = sorted(path for path in scenario_root.iterdir() if path.is_dir())
    else:
        return [], [f"scenario directory missing: {scenario_root}"]

    runs: list[Run] = []
    for path in directories:
        manifest_path = path / "run_manifest.json"
        if not manifest_path.exists():
            warnings.append(f"跳过 {path.name}: run_manifest.json 缺失")
            continue
        try:
            manifest = _read_json(manifest_path)
        except Exception as exc:
            warnings.append(f"跳过 {path.name}: manifest 读取失败 ({type(exc).__name__})")
            continue
        if manifest.get("purpose") != "p3a_profiling":
            warnings.append(f"跳过 {path.name}: purpose 不是 p3a_profiling")
            continue
        if str(manifest.get("scenario", scenario)) != scenario:
            warnings.append(f"跳过 {path.name}: scenario 不匹配")
            continue
        actual_id = str(manifest.get("run_id", path.name))
        run = Run(actual_id, scenario, path, manifest)
        state, db_rows, db_error = _safe_db_rows(path / "state.sqlite")
        run.ledger_state, run.db_error = state, db_error
        by_id: dict[str, Attempt] = {}
        attempts_dir = path / "attempts"
        if attempts_dir.exists():
            for artifact in sorted(attempts_dir.glob("*.json")):
                try:
                    record = _read_json(artifact)
                except Exception as exc:
                    warnings.append(f"{path.name}/{artifact.name}: JSON 读取失败 ({type(exc).__name__})")
                    continue
                aid = str(record.get("attempt_id") or artifact.stem)
                ledger_row = db_rows.get(aid)
                ledger_status = str(ledger_row[3]) if ledger_row else None
                item = Attempt(actual_id, scenario, path, artifact, record,
                               ledger_status=ledger_status,
                               legacy=not bool(record.get("status")))
                by_id[aid] = item
                run.attempts.append(item)
        # Include ledger-only rows.  This makes interrupted/reset attempts visible
        # even if a process died before publishing an artifact.
        for aid, row in db_rows.items():
            if aid not in by_id:
                run.attempts.append(_placeholder_from_row(run, row))
        run.attempts.sort(key=lambda item: (
            str(item.record.get("started_at") or ""),
            str(item.record.get("attempt_id") or ""),
        ))
        runs.append(run)
    if run_id and not runs:
        warnings.append(f"未找到可审计 run: {run_id}")
    return runs, warnings


def upstream_extract(response: str) -> str:
    """Mirror upstream ``shop_agent._extract_action_from_response``."""

    normalized = str(response).replace("\\n", "\n")
    marker = "\nAction: "
    if marker in normalized:
        return normalized.split(marker)[1]
    return normalized


def parse_upstream(action: str) -> tuple[str | None, str | None]:
    """Mirror the upstream ``parse_action`` shape, with report-friendly trim."""

    match = ACTION_RE.match(str(action))
    if match is None:
        return str(action).strip() or None, None
    return match.group(1).strip().lower(), match.group(2).strip()


def normalized_action(value: Any) -> tuple[str | None, str | None]:
    name, arg = parse_upstream(str(value))
    return (name.lower() if isinstance(name, str) else None,
            arg.strip().lower() if isinstance(arg, str) else None)


def action_from_visible(response: str) -> tuple[str | None, str | None, str]:
    extracted = upstream_extract(response)
    name, arg = parse_upstream(extracted.strip())
    return name, arg, extracted


def visible_action_line(response: str) -> str | None:
    match = VISIBLE_ACTION_RE.search(str(response).replace("\\n", "\n"))
    return match.group(1).strip() if match else None


def action_kind(name: str | None, arg: str | None) -> str:
    if name == "search":
        return "search"
    if name != "click" or not arg:
        return "malformed/other"
    normalized = arg.strip().lower()
    if PRODUCT_ID_RE.fullmatch(normalized):
        return "product ASIN/numeric click"
    if normalized in {"buy now", "buy", "购买", "立即购买"}:
        return "Buy Now"
    if normalized in {"next >", "next", "下一页", ">"}:
        return "Next >"
    if normalized in {"back to search", "< prev", "prev", "返回搜索", "返回"}:
        return "Back/navigation"
    if normalized in {"description", "features", "reviews"}:
        return "product section"
    return "option/other click"


def contains_argument(argument: str | None, observation: Any) -> bool | None:
    if not isinstance(observation, str) or not argument:
        return None
    needle = argument.strip().casefold()
    if not needle:
        return None
    haystack = observation.casefold()
    # Very short labels are too noisy for a useful substring audit.
    if len(needle) <= 1:
        return bool(re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack))
    return needle in haystack


def observation_list(attempt: Attempt) -> list[str]:
    value = attempt.record.get("observations", [])
    return [str(item) for item in value] if isinstance(value, list) else []


def actions_list(attempt: Attempt) -> list[str]:
    value = attempt.record.get("actions", [])
    return [str(item) for item in value] if isinstance(value, list) else []


def visible_list(attempt: Attempt) -> list[str]:
    value = attempt.record.get("visible_responses", [])
    return [str(item) for item in value] if isinstance(value, list) else []


def api_list(attempt: Attempt) -> list[Mapping[str, Any]]:
    value = attempt.record.get("api_diagnostics", [])
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def invalid_candidate(attempt: Attempt) -> tuple[str, int | None, str | None]:
    """Classify an old invalid label using archived pre/post observation evidence."""

    actions = actions_list(attempt)
    observations = observation_list(attempt)
    if not actions:
        if visible_list(attempt):
            return "genuine_malformed_action", None, "no parsed action"
        return "unresolved", None, "no action artifact"
    # Only old labels that terminated on invalid_action (or were relabelled
    # profile_unsolved after an invalid action) are eligible for this audit.
    old_invalid = attempt.status == "invalid_action" or int(attempt.record.get("invalid_action_count", 0) or 0) > 0
    if not old_invalid:
        return "not-invalid-labelled", None, None
    idx = len(actions) - 1
    name, arg = normalized_action(actions[idx])
    step = idx + 1
    if name != "click" or not arg:
        return "unresolved", step, "last action is not a click"
    pre = contains_argument(arg, observations[idx] if idx < len(observations) else None)
    post = contains_argument(arg, observations[idx + 1] if idx + 1 < len(observations) else None)
    if pre is True and post is False:
        return "false_invalid_diagnostic", step, "argument present pre-step and absent after state-changing step"
    if pre is False:
        return "genuinely_illegal_click_candidate", step, "argument absent from archived pre-step observation"
    return "unresolved", step, f"pre_contains={pre}, post_contains={post}"


def parser_trace(attempt: Attempt) -> list[dict[str, Any]]:
    actions = actions_list(attempt)
    responses = visible_list(attempt)
    traces: list[dict[str, Any]] = []
    for idx, response in enumerate(responses):
        name, arg, extracted = action_from_visible(response)
        recorded = actions[idx] if idx < len(actions) else None
        rec_name, rec_arg = normalized_action(recorded) if recorded is not None else (None, None)
        parsed_norm = (name, arg.casefold() if isinstance(arg, str) else None)
        recorded_norm = (rec_name, rec_arg)
        equal = recorded is not None and parsed_norm == recorded_norm
        traces.append({
            "step": idx + 1, "response": response, "extracted": extracted,
            "parsed_name": name, "parsed_arg": arg,
            "recorded": recorded, "equal": equal,
            "has_action_line": visible_action_line(response) is not None,
        })
    return traces


def _status_counter(runs: Iterable[Run]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for run in runs:
        counter.update(attempt.status for attempt in run.attempts)
    return counter


def _fmt_counter(counter: Mapping[Any, int], *, code: bool = True) -> str:
    if not counter:
        return "N/A"
    parts = []
    for key, value in sorted(counter.items(), key=lambda item: str(item[0])):
        label = f"`{key}`" if code else str(key)
        parts.append(f"{label}: {value}")
    return ", ".join(parts)


def _fmt_num(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _sample_attempts(runs: list[Run], minimum: int = 5) -> list[tuple[Attempt, int, str]]:
    candidates: list[tuple[Attempt, int, str]] = []
    seen_tasks: set[str] = set()
    # Prefer attempts with an actual visible response and an old invalid label,
    # then fill with any task.  Distinct task IDs keep the report useful.
    ordered = [a for run in runs for a in run.attempts]
    ordered.sort(key=lambda a: (a.task_id in {"N/A", ""}, a.status != "invalid_action",
                                str(a.record.get("started_at") or "")))
    for attempt in ordered:
        responses = visible_list(attempt)
        if not responses or attempt.task_id in seen_tasks:
            continue
        traces = parser_trace(attempt)
        if not traces:
            continue
        # Show the terminal parsed response when available; this is the most
        # informative view for invalid-action diagnosis.
        trace_index = min(len(traces), len(actions_list(attempt))) - 1
        if trace_index < 0:
            trace_index = 0
        candidates.append((attempt, trace_index, "terminal" if trace_index else "first"))
        seen_tasks.add(attempt.task_id)
        if len(candidates) >= minimum:
            break
    return candidates


def _run_audit(runs: list[Run]) -> dict[str, Any]:
    statuses = _status_counter(runs)
    requested_statuses = (
        "invalid_action", "malformed_action", "infrastructure_interrupted",
        "teacher_empty_response", "terminal_unsuccessful", "success", "max_steps",
        "profile_unsolved", "teacher_failure", "provider_config_error", "unknown",
    )
    for status in requested_statuses:
        statuses.setdefault(status, 0)

    invalid_steps: Counter[int | str] = Counter()
    calls: Counter[int] = Counter()
    invalid_calls: Counter[int] = Counter()
    action_kinds: Counter[str] = Counter()
    final_invalid_kinds: Counter[str] = Counter()
    parser_mismatches: list[tuple[Attempt, int, dict[str, Any]]] = []
    status_discrepancies: list[Attempt] = []
    parser_checked = 0
    pre_true = pre_false = pre_unknown = 0
    post_missing = post_present = post_unknown = 0
    classification: Counter[str] = Counter()
    api_statuses: Counter[str] = Counter()
    api_models: Counter[str] = Counter()
    retry_total = 0
    click_transitions = 0
    old_invalid_attempts = 0
    malformed_visible = 0
    malformed_field_count = 0
    for run in runs:
        for attempt in run.attempts:
            if attempt.legacy and attempt.ledger_status and attempt.ledger_status != "unknown":
                status_discrepancies.append(attempt)
            actions = actions_list(attempt)
            observations = observation_list(attempt)
            calls[len(api_list(attempt))] += 1
            for diag in api_list(attempt):
                if diag.get("http_status") is not None:
                    api_statuses[str(diag.get("http_status"))] += 1
                if diag.get("model"):
                    api_models[str(diag.get("model"))] += 1
                try:
                    retry_total += int(diag.get("retries", 0) or 0)
                except (TypeError, ValueError):
                    pass
            candidate, step, _reason = invalid_candidate(attempt)
            old_invalid = attempt.status == "invalid_action" or int(attempt.record.get("invalid_action_count", 0) or 0) > 0
            is_malformed = (
                attempt.status == "malformed_action"
                or attempt.record.get("termination_reason") == "malformed_action"
                or int(attempt.record.get("malformed_action_count", 0) or 0) > 0
            )
            # Classification is only meaningful for the old invalid-labelled
            # population (plus an explicitly malformed visible response).  Do
            # not turn unrelated infrastructure/config stops into "unresolved".
            if old_invalid or is_malformed:
                classification[candidate] += 1
            if old_invalid:
                old_invalid_attempts += 1
                invalid_calls[len(api_list(attempt))] += 1
                if step is not None:
                    invalid_steps[step] += 1
                    name, arg = normalized_action(actions[step - 1])
                    final_invalid_kinds[action_kind(name, arg)] += 1
            if attempt.status == "malformed_action" or (
                attempt.record.get("termination_reason") == "malformed_action"
            ):
                malformed_visible += 1
            try:
                malformed_field_count += int(attempt.record.get("malformed_action_count", 0) or 0)
            except (TypeError, ValueError):
                pass
            for idx, action in enumerate(actions):
                name, arg = normalized_action(action)
                kind = action_kind(name, arg)
                action_kinds[kind] += 1
                if name != "click" or not arg:
                    continue
                click_transitions += 1
                pre = contains_argument(arg, observations[idx] if idx < len(observations) else None)
                post = contains_argument(arg, observations[idx + 1] if idx + 1 < len(observations) else None)
                if pre is True:
                    pre_true += 1
                elif pre is False:
                    pre_false += 1
                else:
                    pre_unknown += 1
                if post is True:
                    post_present += 1
                elif post is False:
                    post_missing += 1
                else:
                    post_unknown += 1
            for trace_index, trace in enumerate(parser_trace(attempt)):
                parser_checked += 1
                if trace["recorded"] is not None and not trace["equal"]:
                    parser_mismatches.append((attempt, trace_index, trace))
    total_attempts = sum(len(run.attempts) for run in runs)
    total_api = sum(calls.values())  # calls counter counts attempts, not requests
    total_requests = sum(len(api_list(a)) for run in runs for a in run.attempts)
    return {
        "runs": runs, "statuses": statuses, "invalid_steps": invalid_steps,
        "calls": calls, "invalid_calls": invalid_calls, "action_kinds": action_kinds,
        "final_invalid_kinds": final_invalid_kinds,
        "classification": classification, "api_statuses": api_statuses,
        "api_models": api_models, "retry_total": retry_total,
        "click_transitions": click_transitions, "pre_true": pre_true,
        "pre_false": pre_false, "pre_unknown": pre_unknown,
        "post_missing": post_missing, "post_present": post_present,
        "post_unknown": post_unknown, "old_invalid_attempts": old_invalid_attempts,
        "malformed_visible": malformed_visible, "malformed_field_count": malformed_field_count,
        "parser_checked": parser_checked, "parser_mismatches": parser_mismatches,
        "status_discrepancies": status_discrepancies,
        "total_attempts": total_attempts, "total_api": total_api,
        "total_requests": total_requests,
    }


def _sample_text(value: str, limit: int = 900) -> str:
    value = _redact(value).replace("```", "'''" ).strip()
    if len(value) > limit:
        value = value[:limit] + "…"
    return value


def _report(audit: Mapping[str, Any], warnings: list[str], *, invalidated: bool = False) -> str:
    runs: list[Run] = list(audit["runs"])
    statuses: Mapping[str, int] = audit["statuses"]
    lines: list[str] = []
    lines += [
        "# P3a invalid_action debug audit",
        "",
        "> 本报告由 `scripts/debug_p3a_invalid_action.py` 生成；只读审计旧 paid artifacts，"
        "不代表正式 P3a 统计。报告不包含 API key、URL 或 provider raw body。",
        "",
        "## 1. 症状与审计范围",
        "",
        f"- scenario: `{runs[0].scenario if runs else 'N/A'}`",
        f"- audited profiling runs: {len(runs)}",
        f"- audited attempt artifacts/ledger rows: {audit['total_attempts']}",
        f"- API requests recorded: {audit['total_requests']}; retries: {audit['retry_total']}",
        "- 所有旧数据仅用于 debug/audit；不得进入最终 P3a report、SFT 或 evaluation。",
        "",
        "### Run 状态",
        "",
        "| run_id | manifest status | SQLite status | attempts |",
        "|---|---|---|---:|",
    ]
    for run in runs:
        lines.append(
            f"| `{run.run_id}` | `{run.manifest.get('status', '未设置')}` | "
            f"`{run.ledger_state.get('status', '未设置')}` | {len(run.attempts)} |"
        )
    if not runs:
        lines.append("| N/A | N/A | N/A | 0 |")
    lines += [
        "",
        "## 2. 旧标签与 invalid step 分布",
        "",
        "要求的终止标签统计：",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for status in ("invalid_action", "malformed_action", "infrastructure_interrupted",
                   "teacher_empty_response", "terminal_unsuccessful", "success", "max_steps",
                   "profile_unsolved", "teacher_failure", "provider_config_error", "unknown"):
        lines.append(f"| `{status}` | {statuses.get(status, 0)} |")
    lines += [
        "",
        f"- 旧 invalid 标签相关 attempts（含后来被 relabel 为 `profile_unsolved` 且 `invalid_action_count>0`）：{audit['old_invalid_attempts']}",
        f"- artifact `malformed_action_count` 累计：{audit['malformed_field_count']}（其中可能已被旧 profiler relabel 为 `profile_unsolved`）",
        f"- inferred invalid step（1-based，来自最后一条已执行 action）：{_fmt_counter(audit['invalid_steps'])}",
        "- 上述 step 是 artifact audit inference；不是把旧 label 当作 ground truth。",
        "",
        "## 3. API calls / attempt",
        "",
        f"- total recorded API requests: {audit['total_requests']}",
        f"- API calls per attempt（全部 attempts）: {_fmt_counter(audit['calls'])}",
        f"- API calls per old-invalid attempt（含 relabelled profile_unsolved）: {_fmt_counter(audit['invalid_calls'])}",
        f"- HTTP status: {_fmt_counter(audit['api_statuses'])}",
        f"- returned model: {_fmt_counter(audit['api_models'])}",
        "- 诊断字段缺失时按 `N/A` 处理；本报告不读取 provider body。",
        "",
        "## 4. Action 分布",
        "",
        f"- all recorded transitions: {_fmt_counter(audit['action_kinds'])}",
        f"- old-invalid attempts 的最后 action: {_fmt_counter(audit['final_invalid_kinds'])}",
        f"- legacy artifact 无 `status` 但 SQLite 有标签：{len(audit['status_discrepancies'])}（按 artifact audit 计为 `unknown`）",
        "",
        "## 5. Pre-step legality audit（archived observation heuristic）",
        "",
        f"- click transitions audited: {audit['click_transitions']}",
        f"- argument present in archived pre-step observation: {audit['pre_true']}",
        f"- argument absent in archived pre-step observation: {audit['pre_false']}",
        f"- pre-step unknown: {audit['pre_unknown']}",
        f"- argument absent after the step (post-state mutation signal): {audit['post_missing']}",
        f"- argument still present after the step: {audit['post_present']}",
        f"- post-step unknown: {audit['post_unknown']}",
        "",
        "### Re-audit classification",
        "",
        "| audited_reason | count | interpretation |",
        "|---|---:|---|",
        f"| `false_invalid_diagnostic` | {audit['classification'].get('false_invalid_diagnostic', 0)} | pre-step 可见且执行后页面改变，符合 post-state validity bug |",
        f"| `genuinely_illegal_click_candidate` | {audit['classification'].get('genuinely_illegal_click_candidate', 0)} | pre-step archived observation 未找到参数（仍需环境 replay 定论） |",
        f"| `genuine_malformed_action` | {audit['classification'].get('genuine_malformed_action', 0)} | 没有可执行 parsed action；不应归为 validity bug |",
        f"| `unresolved` | {audit['classification'].get('unresolved', 0)} | artifact 信息不足 |",
        "",
        "> `argument present/absent` 是保守的文本审计信号；完整 pre-step legality 仍应由 non-paid remote replay regression 覆盖。",
        "",
        "## 6. Teacher visible response 样本",
        "",
    ]
    samples = _sample_attempts(runs, minimum=5)
    if not samples:
        lines.append("没有包含 visible response 的样本 artifact。")
    for number, (attempt, trace_index, selected) in enumerate(samples, 1):
        traces = parser_trace(attempt)
        trace = traces[trace_index] if trace_index < len(traces) else traces[0]
        name, arg = trace.get("parsed_name"), trace.get("parsed_arg")
        candidate, step, reason = invalid_candidate(attempt)
        lines += [
            f"### Sample {number}: task `{attempt.task_id}` (run `{attempt.run_id}`, status `{attempt.status}`)",
            "",
            f"- response step: {trace.get('step')} ({selected}); extracted action: `{_redact(str(trace.get('extracted', '')).strip())}`",
            f"- normalized action: type=`{name or 'N/A'}`, argument=`{_redact(str(arg)) if arg is not None else 'N/A'}`, kind=`{action_kind(name, arg)}`",
            f"- audited reason: `{candidate}` (step={step or 'N/A'}; {reason or 'N/A'})",
            "",
            "```text",
            _sample_text(str(trace.get("response", ""))),
            "```",
            "",
        ]
    # Keep at least one malformed visible response visible even when the first
    # five distinct-task samples above are all ordinary invalid-labelled runs.
    sampled_ids = {(item[0].run_id, item[0].task_id, item[0].record.get("attempt_id")) for item in samples}
    malformed_examples = [
        attempt for run in runs for attempt in run.attempts
        if int(attempt.record.get("malformed_action_count", 0) or 0) > 0
        or attempt.status == "malformed_action"
    ]
    for attempt in malformed_examples[:2]:
        key = (attempt.run_id, attempt.task_id, attempt.record.get("attempt_id"))
        if key in sampled_ids:
            continue
        traces = parser_trace(attempt)
        if not traces:
            continue
        trace = traces[0]
        lines += [
            f"### Malformed sample: task `{attempt.task_id}` (run `{attempt.run_id}`)",
            "",
            f"- extracted action: `{_sample_text(str(trace.get('extracted', '')).strip(), 400)}`; "
            "upstream parse 未产生可执行的 canonical action。",
            "",
            "```text",
            _sample_text(str(trace.get("response", ""))),
            "```",
            "",
        ]
    lines += [
        "## 7. Parser trace",
        "",
        f"- visible responses checked: {audit['parser_checked']}",
        f"- normalized upstream-extraction mismatches against recorded action: {len(audit['parser_mismatches'])}",
        "- trailing whitespace/newline differences are normalized before mismatch counting.",
    ]
    if audit["parser_mismatches"]:
        lines += ["", "Mismatch samples:"]
        for attempt, index, trace in audit["parser_mismatches"][:5]:
            lines.append(
                f"- `{attempt.run_id}/{attempt.task_id}/step-{index + 1}`: "
                f"extracted=`{_sample_text(str(trace.get('extracted', '')), 240)}`; "
                f"recorded=`{_sample_text(str(trace.get('recorded', '')), 240)}`"
            )
    lines += [
        "",
        "## 8. Official protocol comparison / root-cause evidence",
        "",
        "1. 当前 `scripts/remote_teacher_env_service.py` 在 `env.step(action)` 后才调用 `get_available_actions()`，"
        "并用已经刷新过的 `text_to_clickable` 计算 `action_valid`；这把 PRE-STEP 合法性判断变成了 POST-STEP 判断。",
        "2. 当前 profiler 把 remote 返回的 raw `observation` 直接追加为下一条 user message；"
        "官方 Single formatter 还要附加“搜索功能是否可用”和“可点击的按钮”，因此必须以明确的 policy-visible observation contract 对齐。",
        "3. 官方 Persona reset/interaction 使用 `instruction_simple` 投影；应确认 remote/profiler 不把 evaluator/full instruction 泄漏给 teacher。",
        "4. upstream action extraction/parse 是 source of truth；本审计未把不同 Thought 文本视为不同 action。",
        "",
        "### 本次审计结论",
        "",
        f"- artifact 证据支持 post-state validity 假阳性候选：{audit['classification'].get('false_invalid_diagnostic', 0)} 条。",
        f"- 真正 malformed 候选：{audit['classification'].get('genuine_malformed_action', 0)} 条。",
        "- 在修复并通过 non-paid pre-step/official-observation regression 前，不应恢复旧 run 或把旧数据纳入统计。",
        "",
        "## 9. Fix / regression contract",
        "",
        "- 已修复：`action_valid` 针对 pre-step available actions 计算；step 后只返回 next-state available actions。",
        "- 已修复：remote service 返回 canonical `policy_observation`，profiler 只消费该字段；raw observation 仅作 diagnostics。",
        "- 已修复：Single/Persona system prompt 来自 pinned `single_eval` YAML；Persona 使用 `instruction_simple` projection，且不泄漏 evaluator-only fields。",
        "- 已修复：action extraction/parser trace 复用 upstream 的精确 marker/regex，不修复 markdown、大小写或非法格式。",
        "- 已加入 regression：合法 product/option/Buy Now、非法 click、Single/Persona observation formatter、upstream extraction、旧 run invalidation/resume guard。",
        "- 本机未执行 remote live replay：当前网络无法解析 `rtx-pro-6000-3`；remote replay 需由用户网络恢复后手动 smoke。",
        "",
        "## 10. Old paid run handling",
        "",
        ("- 本次运行已执行显式 `--invalidate`；旧 artifacts 保留，run 状态为 "
         "`invalidated_by_implementation_bug`，禁止 `--resume`。" if invalidated else
         "- 默认审计未修改旧 run。确认 implementation bug 后，需显式执行 `--invalidate`；旧 artifacts 必须保留，且禁止 resume。"),
        "- 新修复应 bump immutable protocol/environment version，并创建全新 profiling run。",
        "",
        "## 11. Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- None")
    lines += ["", "生成时间：" + utc_now(), ""]
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _invalidate_run(run: Run, reason: str) -> None:
    """Mark one run invalidated, preserving all attempt/trajectory files."""

    manifest = dict(run.manifest)
    # Idempotence: retain the first timestamp/reason when already invalidated.
    manifest.setdefault("invalidated_at", utc_now())
    manifest.setdefault("invalidation_reason", reason)
    manifest.setdefault("invalidation_code_version", INVALIDATION_CODE_VERSION)
    manifest["status"] = "invalidated_by_implementation_bug"
    _atomic_write_text(run.path / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    db_path = run.path / "state.sqlite"
    if not db_path.exists():
        return
    db = sqlite3.connect(db_path)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS run_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        with db:
            values = {
                "status": "invalidated_by_implementation_bug",
                "invalidation_reason": manifest["invalidation_reason"],
                "invalidated_at": manifest["invalidated_at"],
                "invalidation_code_version": manifest["invalidation_code_version"],
            }
            for key, value in values.items():
                db.execute(
                    "INSERT INTO run_state(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value))
                )
    finally:
        db.close()
    run.manifest = manifest
    run.ledger_state.update({
        "status": "invalidated_by_implementation_bug",
        "invalidation_reason": str(manifest["invalidation_reason"]),
        "invalidated_at": str(manifest["invalidated_at"]),
        "invalidation_code_version": str(manifest["invalidation_code_version"]),
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit P3a invalid_action artifacts without paid API calls")
    parser.add_argument("--data-root", type=Path, default=Path("data/teacher_profile"))
    parser.add_argument("--scenario", default="single")
    parser.add_argument("--run-id", help="只审计指定 run；默认审计 scenario 下全部 p3a runs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--invalidate", action="store_true",
                        help="显式将所选旧 p3a runs 标记 invalidated_by_implementation_bug（保留 artifacts）")
    parser.add_argument("--reason", default=(
        "post-state action_valid diagnostic bug and policy-visible observation protocol audit; "
        "old paid profiling data is invalid for P3a statistics"
    ))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Deliberately do not load .env.teacher or any provider configuration.
    runs, warnings = load_runs(args.data_root, args.scenario, args.run_id)
    if not runs:
        print("没有可审计的 p3a profiling run。", file=sys.stderr)
        # Still emit a report, which makes the missing-data condition explicit.
    if args.invalidate:
        for run in runs:
            _invalidate_run(run, args.reason)
        print(f"已显式 invalidation {len(runs)} 个 run；所有 artifacts 保留。")
    audit = _run_audit(runs)
    report = _report(audit, warnings, invalidated=args.invalidate)
    _atomic_write_text(args.output, report)
    print(f"Debug report: {args.output}")
    print(f"Runs: {len(runs)} | Attempts: {audit['total_attempts']} | API requests: {audit['total_requests']}")
    print("Statuses: " + _fmt_counter(audit["statuses"]))
    print("Audited invalid reasons: " + _fmt_counter(audit["classification"]))
    if audit["parser_mismatches"]:
        print(f"Parser mismatches: {len(audit['parser_mismatches'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the canonical, audit-friendly P3a summary from local artifacts.

This command never calls a teacher API. It selects completed profiling runs,
overlays the Persona repair run on the clean 22-task base run, writes a
machine-readable selection/summary under ``data/`` and renders Markdown.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rollout.diversity import behavior_fingerprint, exact_duplicate, similarity_features  # noqa: E402

INFRA_STATUSES = {"infrastructure_interrupted", "provider_config_error"}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _state(run: Path) -> str | None:
    import sqlite3
    try:
        db = sqlite3.connect(run / "state.sqlite")
        row = db.execute("SELECT value FROM run_state WHERE key='status'").fetchone()
        db.close()
        return str(row[0]) if row else None
    except (OSError, sqlite3.Error):
        return None


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _complete_candidates(root: Path, scenario: str, purpose: str) -> list[tuple[Path, dict[str, Any]]]:
    result = []
    for run in sorted((root / scenario).glob("*/")):
        manifest_path = run / "run_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = _read(manifest_path)
        except (OSError, ValueError):
            continue
        if manifest.get("purpose") == purpose and _state(run) == "complete":
            result.append((run, manifest))
    return result


def _select_one(root: Path, scenario: str, purpose: str, count: int,
                required_ids: set[str] | None = None) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for run, manifest in _complete_candidates(root, scenario, purpose):
        ids = {str(x) for x in manifest.get("selected_task_ids", [])}
        if len(ids) == count and (required_ids is None or ids == required_ids):
            candidates.append((str(manifest.get("created_at", "")), run, manifest))
    if not candidates:
        raise RuntimeError(f"未找到完整 canonical run: scenario={scenario}, purpose={purpose}")
    _, run, manifest = sorted(candidates)[-1]
    return run, manifest


def build_selection(root: Path) -> dict[str, Any]:
    single_run, single_manifest = _select_one(root, "single", "p3a_profiling", 24)
    persona_base_run, persona_base_manifest = _select_one(root, "single_persona", "p3a_profiling", 24)
    repair_ids = {"934909004241", "895516447505"}
    repair_run, repair_manifest = _select_one(root, "single_persona", "p3a_repair", 2, repair_ids)
    persona_ids = {str(x) for x in persona_base_manifest["selected_task_ids"]}
    if not repair_ids.issubset(persona_ids):
        raise RuntimeError("repair IDs 不属于 Persona base task list")
    excluded_repair_runs = []
    for run in sorted((root / "single_persona").glob("*/")):
        manifest_path = run / "run_manifest.json"
        if run == repair_run or not manifest_path.exists():
            continue
        try:
            manifest = _read(manifest_path)
        except (OSError, ValueError):
            continue
        if manifest.get("purpose") == "p3a_repair":
            excluded_repair_runs.append({"run_id": manifest.get("run_id", run.name), "status": _state(run), "reason": "non-canonical repair run; retained for audit"})
    result = {
        "selection_version": "p3a-canonical-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "single": {
            "base_run_id": single_manifest.get("run_id", single_run.name),
            "base_run_path": str(single_run), "task_ids": [str(x) for x in single_manifest["selected_task_ids"]],
            "excluded_task_ids": [],
        },
        "single_persona": {
            "base_run_id": persona_base_manifest.get("run_id", persona_base_run.name),
            "base_run_path": str(persona_base_run), "repair_run_id": repair_manifest.get("run_id", repair_run.name),
            "repair_run_path": str(repair_run), "task_ids": sorted(persona_ids),
            "excluded_task_ids": sorted(repair_ids),
            "exclusion_reason": "early infrastructure/config/resume attempt-accounting contamination",
            "excluded_repair_runs": excluded_repair_runs,
        },
        "protocol": {"environment_version": repair_manifest.get("environment_version"),
                      "policy_observation_version": repair_manifest.get("policy_observation_version"),
                      "profiler_protocol_version": repair_manifest.get("profiler_protocol_version")},
    }
    return result


def _records(run: Path) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for folder in (run / "attempts", run / "trajectories"):
        if not folder.exists():
            continue
        for path in folder.glob("*.json"):
            try:
                item = _read(path)
            except (OSError, ValueError):
                continue
            values[str(item.get("attempt_id", path.name))] = item
    return list(values.values())


def _parse_time(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _wall(record: Mapping[str, Any]) -> float | None:
    start = _parse_time(record.get("started_at"))
    end = _parse_time(record.get("finished_at")) or _parse_time(record.get("saved_at"))
    return end - start if start is not None and end is not None and end >= start else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))]


def _stats(values: list[float]) -> dict[str, float | None]:
    return {"mean": statistics.mean(values) if values else None, "p50": _percentile(values, .5),
            "p90": _percentile(values, .9), "p95": _percentile(values, .95),
            "max": max(values) if values else None}


def _wilson(successes: int, total: int) -> dict[str, float | None]:
    if not total:
        return {"estimate": None, "low": None, "high": None}
    z, p = 1.959963984540054, successes / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    half = z * (p * (1 - p) / total + z * z / (4 * total * total)) ** .5 / den
    return {"estimate": p, "low": max(0.0, centre - half), "high": min(1.0, centre + half)}


def _genuine(record: Mapping[str, Any]) -> bool:
    return str(record.get("status")) not in INFRA_STATUSES | {"in_progress"}


def _fingerprint(record: Mapping[str, Any]) -> dict[str, Any]:
    fp = record.get("behavior_fingerprint")
    if isinstance(fp, Mapping):
        return dict(fp)
    purchase = record.get("evaluator_only", {}).get("purchase", {}) or {}
    return behavior_fingerprint(record.get("actions", []), final_purchase_asin=purchase.get("asin"))


def _task_outcomes(records: list[dict[str, Any]], task_ids: list[str]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        if _genuine(item):
            by_task[str(item.get("task_id"))].append(item)
    first_success, first_success_ordinal, first_counts, second = {}, {}, {}, []
    for task in task_ids:
        rows = sorted(by_task.get(task, []), key=lambda x: (_parse_time(x.get("started_at")) or 0, int(x.get("attempt_index", 0) or 0)))
        first = [x for x in rows if x.get("attempt_phase") == "first_success"]
        first_counts[task] = len(first)
        for ordinal, row in enumerate(first, 1):
            if row.get("success") is True:
                first_success[task] = row
                first_success_ordinal[task] = ordinal
                break
        second.extend(x for x in rows if x.get("attempt_phase") == "second_demo")
    return {"by_task": by_task, "first_success": first_success, "first_success_ordinal": first_success_ordinal, "first_counts": first_counts, "second": second}


def collect_canonical(root: Path, selection: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    selected = selection[scenario]
    runs = [Path(selected["base_run_path"])] + ([Path(selected["repair_run_path"])] if scenario == "single_persona" else [])
    excluded = set(selected.get("excluded_task_ids", []))
    records = [item for run in runs for item in _records(run) if str(item.get("task_id")) not in excluded]
    task_ids = [str(x) for x in selected["task_ids"]]
    outcome = _task_outcomes(records, task_ids)
    genuine = [x for x in records if _genuine(x)]
    infra = [x for x in records if str(x.get("status")) in INFRA_STATUSES]
    interrupted = [x for x in infra if x.get("status") == "infrastructure_interrupted"]
    config_errors = [x for x in infra if x.get("status") == "provider_config_error"]
    first_success = outcome["first_success"]
    first_attempt_success = sum(1 for ordinal in outcome["first_success_ordinal"].values() if ordinal == 1)
    within2 = sum(1 for x in outcome["first_success_ordinal"].values() if x <= 2)
    within3 = sum(1 for x in outcome["first_success_ordinal"].values() if x <= 3)
    successful = [x for x in genuine if x.get("success") is True]
    second = outcome["second"]
    second_success = [x for x in second if x.get("success") is True]
    task_fps: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for item in successful:
        task_fps[str(item.get("task_id"))].append((str(item.get("attempt_phase")), _fingerprint(item)))
    duplicate_pairs = second_duplicate = near_count = 0
    feature_equal = Counter()
    for entries in task_fps.values():
        for i in range(len(entries)):
            for j in range(i):
                if exact_duplicate(entries[i][1], entries[j][1]):
                    duplicate_pairs += 1
                    if entries[i][0] == "second_demo" or entries[j][0] == "second_demo":
                        second_duplicate += 1
                left, right = entries[i][1], entries[j][1]
                for key in ("normalized_actions", "search_queries", "clicked_products", "selected_options", "final_purchase_asin"):
                    if left.get(key) == right.get(key):
                        feature_equal[key] += 1
                if similarity_features(entries[i][1], entries[j][1]).get("provisional_near_duplicate"):
                    near_count += 1
    actions = [float(x.get("action_steps")) for x in successful if isinstance(x.get("action_steps"), (int, float))]
    groups = {"successful": successful, "genuine_failed": [x for x in genuine if x.get("success") is not True], "infrastructure_interrupted": infra}
    timing, tokens = {}, {}
    for name, rows in groups.items():
        api = [float(d["latency_s"]) for r in rows for d in r.get("api_diagnostics", []) if isinstance(d.get("latency_s"), (int, float))]
        env = [float(d["latency_s"]) for r in rows for d in r.get("environment_diagnostics", []) if isinstance(d.get("latency_s"), (int, float))]
        walls = [w for r in rows if (w := _wall(r)) is not None]
        api_total = sum(float(r.get("api_latency_total_s", 0) or 0) for r in rows)
        timing[name] = {"api_latency_s": _stats(api), "environment_steps_latency_s": _stats(env), "total_wall_s": _stats(walls), "api_wall_proportion": api_total / sum(walls) if sum(walls) else None}
        ins = [float(r.get("total_input_tokens", 0) or 0) for r in rows]; outs = [float(r.get("total_output_tokens", 0) or 0) for r in rows]
        tokens[name] = {"input": _stats(ins), "output": _stats(outs), "total": _stats([i + o for i, o in zip(ins, outs)])}
    terminal = Counter(str(x.get("termination_reason", x.get("status", "unknown"))) for x in genuine)
    infra_http = Counter(str(d.get("http_status")) for r in infra for d in r.get("api_diagnostics", []) if d.get("http_status") is not None)
    hard = []
    for task in task_ids:
        rows = outcome["by_task"].get(task, [])
        if any(x.get("status") in {"profile_unsolved", "max_steps"} for x in rows):
            normalized = [str(a).lower() for x in rows for a in x.get("actions", [])]
            hard.append({"task_id": task, "outcomes": dict(Counter(str(x.get("termination_reason", x.get("status"))) for x in rows)), "max_steps_attempts": sum(x.get("status") == "max_steps" for x in rows), "repeated_search": len(normalized) != len(set(normalized)) and sum(x.startswith("search[") for x in normalized) > 1, "repeated_click": len(normalized) != len(set(normalized)) and sum(x.startswith("click[") for x in normalized) > 1, "action_count": len(normalized), "target_product_clicked": any(task.lower() in x for x in normalized), "last_reward": rows[-1].get("reward_metrics", {}) if rows else {}})
    success_wall = [_wall(x) for x in successful if _wall(x) is not None]
    failed_wall = [_wall(x) for x in groups["genuine_failed"] if _wall(x) is not None]
    success_mean = statistics.mean(success_wall) if success_wall else None
    projection = {
        "observed_eventual_success_rate": len(first_success) / len(task_ids) if task_ids else None,
        "eventual_success_rate_wilson": _wilson(len(first_success), len(task_ids)),
        "rough_initial_unique_successes_for_3000_tasks": 3000 * len(first_success) / len(task_ids) if task_ids else None,
        "rough_initial_unique_successes_wilson_low_high": [3000 * _wilson(len(first_success), len(task_ids))["low"], 3000 * _wilson(len(first_success), len(task_ids))["high"]] if task_ids else [None, None],
        "successful_attempt_mean_wall_s": success_mean,
        "genuine_failed_attempt_mean_wall_s": statistics.mean(failed_wall) if failed_wall else None,
        "serial_6000_successes_hours": (6000 * success_mean / 3600) if success_mean is not None else None,
        "serial_12000_successes_hours": (12000 * success_mean / 3600) if success_mean is not None else None,
        "note": "rough serial baseline only; assumes P3a behavior/latency remains stable and excludes formal policy decisions",
    }
    return {
        "scenario": scenario, "tasks": len(task_ids), "task_ids": task_ids,
        "records": {"total": len(records), "genuine": len(genuine), "infrastructure_or_config": len(infra), "infrastructure_interrupted": len(interrupted), "provider_config_error": len(config_errors)},
            "first_success": {"first_attempt": _wilson(first_attempt_success, len(task_ids)), "within_2": _wilson(within2, len(task_ids)), "within_3": _wilson(within3, len(task_ids)), "eventual": _wilson(len(first_success), len(task_ids)), "successes": len(first_success), "profile_unsolved": len(task_ids) - len(first_success), "genuine_attempts": sum(outcome["first_counts"].values()), "attempts_per_success": statistics.mean(outcome["first_success_ordinal"].values()) if first_success else None, "attempt_distribution": dict(Counter(outcome["first_success_ordinal"].values()))},
        "second_demo": {"tasks_entering": len(first_success), "attempts": len(second), "successes": len(second_success), "tasks_with_success": len({str(x.get("task_id")) for x in second_success}), "at_least_one_success": _wilson(len({str(x.get("task_id")) for x in second_success}), len(first_success)), "success_per_attempt": _wilson(len(second_success), len(second))},
        "diversity": {"successful_trajectories": len(successful), "exact_duplicate_pairs": duplicate_pairs, "first_vs_second_exact_duplicate_pairs": second_duplicate, "exact_duplicate_rate": duplicate_pairs / len(successful) if successful else None, "provisional_near_duplicate_pairs": near_count, "behavioral_feature_equal_pair_counts": dict(feature_equal), "note": "provisional near-duplicate is diagnostic only; no formal P3c threshold is frozen", "action_length": _stats(actions)},
        "failures": {"termination": dict(terminal), "hard_tasks": hard}, "timing": timing, "tokens": tokens, "formal_collection_projection": projection,
        "infrastructure": {"http_status_counts": dict(infra_http), "retry_total": sum(int(x.get("infrastructure_retries", 0) or 0) for x in records), "interrupted_attempts": len(interrupted), "provider_config_errors": len(config_errors)},
        "provenance": {"base_run_id": selected.get("base_run_id"), "repair_run_id": selected.get("repair_run_id"), "excluded_task_ids": sorted(excluded)},
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None: return "N/A"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _rate(value: Mapping[str, Any]) -> str:
    return f"{_fmt(value.get('estimate'))} [{_fmt(value.get('low'))}, {_fmt(value.get('high'))}]"


def _seconds(value: Any) -> str:
    return "N/A" if value is None else f"{_fmt(value)}s"


def render_report(summary: Mapping[str, Any], selection: Mapping[str, Any]) -> str:
    lines = ["# P3a Teacher Profile", "", "本报告由本地 profiling SQLite/JSON artifacts 自动生成；不调用 teacher API，不把旧 contaminated records 当作 canonical 数据。", "", "> P3a 的 3/2 次数是 profiling operational limits，不是 P3b formal collection cap；diversity threshold、reserve policy、并发数均未冻结。", "", "## 1. Data provenance", "", f"- Selection version: `{selection['selection_version']}`", f"- Protocol: `{json.dumps(selection['protocol'], ensure_ascii=False)}`", "- Single: clean 24-task completed run.", "- Single&Pers: base 24-task run excluding `934909004241`, `895516447505`, overlaid with the completed 2-task repair run.", f"- Non-canonical repair runs retained for audit: `{json.dumps(selection['single_persona'].get('excluded_repair_runs', []), ensure_ascii=False)}`", ""]
    for scenario, title in (("single", "Single"), ("single_persona", "Single & Personalization")):
        item, fs, sd, div = summary[scenario], summary[scenario]["first_success"], summary[scenario]["second_demo"], summary[scenario]["diversity"]
        lines += [f"## {title}", "", f"- Tasks: {item['tasks']}; genuine attempts: {item['records']['genuine']}; infrastructure interruptions excluded: {item['records']['infrastructure_interrupted']}; provider/config errors excluded: {item['records']['provider_config_error']}", f"- First-attempt success (Wilson 95% CI): {_rate(fs['first_attempt'])}", f"- Success within 2 / 3 attempts: {_rate(fs['within_2'])} / {_rate(fs['within_3'])}", f"- Eventual first-success: {_rate(fs['eventual'])}; profile_unsolved: {fs['profile_unsolved']}", f"- First-success attempts: {fs['genuine_attempts']}; attempts/success: {_fmt(fs['attempts_per_success'])}; distribution: `{json.dumps(fs['attempt_distribution'], sort_keys=True)}`", f"- Second-demo: {sd['tasks_entering']} tasks entered, {sd['attempts']} attempts, {sd['successes']} successes; at-least-one-success: {_rate(sd['at_least_one_success'])}; success/attempt: {_rate(sd['success_per_attempt'])}", f"- Exact duplicate pairs: {div['exact_duplicate_pairs']}; first-vs-second duplicate pairs: {div['first_vs_second_exact_duplicate_pairs']}; provisional near-duplicate pairs: {div['provisional_near_duplicate_pairs']} (diagnostic only)", f"- Action length mean/p50/p90/max: {_fmt(div['action_length']['mean'])}/{_fmt(div['action_length']['p50'])}/{_fmt(div['action_length']['p90'])}/{_fmt(div['action_length']['max'])}", f"- Failure modes: `{json.dumps(item['failures']['termination'], ensure_ascii=False, sort_keys=True)}`", ""]
        for group in ("successful", "genuine_failed", "infrastructure_interrupted"):
            t, tok = item["timing"][group], item["tokens"][group]
            lines.append(f"- {group}: wall p50/p90/max={_seconds(t['total_wall_s']['p50'])}/{_seconds(t['total_wall_s']['p90'])}/{_seconds(t['total_wall_s']['max'])}; API p50/p90/max={_seconds(t['api_latency_s']['p50'])}/{_seconds(t['api_latency_s']['p90'])}/{_seconds(t['api_latency_s']['max'])}; env-step p50/p90/max={_seconds(t['environment_steps_latency_s']['p50'])}/{_seconds(t['environment_steps_latency_s']['p90'])}/{_seconds(t['environment_steps_latency_s']['max'])}; API wall proportion={_fmt(t['api_wall_proportion'])}; tokens mean input/output={_fmt(tok['input']['mean'],0)}/{_fmt(tok['output']['mean'],0)}")
        lines += [f"- Behavioral feature equal-pair counts (action/search/clicked-product/options/final-purchase): `{json.dumps(div['behavioral_feature_equal_pair_counts'], ensure_ascii=False, sort_keys=True)}`", f"- Infrastructure: retries={item['infrastructure']['retry_total']}; HTTP statuses={json.dumps(item['infrastructure']['http_status_counts'], sort_keys=True)}", f"- Hard tasks: `{json.dumps(item['failures']['hard_tasks'], ensure_ascii=False)}`", ""]
    lines += ["## Formal collection implications", "", "这些是 P3b 的数据输入，不是已冻结决策：API latency dominates serial wall time；hard-task retries can be disproportionately costly；24-task estimates have wide uncertainty；Persona project pool is 3323 rather than paper 3383。Formal caps, diversity threshold, reserve replacement and concurrency remain open for P3b。", "", "串行 rough projection（假设行为和延迟稳定）：", ""]
    for scenario, title in (("single", "Single"), ("single_persona", "Single&Pers")):
        p = summary[scenario]["formal_collection_projection"]
        lines.append(f"- {title}: 3000-task initial unique-success estimate={_fmt(p['rough_initial_unique_successes_for_3000_tasks'])} (Wilson scaled range {_fmt(p['rough_initial_unique_successes_wilson_low_high'][0])}–{_fmt(p['rough_initial_unique_successes_wilson_low_high'][1])}); 6000 successes≈{_fmt(p['serial_6000_successes_hours'])}h; 12000 total≈{_fmt(p['serial_12000_successes_hours'])}h。")
    lines += ["", "## Excluded history", "", "旧 implementation/config/resume 污染记录保留在本地 audit artifacts 中，不进入 canonical summary；不应作为 teacher failure 或 success rate denominator。"]
    return "\n".join(lines) + "\n"


def build_outputs(data_root: Path, summary_path: Path, report_path: Path, selection_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = build_selection(data_root)
    summary = {"summary_version": "p3a-summary-v2", "created_at": datetime.now(timezone.utc).isoformat(), "selection_hash": _hash_json(selection), "selection": selection, "single": collect_canonical(data_root, selection, "single"), "single_persona": collect_canonical(data_root, selection, "single_persona"), "formal_policy_status": "not_frozen_p3b_pending"}
    selection_path.parent.mkdir(parents=True, exist_ok=True); summary_path.parent.mkdir(parents=True, exist_ok=True); report_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(summary, selection), encoding="utf-8")
    return selection, summary


# Compatibility helpers retained for small local aggregation tests and ad-hoc
# inspection. Canonical production output uses ``build_outputs`` above.
def collect(root: Path, scenario: str) -> dict[str, Any]:
    runs = [p for p in sorted((root / scenario).glob("*/")) if (p / "run_manifest.json").exists()]
    records = [item for run in runs for item in _records(run)]
    task_ids = sorted({str(x.get("task_id")) for x in records if x.get("task_id") is not None})
    selection = {scenario: {"base_run_path": str(runs[-1]) if runs else str(root / scenario), "task_ids": task_ids, "excluded_task_ids": []}}
    result = collect_canonical(root, selection, scenario) if runs else {"scenario": scenario, "tasks": 0}
    result["manifest_count"] = len(runs)
    result["tasks_profiled"] = result.get("tasks", 0)
    result["attempts"] = result.get("records", {}).get("genuine", 0)
    return result


def render(stats: list[dict[str, Any]]) -> str:
    if len(stats) == 2 and {x.get("scenario") for x in stats} == {"single", "single_persona"}:
        summary = {x["scenario"]: x for x in stats}
        return render_report(summary, {"selection_version": "legacy", "protocol": {}})
    return "\n".join(["# P3a Teacher Profile", "", "Canonical report requires both selected scenario runs; unavailable fields remain N/A.", ""])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "teacher_profile")
    parser.add_argument("--summary", type=Path, default=ROOT / "data" / "teacher_profile" / "p3a_summary.json")
    parser.add_argument("--selection", type=Path, default=ROOT / "data" / "teacher_profile" / "p3a_canonical_selection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "P3A_TEACHER_PROFILE.md")
    args = parser.parse_args()
    build_outputs(args.data_root, args.summary, args.output, args.selection)
    print(f"Wrote {args.summary}\nWrote {args.selection}\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

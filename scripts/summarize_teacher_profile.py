#!/usr/bin/env python3
"""Aggregate local P3a profiling artifacts without inventing missing values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[index]


def fmt(value: float | int | None) -> str:
    return "N/A" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


def collect(root: Path, scenario: str) -> dict:
    attempts: list[dict] = []
    trajectories: list[dict] = []
    manifests: list[dict] = []
    for run in sorted((root / scenario).glob("*/")):
        manifest_path = run / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        for path in (run / "attempts").glob("*.json"):
            try: attempts.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError): pass
        for path in (run / "trajectories").glob("*.json"):
            try: trajectories.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError): pass
    success = [item for item in attempts if item.get("success") is True]
    first = [item for item in attempts if item.get("attempt_phase") == "first_success"]
    first_task_ids = {item.get("task_id") for item in first}
    first_success_by_task: dict[str, int] = {}
    for item in first:
        if item.get("success") and item.get("task_id") not in first_success_by_task:
            first_success_by_task[str(item.get("task_id"))] = int(item.get("attempt_index", 0))
    second = [item for item in attempts if item.get("attempt_phase") == "second_demo"]
    second_success_tasks = {str(item.get("task_id")) for item in second if item.get("success")}
    term = {}
    for item in attempts: term[item.get("termination_reason", "unknown")] = term.get(item.get("termination_reason", "unknown"), 0) + 1
    api_latency = [float(x.get("latency_s")) for a in attempts for x in a.get("api_diagnostics", []) if isinstance(x.get("latency_s"), (int,float))]
    env_latency = [float(x.get("latency_s")) for a in attempts for x in a.get("environment_diagnostics", []) if isinstance(x.get("latency_s"), (int,float))]
    action_lengths = [float(a.get("action_steps")) for a in success if isinstance(a.get("action_steps"), (int,float))]
    duplicate = [a for a in trajectories if a.get("exact_duplicate")]
    near = [a for a in trajectories if a.get("potential_near_duplicate")]
    total_in = sum(int(a.get("total_input_tokens", 0) or 0) for a in attempts)
    total_out = sum(int(a.get("total_output_tokens", 0) or 0) for a in attempts)
    return {
        "scenario": scenario, "tasks_profiled": len(first_task_ids), "attempts": len(attempts),
        "first_attempt_success_rate": sum(bool(a.get("success")) and a.get("attempt_index") == 1 for a in first) / len(first_task_ids) if first_task_ids else None,
        "success_within_2": sum(v <= 2 for v in first_success_by_task.values()) / len(first_task_ids) if first_task_ids else None,
        "success_within_3": sum(v <= 3 for v in first_success_by_task.values()) / len(first_task_ids) if first_task_ids else None,
        "profile_unsolved": sum(a.get("status") == "profile_unsolved" for a in attempts),
        "attempts_per_first_success": statistics.mean(first_success_by_task.values()) if first_success_by_task else None,
        "second_success_rate": len(second_success_tasks) / len(first_task_ids) if first_task_ids else None,
        "exact_duplicate_rate": len(duplicate) / len(trajectories) if trajectories else None,
        "near_duplicate_count": len(near), "trajectories": len(trajectories),
        "action_p50": percentile(action_lengths, .5), "action_p90": percentile(action_lengths, .9),
        "action_max": max(action_lengths) if action_lengths else None,
        "malformed_rate": sum(a.get("termination_reason") == "malformed_action" for a in attempts) / len(attempts) if attempts else None,
        "invalid_rate": sum(a.get("termination_reason") == "invalid_action" for a in attempts) / len(attempts) if attempts else None,
        "max_step_rate": sum(a.get("termination_reason") == "max_steps" for a in attempts) / len(attempts) if attempts else None,
        "api_latency_p50": percentile(api_latency, .5), "api_latency_p90": percentile(api_latency, .9), "api_latency_max": max(api_latency) if api_latency else None,
        "api_retries": sum(int(a.get("infrastructure_retries", 0) or 0) for a in attempts),
        "api_retry_rate": (sum(int(a.get("infrastructure_retries", 0) or 0) > 0 for a in attempts) / len(attempts)) if attempts else None,
        "input_tokens": total_in, "output_tokens": total_out,
        "tokens_per_success": ((total_in + total_out) / len(success)) if success else None,
        "environment_latency_p50": percentile(env_latency, .5), "environment_latency_p90": percentile(env_latency, .9),
        "environment_latency_max": max(env_latency) if env_latency else None,
        "infrastructure_interruptions": sum(a.get("status") == "infrastructure_interrupted" for a in attempts),
        "termination": term, "reward_metrics": {k: [a.get("reward_metrics", {}).get(k) for a in success if k in a.get("reward_metrics", {})] for k in ("r_succ", "r_strict", "r_loose")},
        "manifest_count": len(manifests),
    }


def render(stats: list[dict]) -> str:
    lines = ["# P3a Teacher Profile", "", "本报告只聚合 `data/teacher_profile/` 中已存在的 profiling artifacts；不会填充或推断缺失数字。", "", "> P3a limits are operational profiling limits, not formal collection caps. Formal caps/diversity threshold remain a P3b decision.", ""]
    for item in stats:
        lines += [f"## {item['scenario']}", "", f"- Tasks profiled: {item['tasks_profiled']}", f"- Total teacher attempts: {item['attempts']}", f"- First-attempt success rate: {fmt(item['first_attempt_success_rate'])}", f"- Success within 2/3 attempts: {fmt(item['success_within_2'])} / {fmt(item['success_within_3'])}", f"- Profile unsolved: {item['profile_unsolved']}", f"- Attempts per first success: {fmt(item['attempts_per_first_success'])}", f"- Second-success rate: {fmt(item['second_success_rate'])}", f"- Exact duplicate rate: {fmt(item['exact_duplicate_rate'])}", f"- Provisional near-duplicate trajectories: {item['near_duplicate_count']} / {item['trajectories']} (not a formal rule)", f"- Actions p50/p90/max: {fmt(item['action_p50'])} / {fmt(item['action_p90'])} / {fmt(item['action_max'])}", f"- Malformed / invalid / max-step rate: {fmt(item['malformed_rate'])} / {fmt(item['invalid_rate'])} / {fmt(item['max_step_rate'])}", f"- API latency p50/p90/max (s): {fmt(item['api_latency_p50'])} / {fmt(item['api_latency_p90'])} / {fmt(item['api_latency_max'])}", f"- API retries: {item['api_retries']} (attempt retry rate: {fmt(item['api_retry_rate'])}); input/output tokens: {item['input_tokens']} / {item['output_tokens']}; tokens/success: {fmt(item['tokens_per_success'])}", f"- Environment latency p50/p90/max (s): {fmt(item['environment_latency_p50'])} / {fmt(item['environment_latency_p90'])} / {fmt(item['environment_latency_max'])}", f"- Infrastructure interruptions: {item['infrastructure_interruptions']}", f"- Termination types: `{json.dumps(item['termination'], ensure_ascii=False, sort_keys=True)}`", ""]
        reward_lines = []
        for metric in ("r_succ", "r_strict", "r_loose"):
            values = [float(v) for v in item["reward_metrics"].get(metric, []) if isinstance(v, (int, float))]
            reward_lines.append(f"{metric} p50/p90/max={fmt(percentile(values, .5))}/{fmt(percentile(values, .9))}/{fmt(max(values) if values else None)}")
        lines += [f"- Reward distributions: {'; '.join(reward_lines)}"]
    lines += ["## Comparison", ""]
    if len(stats) == 2 and all(item["manifest_count"] for item in stats):
        lines += ["两个 scenario 的关键 profiling 指标：", "", "| Metric | single | single_persona |", "|---|---:|---:|"]
        for key, label in (("tasks_profiled", "Tasks profiled"), ("attempts", "Attempts"),
                           ("first_attempt_success_rate", "First-attempt success rate"),
                           ("second_success_rate", "Second-success rate"),
                           ("api_latency_p50", "API latency p50 (s)"),
                           ("environment_latency_p50", "Environment latency p50 (s)")):
            lines.append(f"| {label} | {fmt(stats[0].get(key))} | {fmt(stats[1].get(key))} |")
        lines.append("")
    else:
        lines += ["两个 scenario 的对比将在各自 profiling artifacts 产生后显示；未运行的 scenario 保持 N/A。", ""]
    lines += ["Cost: N/A — pricing not configured for the third-party relay.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/teacher_profile"))
    parser.add_argument("--output", type=Path, default=Path("docs/P3A_TEACHER_PROFILE.md"))
    args = parser.parse_args()
    stats = [collect(args.data_root, scenario) for scenario in ("single", "single_persona")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(stats), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

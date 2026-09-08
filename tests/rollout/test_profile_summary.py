import json
from pathlib import Path

from scripts.summarize_teacher_profile import collect, render
from scripts.summarize_teacher_profile import collect_canonical


def test_summary_aggregates_only_existing_artifacts(tmp_path):
    run = tmp_path / "single" / "run-1"
    (run / "attempts").mkdir(parents=True)
    (run / "trajectories").mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"purpose": "p3a_profiling", "scenario": "single"}))
    payload = {"task_id": "t1", "attempt_phase": "first_success", "attempt_index": 1,
               "success": True, "status": "success", "termination_reason": "success", "action_steps": 2,
               "api_diagnostics": [{"latency_s": 1.0}], "environment_diagnostics": [{"latency_s": .1}],
               "reward_metrics": {"r_succ": 1, "r_strict": 1, "r_loose": 1}}
    (run / "attempts" / "a.json").write_text(json.dumps(payload))
    (run / "trajectories" / "a.json").write_text(json.dumps({**payload, "exact_duplicate": False}))
    stats = collect(tmp_path, "single")
    assert stats["tasks_profiled"] == 1
    assert stats["attempts"] == 1
    assert "N/A" in render([stats])  # second scenario/cost remain unconfigured, not fabricated


def test_canonical_overlay_excludes_infrastructure_and_keeps_genuine_denominator(tmp_path):
    run = tmp_path / "single" / "run-1"
    (run / "attempts").mkdir(parents=True)
    (run / "trajectories").mkdir()
    rows = [
        {"attempt_id": "infra", "task_id": "t1", "attempt_phase": "first_success", "attempt_index": 1,
         "status": "infrastructure_interrupted", "success": False},
        {"attempt_id": "ok", "task_id": "t1", "attempt_phase": "first_success", "attempt_index": 2,
         "status": "success", "success": True, "termination_reason": "success", "action_steps": 2,
         "actions": ["search[x]"], "api_diagnostics": [], "environment_diagnostics": []},
    ]
    for row in rows:
        (run / "attempts" / f"{row['attempt_id']}.json").write_text(json.dumps(row))
    selection = {"single": {"base_run_path": str(run), "task_ids": ["t1"], "excluded_task_ids": []}}
    stats = collect_canonical(tmp_path, selection, "single")
    assert stats["records"] == {"total": 2, "genuine": 1, "infrastructure_or_config": 1,
                                 "infrastructure_interrupted": 1, "provider_config_error": 0}
    assert stats["first_success"]["first_attempt"]["estimate"] == 1.0
    assert stats["infrastructure"]["interrupted_attempts"] == 1

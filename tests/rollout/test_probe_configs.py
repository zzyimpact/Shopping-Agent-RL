from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_paid_probe_config_is_ten_frozen_train_tasks():
    probe = json.loads((ROOT / "configs/teacher/p3a_concurrency_probe_tasks.json").read_text())
    source = json.loads((ROOT / "configs/teacher/p3a_single_tasks.json").read_text())
    probe_ids = [str(row["task_id"]) for row in probe["tasks"]]
    source_ids = {str(row["task_id"]) for row in source["tasks"]}
    assert len(probe_ids) == 10 and len(set(probe_ids)) == 10
    assert set(probe_ids) <= source_ids
    assert all(row["official_split"] == "train" for row in probe["tasks"])


def test_nonpaid_stress_config_is_mixed_and_scripted():
    data = json.loads((ROOT / "configs/teacher/p3a_environment_stress_tasks.json").read_text())
    assert len(data["tasks"]) == 10
    assert {row["scenario"] for row in data["tasks"]} == {"single", "single_persona"}
    assert all(row["official_split"] == "train" and row["actions"] for row in data["tasks"])

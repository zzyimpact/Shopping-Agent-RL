from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from rollout.collection_policy import (
    CollectionPolicyError,
    canonical_policy_hash,
    collection_terminal_status,
    has_deterministic_no_progress_loop,
    interpret_post_success_attempts,
    load_collection_policy,
    next_coverage_action,
    validate_collection_policy,
)


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "configs" / "teacher" / "formal_collection_p3b_v1.yaml"
DOC_PATH = ROOT / "docs" / "P3B_COLLECTION_POLICY.md"


@pytest.fixture()
def policy():
    return load_collection_policy(POLICY_PATH)


def test_yaml_parse_and_deterministic_hash(policy):
    stored = policy["identity"]["policy_hash"]
    assert stored == canonical_policy_hash(policy)
    reordered = dict(reversed(list(deepcopy(policy).items())))
    assert canonical_policy_hash(reordered) == stored
    assert policy["identity"]["policy_version"] == "p3b-v1"


def test_frozen_core_values(policy):
    assert policy["identity"]["seed"] == 1
    assert policy["scope"]["pass_order"] == ["A", "B", "C"]
    expected_teacher = {
        "model": "gpt-5.6-sol",
        "api_style": "responses",
        "reasoning_effort": "high",
        "request_semantics": "p3a-compatible",
    }
    assert {key: policy["teacher"][key] for key in expected_teacher} == expected_teacher
    assert policy["concurrency"]["workers_default"] == 8
    assert policy["concurrency"]["workers_immutable_per_run"] is True
    assert policy["targets"]["primary_tasks_per_scenario"] == 3000
    assert policy["targets"]["accepted_trajectories_per_scenario"] == {
        "target": 6000, "hard": True,
    }
    assert policy["targets"]["unique_successful_tasks_per_scenario"]["hard"] is False
    assert policy["targets"]["accepted_demos_per_task"] == {
        "minimum": 1, "target": 2, "maximum": 3,
    }
    assert policy["attempt_budgets"] == {
        "first_success_genuine_attempts_max": 2,
        "second_demo_attempts_max": 2,
        "post_first_success_attempts_max": 3,
        "infrastructure_consumes_genuine_attempt": False,
        "unsuccessful_or_rejected_post_success_attempt_consumes_budget": True,
    }
    assert policy["protocol"]["environment_version"] == "task-scoped-v3-multisession"
    assert policy["diversity"]["exact_behavioral_duplicate"]["disposition"] == "hard_reject"
    assert policy["diversity"]["provisional_near_duplicate"]["disposition"] == "diagnostic_only"
    assert policy["quality"]["no_progress_loop"]["consecutive_identical_normalized_actions"] == 3


def test_manifest_and_reserve_provenance_is_frozen(policy):
    single = policy["task_sources"]["single"]
    persona = policy["task_sources"]["single_persona"]
    assert single["primary_task_count"] == persona["primary_task_count"] == 3000
    assert single["reserve_task_count"] == 18962
    assert persona["reserve_task_count"] == 323
    assert single["primary_task_ids_sha256"] == (
        "870c399fbb2034f58a0dc4b18b8563d4a8748a35d78a9e89de509db5ac2b969d"
    )
    assert persona["primary_task_ids_sha256"] == (
        "f8ca0e5d3982661d19343df337a9713da16211ac43a361074a2c5d9c35fb6521"
    )
    reserve = policy["passes"]["A"]["reserve"]
    assert reserve["selection"] == "deterministic_unused_same_stratum"
    assert reserve["seed"] == 1
    assert reserve["cross_stratum_fallback"] is False


def test_no_secret_or_url_config_fields(policy):
    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield str(key).lower()
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    forbidden = {"api_key", "teacher_api_key", "authorization", "api_url", "teacher_api_url", "base_url"}
    assert forbidden.isdisjoint(set(keys(policy)))
    values = json.dumps(policy, ensure_ascii=False).lower()
    assert "https://" not in values and "http://" not in values

    contaminated = deepcopy(policy)
    contaminated["teacher"]["api_url"] = "https://relay.invalid"
    contaminated["identity"]["policy_hash"] = canonical_policy_hash(contaminated)
    with pytest.raises(CollectionPolicyError, match="secret/URL"):
        validate_collection_policy(contaminated)


def test_docs_and_config_key_values_agree(policy):
    text = DOC_PATH.read_text(encoding="utf-8")
    assert f"`{policy['identity']['policy_version']}` 已冻结" in text
    assert policy["identity"]["policy_hash"] in text
    for snippet in (
        "正式默认 `workers=8`",
        "各自 hard target 为 **6,000 accepted successful",
        "最多 2 次 genuine first-success",
        "Pass B 最多\n2 attempts",
        "Post-first-success attempts/task（B+C 合计） | 3",
        "exact normalized behavioral action trajectory equality hard reject",
        "Provisional near duplicate 只做 diagnostic",
        "Pass 必须严格 A → B → C",
        "`data/teacher_raw/collector.log`",
        "`scripts/inspect_teacher_run.py --scenario <...> --latest`",
    ):
        assert snippet in text


def test_p3a_evidence_and_probe_values_are_recorded(policy):
    evidence = policy["evidence"]
    single = evidence["canonical_p3a"]["single"]
    persona = evidence["canonical_p3a"]["single_persona"]
    assert (single["eventual_first_success"], single["tasks"], single["first_phase_genuine_attempts"]) == (19, 24, 34)
    assert (persona["eventual_first_success"], persona["tasks"], persona["first_phase_genuine_attempts"]) == (20, 24, 35)
    assert evidence["canonical_p3a"]["historical_retry_event_telemetry_complete"] is False
    assert evidence["concurrency_probes"]["workers_8"]["HTTP_429"] == 0
    assert evidence["concurrency_probes"]["workers_10"]["HTTP_429"] == 15
    assert evidence["concurrency_probes"]["workers_8"]["trajectories_per_hour"] > evidence["concurrency_probes"]["workers_10"]["trajectories_per_hour"]


def test_example_a_first_success_then_b1_diverse_success(policy):
    assert next_coverage_action(
        genuine_attempts=1, first_success=True, same_stratum_reserve_available=True,
        policy=policy,
    ) == "covered"
    state = interpret_post_success_attempts(["accepted"], policy)
    assert (state.accepted_demos, state.attempts_used, state.remaining_attempts) == (2, 1, 2)


def test_example_b_first_two_fail_selects_same_stratum_reserve(policy):
    assert next_coverage_action(
        genuine_attempts=2, first_success=False, same_stratum_reserve_available=True,
        policy=policy,
    ) == "same_stratum_reserve"
    assert next_coverage_action(
        genuine_attempts=2, first_success=False, same_stratum_reserve_available=False,
        policy=policy,
    ) == "reserve_exhausted"


def test_example_c_duplicate_then_recovery_success(policy):
    state = interpret_post_success_attempts(["exact_duplicate", "accepted"], policy)
    assert (state.accepted_demos, state.attempts_used, state.remaining_attempts) == (2, 2, 1)


def test_example_d_pass_c_never_exceeds_three(policy):
    state = interpret_post_success_attempts(
        ["accepted", "accepted", "accepted", "accepted"], policy,
    )
    assert state.accepted_demos == 3
    assert state.attempts_used == 3
    assert state.remaining_attempts == 0


def test_example_e_quota_unmet_when_capacity_exhausted(policy):
    assert collection_terminal_status(
        accepted_trajectories=5999, eligible_capacity=0, policy=policy,
    ) == "quota_unmet"
    assert collection_terminal_status(
        accepted_trajectories=6000, eligible_capacity=0, policy=policy,
    ) == "complete"


def test_deterministic_no_progress_rule_is_exact_and_thought_independent():
    actions = [
        "Thought: one\nAction: click[p1]",
        "Thought: two\nAction: click[P1]",
        "Action: click[p1]",
    ]
    assert has_deterministic_no_progress_loop(actions, ["same", "same", "same", "same"])
    assert not has_deterministic_no_progress_loop(actions, ["same", "changed", "same", "same"])
    assert not has_deterministic_no_progress_loop(
        ["Action: click[p1]", "Action: click[p2]", "Action: click[p1]"],
        ["same", "same", "same", "same"],
    )


def test_local_canonical_p3a_no_progress_audit_when_artifacts_exist():
    selection_path = ROOT / "data" / "teacher_profile" / "p3a_canonical_selection.json"
    if not selection_path.exists():
        pytest.skip("local paid P3a artifacts are intentionally not tracked")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    records = []
    for scenario in ("single", "single_persona"):
        spec = selection[scenario]
        excluded = set(spec.get("excluded_task_ids", []))
        base = Path(spec["base_run_path"])
        records.extend(
            json.loads(path.read_text(encoding="utf-8"))
            for path in (base / "trajectories").glob("*.json")
            if json.loads(path.read_text(encoding="utf-8")).get("task_id") not in excluded
        )
        if spec.get("repair_run_path"):
            repair = Path(spec["repair_run_path"])
            records.extend(
                json.loads(path.read_text(encoding="utf-8"))
                for path in (repair / "trajectories").glob("*.json")
            )
    assert len(records) == 113
    for record in records:
        actions = record["actions"]
        pre_observations = [
            message["content"] for message in record["messages"] if message["role"] == "user"
        ][:len(actions)]
        assert len(pre_observations) == len(actions)
        assert not has_deterministic_no_progress_loop(actions, pre_observations)

import json
import hashlib
from pathlib import Path

import pytest
import scripts.run_sft_gpu_admission as admission
from scripts.prepare_sft_gpu_admission import select
from training.sft import SFT_ATTENTION_BACKEND

from scripts.run_sft_gpu_admission import (
    SELECTION_SHA256,
    canonical_hash,
    phase_plan,
    run_phase_sequence,
    validate_cuda,
    validate_dependency_version,
    validate_dataset_identity,
    validate_output_path,
    validate_success_report,
    validate_selection_payload,
)


SELECTION = Path(__file__).parents[2] / "configs/admission/single_persona_sft_gpu_admission_selection.json"


def test_selection_identity_and_phase_order_are_deterministic():
    payload = json.loads(SELECTION.read_text())
    validate_selection_payload(payload)
    plan = phase_plan(payload)
    assert payload["selection_sha256"] == SELECTION_SHA256
    assert len(plan["C"]) == 32
    assert plan["A"][0] == "b5a98c22ba2448e1917e8a391fd695c3"
    assert set(plan["A"]).isdisjoint(plan["B"])


def test_selection_hash_mismatch_fails_before_execution():
    payload = json.loads(SELECTION.read_text())
    payload["examples"] = payload["examples"][:-1]
    with pytest.raises(ValueError, match="ADMISSION_SELECTION_HASH_MISMATCH"):
        validate_selection_payload(payload)


def test_selection_algorithm_is_order_independent():
    rows = [{"accepted_id": f"a{i:03}", "task_id": f"t{i:03}", "total_tokens": 3000 + i,
             "supervised_tokens": 200 + i, "source_path": f"a{i:03}.json", "source_sha256": "h"}
            for i in range(100)]
    assert select(rows) == select(list(reversed(rows)))
    assert len(select(rows)) == 32


def test_wrong_selection_sanitizer_version_fails():
    payload = json.loads(SELECTION.read_text())
    payload["persona_sanitizer_version"] = "wrong"
    payload["selection_sha256"] = canonical_hash({k: v for k, v in payload.items()
                                                    if k != "selection_sha256"})
    monkey = admission.SELECTION_SHA256
    admission.SELECTION_SHA256 = payload["selection_sha256"]
    try:
        with pytest.raises(ValueError, match="PERSONA_SANITIZER_VERSION_MISMATCH"):
            validate_selection_payload(payload)
    finally:
        admission.SELECTION_SHA256 = monkey


def test_selection_hash_uses_canonical_payload():
    payload = json.loads(SELECTION.read_text())
    digest = payload.pop("selection_sha256")
    assert canonical_hash(payload) == digest


def test_no_gpu_fails_fast():
    class Cuda:
        device_count = staticmethod(lambda: 0)
        is_available = staticmethod(lambda: False)
        is_bf16_supported = staticmethod(lambda: False)
    class Torch:
        cuda = Cuda()
    with pytest.raises(RuntimeError, match="USER_GPU_NOT_AVAILABLE"):
        validate_cuda(Torch())


def test_cuda_torch_local_version_suffix_is_provenance_not_release_drift():
    validate_dependency_version("torch", "2.8.0+cu128", "2.8.0")
    validate_dependency_version("transformers", "4.57.6", "4.57.6")
    with pytest.raises(RuntimeError, match="PINNED_DEPENDENCY_MISMATCH:torch:2.9.0"):
        validate_dependency_version("torch", "2.9.0+cu128", "2.8.0")
    with pytest.raises(RuntimeError, match="PINNED_DEPENDENCY_MISMATCH:transformers"):
        validate_dependency_version("transformers", "4.57.6+local", "4.57.6")


def test_phase_a_failure_skips_b_and_c():
    called = []
    def fail():
        called.append("A")
        raise RuntimeError("oom")
    with pytest.raises(RuntimeError, match="oom"):
        run_phase_sequence(fail, lambda: called.append("B"), lambda: called.append("C"))
    assert called == ["A"]


def test_effective_batch_and_no_external_service_path():
    source = (Path(__file__).parents[2] / "scripts/run_sft_gpu_admission.py").read_text()
    assert '"microbatch": 1, "gradient_accumulation": 32, "effective_batch": 32' in source
    assert "teacher_env" not in source
    assert "collect_teacher" not in source
    assert SFT_ATTENTION_BACKEND == "sdpa"
    assert "attn_implementation=SFT_ATTENTION_BACKEND" in source
    assert "torch_dtype=" not in source
    formal = (Path(__file__).parents[2] / "scripts/train_sft.py").read_text()
    assert "attn_implementation=SFT_ATTENTION_BACKEND" in formal
    assert "torch_dtype=" not in formal


def test_dataset_hash_and_sanitizer_mismatch_fail(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    train.write_text("wrong")
    manifest = {"dataset_version": "single_persona_budgeted_v1", "frozen_count": 2659,
                "dataset_content_sha256": "wrong", "persona_sanitizer_version": "wrong"}
    with pytest.raises(RuntimeError, match="FROZEN_DATASET_IDENTITY_MISMATCH"):
        validate_dataset_identity(manifest, train)
    digest = hashlib.sha256(b"wrong").hexdigest()
    monkeypatch.setattr(admission, "DATASET_SHA256", digest)
    manifest.update(dataset_content_sha256=digest)
    with pytest.raises(RuntimeError, match="PERSONA_SANITIZER_VERSION_MISMATCH"):
        validate_dataset_identity(manifest, train)


def test_checkpoint_output_path_is_scoped_to_runs():
    assert admission.RUNTIME_RUNS == (admission.ROOT.parent / "runs").resolve()
    validate_output_path(admission.RUNTIME_RUNS / "bounded-admission")
    with pytest.raises(RuntimeError, match="UNSAFE_ADMISSION_OUTPUT_PATH"):
        validate_output_path(Path("/tmp/not-a-project-run"))


def test_shell_runner_appends_chronology_but_never_reuses_run_output():
    source = (Path(__file__).parents[2] / "scripts/run_sft_gpu_admission.sh").read_text()
    assert 'OUT_DIR="${OUT_DIR:-$ROOT/../runs/' in source
    assert "tee -a \"$LOG\"" in source
    assert "LOG_ALREADY_EXISTS_REFUSING_OVERWRITE" not in source
    assert "OUTPUT_DIR_ALREADY_EXISTS_REFUSING_OVERWRITE" in source


def test_success_report_schema_requires_real_update_fields():
    report = {key: {} for key in ("candidate_runtime", "phase_a", "phase_b", "checkpoint")}
    report.update(status="PASS", dataset_version="v", dataset_sha256="h",
                  selection_sha256="s", sanitizer_version="p",
                  phase_c={"examples_processed": 32, "optimizer_step": 1,
                           "effective_batch": 32, "lora_changed": True})
    validate_success_report(report)
    report["phase_c"]["lora_changed"] = False
    with pytest.raises(RuntimeError, match="ADMISSION_REPORT_PHASE_C_INCOMPLETE"):
        validate_success_report(report)

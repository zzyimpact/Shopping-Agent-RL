import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from training.runtime import load_config, prepare_run
from training.sft import SFTConfigSpec, build_sft_trainer, train_sft


ROOT = Path(__file__).resolve().parents[2]


def test_defaults_keep_lora_unresolved_and_small_override(tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text("scenario: single_persona\nsft:\n  lora_r: 16\n")
    config = load_config(override)
    assert config["seed"] == 1 and config["max_action_steps"] == 30
    assert config["sft"]["num_train_epochs"] == 4
    assert config["sft"]["lora_r"] == 16
    assert config["sft"]["per_device_train_batch_size"] is None
    with pytest.raises(ValueError, match="GPU-PREFLIGHT"):
        SFTConfigSpec(output_dir=tmp_path).validate()


def test_run_identity_resume_and_no_overwrite(tmp_path):
    output = tmp_path / "run"
    config = {"scenario": "single", "mode": "sft", "seed": 1}
    root = prepare_run(output, config=config, inputs={"selection": "hash-a"})
    checkpoint = root / "checkpoints" / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}")
    prepare_run(output, config=config, inputs={"selection": "hash-a"}, resume_from_checkpoint=checkpoint)
    assert len((root / "metrics.jsonl").read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        prepare_run(output, config=config, inputs={"selection": "hash-a"})
    with pytest.raises(ValueError, match="identity"):
        prepare_run(output, config={**config, "scenario": "single_persona"}, inputs={"selection": "hash-a"},
                    resume_from_checkpoint=checkpoint)
    with pytest.raises(ValueError, match="identity"):
        prepare_run(output, config=config, inputs={"selection": "hash-b"}, resume_from_checkpoint=checkpoint)


def test_sft_construction_bf16_peft_labels_and_hf_resume_passthrough(tmp_path, monkeypatch):
    captured = {}

    def config_factory(**kwargs):
        captured["args"] = kwargs
        return SimpleNamespace(**kwargs)

    class Trainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs
            self.args = kwargs["args"]

        def train(self, **kwargs):
            captured["train"] = kwargs
            return "trained-by-hf-stub"

        def save_model(self, path):
            captured["saved"] = path

    def lora(**kwargs):
        captured["lora"] = kwargs
        return kwargs

    monkeypatch.setattr("training.sft._training_imports", lambda: (config_factory, Trainer, lora, object))
    spec = SFTConfigSpec(output_dir=tmp_path, per_device_train_batch_size=2,
                         gradient_accumulation_steps=16, lora_r=8, lora_alpha=16, target_modules=("q_proj",))
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    rows = [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}]
    trainer = build_sft_trainer(model=model, tokenizer=object(), train_dataset=rows, config=spec)
    assert captured["args"]["bf16"] is True
    assert captured["args"]["dataset_kwargs"] == {"skip_prepare_dataset": True}
    assert captured["trainer"]["train_dataset"] == rows
    assert captured["lora"]["task_type"] == "CAUSAL_LM"
    assert not model.config.use_cache
    assert train_sft(trainer, resume_from_checkpoint="checkpoint-100") == "trained-by-hf-stub"
    assert captured["train"] == {"resume_from_checkpoint": "checkpoint-100"}
    with pytest.raises(ValueError, match="labels"):
        build_sft_trainer(model=model, tokenizer=None, train_dataset=[{"input_ids": [1]}], config=spec)


def test_cli_config_and_reward_exclusivity():
    script = runpy.run_path(str(ROOT / "scripts/eval_policy.py"))
    base = ["--scenario", "single", "--model-path", "model", "--manifest", "test.json", "--output-dir", "run"]
    assert script["parse_config"](base + ["--reward", "strict"])["reward_alpha"] == 1
    assert script["parse_config"](base + ["--reward", "loose"])["reward_alpha"] == 0
    assert script["parse_config"](base + ["--alpha", "0.5"])["reward_alpha"] == 0.5
    with pytest.raises(SystemExit):
        script["parse_config"](base + ["--reward", "strict", "--alpha", "0.5"])


def test_sft_cli_requires_explicit_lora_not_full_training():
    script = runpy.run_path(str(ROOT / "scripts/train_sft.py"))
    args = ["--scenario", "single", "--model-path", "model", "--selection-manifest", "selection.json",
            "--accepted-root", "accepted", "--train-task-manifest", "train.json", "--output-dir", "run"]
    with pytest.raises(ValueError, match="LoRA"):
        script["parse_config"](args)
    config, spec, resume = script["parse_config"](args + ["--per-device-train-batch-size", "2",
        "--gradient-accumulation-steps", "16", "--lora-r", "8", "--lora-alpha", "16",
        "--target-modules", "q_proj", "v_proj", "--resume-from-checkpoint", "checkpoint-100"])
    assert config["sft"]["effective_batch_size"] == 32 and resume == "checkpoint-100"

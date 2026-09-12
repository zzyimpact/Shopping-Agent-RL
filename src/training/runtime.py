"""Small shared config and run-file helpers, independent of teacher storage."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from copy import deepcopy
from pathlib import Path
import subprocess
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("torch", "transformers", "trl", "peft", "accelerate", "datasets")


def load_config(override: str | Path | None = None) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load((PROJECT_ROOT / "configs/training/defaults.yaml").read_text())
    project = yaml.safe_load((PROJECT_ROOT / "configs/spec/project.yaml").read_text())["project"]
    config.update(seed=project["seed"], max_action_steps=project["max_action_steps"])
    if override:
        for key, value in yaml.safe_load(Path(override).read_text()).items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value
    return config


def sha256_file(path: str | Path) -> str:
    # Only small manifests/tokenizer configs/artifacts. Never called on model weights.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def dependency_versions() -> dict[str, str | None]:
    result = {}
    for name in PACKAGES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def append_metrics(path: str | Path, values: Mapping[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(values), ensure_ascii=False, allow_nan=False) + "\n")


def prepare_run(
    output_dir: str | Path, *, config: Mapping[str, Any], inputs: Mapping[str, Any],
    resume_from_checkpoint: str | Path | None = None,
    allow_admission_extension: bool = False,
) -> Path:
    """Record identity once; HF owns checkpoint contents, optimizer and RNG restore."""
    root = Path(output_dir).resolve()
    identity = {"config": dict(config), "inputs": dict(inputs)}
    manifest_path = root / "run_manifest.json"
    checkpoint = Path(resume_from_checkpoint).resolve() if resume_from_checkpoint else None
    if checkpoint:
        if not checkpoint.is_dir() or checkpoint.parent != root / "checkpoints":
            raise ValueError("resume checkpoint must belong to this run's checkpoints directory")
        if not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("resume requires an HF training checkpoint, not the final adapter export")
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != identity:
            if not allow_admission_extension:
                raise ValueError("resume scenario/model/config/dataset identity mismatch")
            old = previous["identity"]
            old_config, new_config = deepcopy(old["config"]), deepcopy(identity["config"])
            old_inputs, new_inputs = deepcopy(old["inputs"]), deepcopy(identity["inputs"])
            old_steps = int(old_config.get("grpo", {}).get("max_steps", 0))
            new_steps = int(new_config.get("grpo", {}).get("max_steps", 0))
            old_config.get("grpo", {}).pop("max_steps", None)
            new_config.get("grpo", {}).pop("max_steps", None)
            for candidate in (old_config, new_config):
                if candidate.get("admission_mode") is not True:
                    raise ValueError("admission resume requires --admission on both phases")
            if old_config != new_config or new_steps <= old_steps:
                raise ValueError("admission resume only permits increasing grpo.max_steps")
            old_schedule_hash = old_inputs.pop("task_schedule_sha256", None)
            old_groups = old_inputs.pop("scheduled_groups", None)
            old_inputs.pop("config_sha256", None)
            new_inputs.pop("task_schedule_sha256", None)
            new_inputs.pop("scheduled_groups", None)
            new_inputs.pop("config_sha256", None)
            prefix_hash = new_inputs.pop("task_schedule_prefix_sha256", None)
            old_inputs.pop("task_schedule_prefix_sha256", None)
            if (old_inputs != new_inputs or old_schedule_hash != prefix_hash
                    or not isinstance(old_groups, int) or old_groups >= identity["inputs"].get("scheduled_groups", 0)):
                raise ValueError("admission schedule prefix/config identity mismatch")
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError("output_dir is not empty; select a new run or explicit checkpoint resume")
    for name in ("checkpoints", "eval"):
        (root / name).mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                         cwd=PROJECT_ROOT, text=True).strip())
    invocation = {"timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
                  "git_dirty": dirty, "dependencies": dependency_versions(),
                  "resume_from_checkpoint": str(checkpoint) if checkpoint else None}
    if not checkpoint:
        manifest = {"identity": identity, **invocation,
                    "upstream_reference_commit": config.get("upstream_reference_commit"),
                    "upstream_reference_note": "audited reference pin, not a verified local checkout commit"}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    append_metrics(root / "metrics.jsonl", {"event": "resume" if checkpoint else "start", **invocation})
    return root


def model_metadata(model_path: str | Path, tokenizer_path: str | Path) -> dict[str, Any]:
    """Fingerprint small configuration files only; never load/checksum weight shards."""
    result: dict[str, Any] = {"model_path": str(Path(model_path).resolve()),
                              "tokenizer_path": str(Path(tokenizer_path).resolve())}
    for name, root, filename in (("model_config", model_path, "config.json"),
                                 ("model_index", model_path, "model.safetensors.index.json"),
                                 ("tokenizer_config", tokenizer_path, "tokenizer_config.json")):
        path = Path(root) / filename
        result[name + "_sha256"] = sha256_file(path) if path.is_file() else None
    return result

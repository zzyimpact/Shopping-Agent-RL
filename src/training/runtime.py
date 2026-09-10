"""Small shared config and run-file helpers, independent of teacher storage."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
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
            raise ValueError("resume scenario/model/config/dataset identity mismatch")
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

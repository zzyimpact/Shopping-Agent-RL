"""Read-only projection of accepted teacher artifacts for SFT."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rollout.prompt import assert_no_evaluator_leakage


ALLOWED_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class SFTExample:
    accepted_id: str
    task_id: str
    messages: list[dict[str, str]]
    source_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted_id": self.accepted_id,
            "task_id": self.task_id,
            "messages": [dict(message) for message in self.messages],
        }


def project_messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """Project only stored policy-visible messages; never stringify the artifact."""
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("accepted artifact missing non-empty messages")
    messages: list[dict[str, str]] = []
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, Mapping):
            raise ValueError(f"message {index} is not an object")
        role = raw.get("role")
        content = raw.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"message {index} has unsupported role: {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"message {index} content must be text")
        messages.append({"role": str(role), "content": content})
    if messages[0]["role"] != "system" or len(messages) < 3:
        raise ValueError("expected system, user, assistant conversation")
    for index, message in enumerate(messages[1:], 1):
        if message["role"] != ("user" if index % 2 else "assistant"):
            raise ValueError("expected alternating user observation / assistant turns")
    visible = record.get("visible_responses")
    if visible is not None and visible != [m["content"] for m in messages if m["role"] == "assistant"]:
        raise ValueError("stored assistant messages differ from visible_responses")
    assert_no_evaluator_leakage(messages)
    return messages


def _manifest_entries(value: Any) -> tuple[str | None, list[Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        scenario = str(value["scenario"]) if value.get("scenario") is not None else None
        entries = value.get("artifacts")
    else:
        scenario, entries = None, value
    if not isinstance(entries, list):
        raise ValueError("selection manifest must contain an artifacts list")
    normalized: list[Mapping[str, Any]] = []
    for entry in entries:
        if isinstance(entry, str):
            normalized.append({"path": entry})
        elif isinstance(entry, Mapping):
            normalized.append(entry)
        else:
            raise ValueError("selection manifest artifact entry must be an object or path")
    return scenario, normalized


def _safe_artifact_path(accepted_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ValueError("selection manifest paths must be relative to accepted_root")
    root = accepted_root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("selection manifest path escapes accepted_root")
    if resolved.parent != root:
        raise ValueError("selection manifest path must point directly inside accepted_root")
    if resolved.suffix != ".json":
        raise ValueError("accepted artifact path must be a .json file")
    return resolved


def load_selected_examples(
    selection_manifest: str | Path,
    *,
    accepted_root: str | Path,
    scenario: str | None = None,
) -> list[SFTExample]:
    """Load exactly the artifacts named by a frozen selection manifest."""
    manifest_path = Path(selection_manifest)
    manifest_scenario, entries = _manifest_entries(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if scenario is not None and manifest_scenario is not None and manifest_scenario != scenario:
        raise ValueError("selection manifest scenario mismatch")
    scenario = scenario or manifest_scenario
    if scenario not in {"single", "single_persona"}:
        raise ValueError("selection requires scenario single or single_persona")
    if not entries:
        raise ValueError("selection must not be empty")
    root = Path(accepted_root)
    examples: list[SFTExample] = []
    seen: set[str] = set()
    for entry in entries:
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("selection manifest entry missing path")
        record_path = _safe_artifact_path(root, raw_path)
        raw = record_path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        if entry.get("sha256") is not None and entry["sha256"] != source_hash:
            raise ValueError("selected artifact hash mismatch")
        record = json.loads(raw)
        if not isinstance(record, Mapping):
            raise ValueError(f"accepted artifact is not an object: {record_path}")
        artifact_scenario = record.get("scenario")
        if artifact_scenario != scenario:
            raise ValueError(f"accepted artifact scenario mismatch: {record_path}")
        accepted_id = str(record.get("accepted_id") or "")
        task_id = str(record.get("task_id") or "")
        if not accepted_id or not task_id:
            raise ValueError(f"accepted artifact missing identity: {record_path}")
        for key in ("accepted_id", "task_id"):
            if entry.get(key) is not None and str(entry[key]) != str(record[key]):
                raise ValueError(f"selected {key} does not match artifact")
        if record.get("accepted") is False:
            raise ValueError("explicitly unaccepted artifact")
        if accepted_id in seen:
            raise ValueError(f"duplicate accepted_id in selection manifest: {accepted_id}")
        seen.add(accepted_id)
        examples.append(SFTExample(accepted_id, task_id, project_messages(record), source_hash))
    return examples


def assistant_turn_mask(messages: Sequence[Mapping[str, str]]) -> list[bool]:
    """Return per-message loss participation: only assistant turns train."""
    return [message.get("role") == "assistant" for message in messages]


def tokenize_with_assistant_mask(
    tokenizer: Any, messages: Sequence[Mapping[str, str]], *, max_length: int = 32768,
) -> dict[str, list[int]]:
    """Qwen3 text-only assistant labels, with exact native-template verification.

    We use character offsets from the fast tokenizer, not token-length deltas
    of prefixes (Qwen's template can rewrite prefixes). Template-added headers
    and whitespace remain masked. The assistant's im_end is trained as EOS.
    Unsupported template rendering fails explicitly instead of guessing a mask.
    """
    rendered = tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=False,
    )
    parts, spans = [], []
    offset = 0
    for message in messages:
        header = f"<|im_start|>{message['role']}\n"
        content = message["content"]
        block = header + content + "<|im_end|>\n"
        if message["role"] == "assistant":
            spans.append((offset + len(header), offset + len(block) - 1))
        parts.append(block)
        offset += len(block)
    if rendered != "".join(parts):
        raise ValueError("Qwen chat template changed visible text; TRAINING-ENV PREFLIGHT REQUIRED")
    if not spans:
        raise ValueError("SFT sample has no assistant loss tokens")
    # Render the original history first: Qwen can rewrite the last assistant if
    # we remove the terminal user before rendering. Only the training sequence
    # ends at the last assistant EOS; source/review messages remain untouched.
    rendered = rendered[:spans[-1][1]]
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    full_ids = tokenizer.apply_chat_template(list(messages), tokenize=True, add_generation_prompt=False)
    if input_ids != full_ids[:len(input_ids)]:
        raise ValueError("chat template/tokenizer IDs disagree")
    if len(input_ids) > max_length:
        raise ValueError("SFT conversation exceeds max_length; select/review data instead of silent truncation")
    labels = [-100] * len(input_ids)
    for index, (start, end) in enumerate(encoded["offset_mapping"]):
        if end > start and any(lo <= start < end <= hi for lo, hi in spans):
            labels[index] = input_ids[index]
        elif any(start < hi and end > lo for lo, hi in spans):
            raise ValueError("token straddles assistant mask boundary; TRAINING-ENV PREFLIGHT REQUIRED")
    if not any(label != -100 for label in labels):
        raise ValueError("SFT sample has no assistant loss tokens")
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if sum(label == eos for label in labels) != len(spans):
        raise ValueError("assistant EOS mask missing; TRAINING-ENV PREFLIGHT REQUIRED")
    return {"input_ids": input_ids, "labels": labels}


def write_sft_jsonl(examples: Sequence[SFTExample], output: str | Path) -> None:
    """Export reviewable conversations offline; caller chooses the selection/freeze."""
    with Path(output).open("x", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.as_dict(), ensure_ascii=False) + "\n")

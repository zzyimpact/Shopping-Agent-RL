import hashlib
import json
import re

import pytest

from training.sft_data import (
    assistant_turn_mask, load_selected_examples, project_messages,
    tokenize_with_assistant_mask, write_sft_jsonl,
)


def record():
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "可见需求"},
                {"role": "assistant", "content": "Thought: search\nAction: search[shoes]"},
                {"role": "user", "content": "可见商品与选项"},
                {"role": "assistant", "content": "Thought: buy\nAction: click[Buy]"},
                {"role": "user", "content": "购买完成"}]
    return {"accepted_id": "a", "task_id": "t", "scenario": "single", "messages": messages,
            "evaluator_only": {"goal": {"asin": "HIDDEN_TARGET", "attributes": ["HIDDEN_GOLD"]}},
            "reward_metrics": {"r_strict": 1}, "terminal_purchase": {"asin": "HIDDEN_PURCHASE"}}


def selected(tmp_path, data=None, entry=None):
    root = tmp_path / "accepted"
    root.mkdir()
    artifact = root / "a.json"
    artifact.write_text(json.dumps(data or record()))
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({"scenario": "single", "artifacts": [entry or {"path": "a.json"}]}))
    return root, manifest


@pytest.mark.parametrize("current", [False, True])
def test_old_and_current_records_visible_only_projection(tmp_path, current):
    data = record()
    if current:
        data.update(accepted=True, visible_responses=[m["content"] for m in data["messages"] if m["role"] == "assistant"],
                    api_diagnostics=[{"irrelevant": "HIDDEN_DIAGNOSTICS"}], formal_work={"pass": "A"})
    root, manifest = selected(tmp_path, data)
    examples = load_selected_examples(manifest, accepted_root=root, scenario="single")
    assert examples[0].messages == data["messages"]
    assert assistant_turn_mask(examples[0].messages) == [False, False, True, False, True, False]
    assert examples[0].source_sha256 == hashlib.sha256((root / "a.json").read_bytes()).hexdigest()
    out = tmp_path / "sft.jsonl"
    write_sft_jsonl(examples, out)
    assert "HIDDEN" not in out.read_text() and "reward_metrics" not in out.read_text()
    assert json.loads(out.read_text())["messages"] == data["messages"]
    with pytest.raises(FileExistsError):
        write_sft_jsonl(examples, out)


@pytest.mark.parametrize("entry,match", [({"path": "../secret.json"}, "escapes"),
                                        ({"path": "a.json", "accepted_id": "wrong"}, "accepted_id"),
                                        ({"path": "a.json", "sha256": "wrong"}, "hash")])
def test_selection_integrity(tmp_path, entry, match):
    root, manifest = selected(tmp_path, entry=entry)
    with pytest.raises(ValueError, match=match):
        load_selected_examples(manifest, accepted_root=root)


def test_scenario_is_checked_even_when_only_manifest_supplies_it(tmp_path):
    data = record()
    data["scenario"] = "single_persona"
    root, manifest = selected(tmp_path, data)
    with pytest.raises(ValueError, match="scenario"):
        load_selected_examples(manifest, accepted_root=root)


def test_visible_response_mismatch_and_leakage_are_rejected():
    data = record()
    data["visible_responses"] = ["substituted answer"]
    with pytest.raises(ValueError, match="visible_responses"):
        project_messages(data)
    del data["visible_responses"]
    data["messages"][1]["content"] = "evaluator_only: secret"
    with pytest.raises(AssertionError, match="leaked"):
        project_messages(data)


class FakeQwenTokenizer:
    """Fast-tokenizer shape only: UTF-8 characters plus atomic ChatML markers."""
    @staticmethod
    def convert_tokens_to_ids(token):
        return {"<|im_start|>": 200000, "<|im_end|>": 200001}[token]

    def __call__(self, text, **kwargs):
        pieces = list(re.finditer(r"<\|im_start\|>|<\|im_end\|>|[\s\S]", text))
        ids = [self.convert_tokens_to_ids(m[0]) if len(m[0]) > 1 else ord(m[0]) for m in pieces]
        return {"input_ids": ids, "offset_mapping": [(m.start(), m.end()) for m in pieces]}

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return self(text)["input_ids"] if tokenize else text


def test_labels_train_assistant_content_and_eos_only():
    messages = project_messages(record())
    tokenizer = FakeQwenTokenizer()
    row = tokenize_with_assistant_mask(tokenizer, messages)
    trained = [label for label in row["labels"] if label != -100]
    expected = [token for m in messages if m["role"] == "assistant"
                for token in tokenizer(m["content"] + "<|im_end|>")["input_ids"]]
    assert trained == expected
    assert row["labels"][0] == -100  # Includes headers, all observations and trailing terminal user.
    with pytest.raises(ValueError, match="silent truncation"):
        tokenize_with_assistant_mask(tokenizer, messages, max_length=5)


def test_changed_template_fails_instead_of_guessing_a_mask():
    class RewritingTokenizer(FakeQwenTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return super().apply_chat_template(messages, **kwargs) + "rewritten"
    with pytest.raises(ValueError, match="PREFLIGHT"):
        tokenize_with_assistant_mask(RewritingTokenizer(), record()["messages"])

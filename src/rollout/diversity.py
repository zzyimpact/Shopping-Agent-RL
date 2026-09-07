"""Explainable behavioural fingerprints for P3a profiling only.

No embedding model, extra LLM, or formal acceptance threshold is used here.
Thought wording is deliberately excluded from the fingerprint.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping


_SPACE = re.compile(r"\s+")
_ACTION = re.compile(r"^\s*(search|click)\s*\[(.*)\]\s*$", re.I | re.S)


def normalize_text(value: Any) -> str:
    return _SPACE.sub(" ", str(value or "")).strip().lower()


def normalize_action(value: Any) -> tuple[str, str]:
    raw = str(value or "")
    action_lines = [line.split(":", 1)[1].strip() for line in raw.splitlines()
                    if line.strip().lower().startswith("action:")]
    if action_lines:
        raw = action_lines[-1]
    match = _ACTION.match(raw)
    if not match:
        return ("invalid", normalize_text(value))
    return (match.group(1).lower(), normalize_text(match.group(2)))


def behavior_fingerprint(actions: Iterable[Any], *, final_purchase_asin: str | None = None) -> dict[str, Any]:
    normalized = [normalize_action(action) for action in actions]
    return {
        "action_types": [kind for kind, _ in normalized],
        "search_queries": [arg for kind, arg in normalized if kind == "search"],
        "clicked_products": [arg for kind, arg in normalized if kind == "click" and _looks_like_asin(arg)],
        "selected_options": [arg for kind, arg in normalized if kind == "click" and not _looks_like_asin(arg)],
        "final_purchase_asin": normalize_text(final_purchase_asin) if final_purchase_asin else None,
        "action_length": len(normalized),
        "normalized_actions": [[kind, arg] for kind, arg in normalized],
    }


def _looks_like_asin(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9]{8,20}", value)) and any(ch.isdigit() for ch in value)


def exact_duplicate(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = ("action_types", "search_queries", "clicked_products", "selected_options",
            "final_purchase_asin", "action_length", "normalized_actions")
    return all(left.get(key) == right.get(key) for key in keys)


def similarity_features(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    def ratio(key: str) -> float:
        left_seq = [repr(item) for item in left.get(key, [])]
        right_seq = [repr(item) for item in right.get(key, [])]
        return SequenceMatcher(None, left_seq, right_seq).ratio()
    features = {
        "action_types_equal": left.get("action_types") == right.get("action_types"),
        "clicked_products_equal": left.get("clicked_products") == right.get("clicked_products"),
        "selected_options_equal": left.get("selected_options") == right.get("selected_options"),
        "search_query_similarity": ratio("search_queries"),
        "normalized_action_similarity": ratio("normalized_actions"),
        "action_length_difference": abs(int(left.get("action_length", 0)) - int(right.get("action_length", 0))),
    }
    # Provisional, explainable flag only.  P3b must decide any formal rule.
    features["provisional_near_duplicate"] = bool(
        features["action_types_equal"] and features["normalized_action_similarity"] >= 0.8
    )
    return features

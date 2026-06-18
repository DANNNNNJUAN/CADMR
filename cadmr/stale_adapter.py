"""STALE dataset compatibility helpers.

This module is an evaluation-layer adapter. It does not change CADMR method logic;
it only normalizes STALE-style JSON records into inputs that can be replayed through
the public CADMR pipeline interface.
"""

from pathlib import Path
import json
from typing import Any


class StaleDatasetLoader:
    """Loads STALE *_MAIN.json files."""

    def load(self, path: str | Path) -> list[dict]:
        dataset_path = Path(path)
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("STALE MAIN file must contain a JSON list.")
        return [normalize_stale_sample(sample, index) for index, sample in enumerate(data)]


def normalize_stale_sample(sample: dict, index: int) -> dict:
    """Normalize one STALE sample while preserving the raw record."""
    if not isinstance(sample, dict):
        sample = {}

    return {
        "uid": sample.get("uid") or f"sample_{index}",
        "M_old": sample.get("M_old", ""),
        "M_new": sample.get("M_new", ""),
        "explanation": sample.get("explanation", ""),
        "haystack_session": sample.get("haystack_session", []),
        "probing_queries": sample.get("probing_queries", {}),
        "raw": sample,
    }


def flatten_haystack_user_turns(sample: dict) -> list[str]:
    """Extract all user turns from STALE haystack_session variants."""
    haystack = sample.get("haystack_session", []) if isinstance(sample, dict) else []
    turns: list[str] = []

    for message in _iter_messages(haystack):
        role = _message_role(message)
        content = _message_content(message)
        if role == "user" and content:
            turns.append(content)

    return turns


def get_dim_queries(sample: dict) -> dict[str, str | None]:
    """Return dim1/dim2/dim3 probing queries from common STALE aliases."""
    probing = sample.get("probing_queries", {}) if isinstance(sample, dict) else {}
    if not isinstance(probing, dict):
        probing = {}

    return {
        "dim1": _first_string(probing, ["dim1_query", "SR", "sr_query"]),
        "dim2": _first_string(probing, ["dim2_query", "PR", "pr_query"]),
        "dim3": _first_string(probing, ["dim3_query", "IPA", "ipa_query"]),
    }


def _iter_messages(node: Any):
    if isinstance(node, list):
        for item in node:
            yield from _iter_messages(item)
        return

    if not isinstance(node, dict):
        return

    for container_key in ["messages", "turns", "conversation"]:
        child = node.get(container_key)
        if child is not None:
            yield from _iter_messages(child)
            return

    if _message_role(node) or _message_content(node):
        yield node


def _message_role(message: dict) -> str:
    role = message.get("role", message.get("speaker", ""))
    return str(role).strip().lower()


def _message_content(message: dict) -> str:
    content = message.get("content", message.get("text", ""))
    if not isinstance(content, str):
        return ""
    return content.strip()


def _first_string(mapping: dict, keys: list[str]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None

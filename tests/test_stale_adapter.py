import json

import pytest

from cadmr.stale_adapter import (
    StaleDatasetLoader,
    flatten_haystack_user_turns,
    get_dim_queries,
    normalize_stale_sample,
)


def test_loader_requires_top_level_list(tmp_path):
    path = tmp_path / "bad_MAIN.json"
    path.write_text(json.dumps({"uid": "x"}), encoding="utf-8")

    with pytest.raises(ValueError):
        StaleDatasetLoader().load(path)


def test_normalize_sample_fills_defaults_and_preserves_raw():
    raw = {"M_old": "old"}

    sample = normalize_stale_sample(raw, 3)

    assert sample["uid"] == "sample_3"
    assert sample["M_old"] == "old"
    assert sample["M_new"] == ""
    assert sample["explanation"] == ""
    assert sample["haystack_session"] == []
    assert sample["raw"] == raw


def test_flatten_haystack_user_turns_supports_nested_sessions():
    sample = {
        "haystack_session": [
            [
                {"role": "user", "content": "第一轮用户内容"},
                {"role": "assistant", "content": "第一轮助手内容"},
            ],
            {
                "messages": [
                    {"speaker": "user", "text": "第二轮用户内容"},
                    {"speaker": "assistant", "text": "第二轮助手内容"},
                ]
            },
        ]
    }

    assert flatten_haystack_user_turns(sample) == ["第一轮用户内容", "第二轮用户内容"]


def test_flatten_haystack_user_turns_supports_flat_messages():
    sample = {
        "haystack_session": [
            {"role": "user", "content": "flat user"},
            {"role": "assistant", "content": "flat assistant"},
        ]
    }

    assert flatten_haystack_user_turns(sample) == ["flat user"]


def test_get_dim_queries_supports_aliases():
    sample = {"probing_queries": {"SR": "sr", "pr_query": "pr", "IPA": "ipa"}}

    assert get_dim_queries(sample) == {"dim1": "sr", "dim2": "pr", "dim3": "ipa"}


def test_loader_normalizes_samples(tmp_path):
    path = tmp_path / "sample_MAIN.json"
    path.write_text(json.dumps([{"uid": "u1"}], ensure_ascii=False), encoding="utf-8")

    samples = StaleDatasetLoader().load(path)

    assert samples[0]["uid"] == "u1"
    assert samples[0]["raw"] == {"uid": "u1"}

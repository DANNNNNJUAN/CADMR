import json
from io import StringIO
from types import SimpleNamespace

import scripts.run_stale_eval as runner
from scripts.run_stale_eval import (
    CachedLLMClient,
    CachedSignalExtractor,
    ConditionalAnswerVerifier,
    HaystackExtractorCache,
    LimitedMemoryRetriever,
    ProgressBar,
    QueryIntentFallbackExtractor,
    normalize_haystack_text,
    rank_constraints_for_judge,
    rank_memories_by_recency,
    rank_memories_for_judge,
    run_stale_dataset,
    run_stale_sample,
)
from cadmr.schemas import MemorySignal


def test_result_meta_promotes_judge_diagnostics_from_structured_output():
    result = SimpleNamespace(
        query_info=None,
        signals=[],
        write_decisions=[],
        judgments=[],
        goal_plan=None,
        verify_result=None,
        structured_output={
            "version": 1,
            "judge_diagnostics": {
                "judge_type": "LLMUsabilityJudge",
                "batches_failed": 1,
                "fallback_used": True,
            },
        },
    )

    meta = runner._result_meta("query", result)

    assert meta["judge_diagnostics"]["judge_type"] == "LLMUsabilityJudge"
    assert meta["judge_diagnostics"]["batches_failed"] == 1
    assert meta["structured_output"]["judge_diagnostics"]["fallback_used"] is True


def make_sample():
    return {
        "uid": "sample_001",
        "M_old": "旧记忆",
        "M_new": "新记忆",
        "explanation": "测试解释",
        "haystack_session": [
            [
                {"role": "user", "content": "我喜欢重辣口味，尤其喜欢川菜和火锅。"},
                {"role": "assistant", "content": "好的。"},
            ],
            [
                {"role": "user", "content": "医生说我接下来四周不能吃辣。"},
                {"role": "assistant", "content": "知道了。"},
            ],
        ],
        "probing_queries": {
            "dim1_query": "我现在还能吃重辣火锅吗？",
            "dim3_query": "帮我安排今晚吃什么。",
        },
    }


def test_run_stale_sample_outputs_stale_compatible_shape():
    result = run_stale_sample(make_sample())

    assert result["uid"] == "sample_001"
    assert set(result["target_model_responses"]) == {
        "dim1_response",
        "dim2_response",
        "dim3_response",
    }
    assert set(result["target_model_meta"]) == {
        "dim1_meta",
        "dim2_meta",
        "dim3_meta",
    }
    assert result["target_model_responses"]["dim2_response"] == ""
    assert result["target_model_meta"]["dim2_meta"]["skipped"] is True
    assert result["cadmr_eval_meta"]["haystack_summary"]["user_turn_count"] == 2


def test_trace_reports_stale_target_retrieval_and_judgment_status():
    result = {
        "uid": "sample_001",
        "cadmr_eval_meta": {
            "m_old": "User has been taking the subway to work every morning.",
            "m_new": "User now drives to work instead of taking the subway.",
            "haystack_summary": {
                "ordinary_memory_count": 2,
                "active_constraint_count": 0,
            },
        },
        "target_model_responses": {
            "dim1_response": "Yes, the user still takes the subway.",
            "dim2_response": "",
            "dim3_response": "",
        },
        "target_model_meta": {
            "dim1_meta": {
                "query": "Based on the conversation history, does the user still take the subway to work every morning?",
                "structured_output": {
                    "retrieved_memories": [
                        {
                            "memory_id": "m_old",
                            "content": "User has been taking the subway to work every morning.",
                            "scope": ["transport"],
                        },
                        {
                            "memory_id": "m_new",
                            "content": "User now drives to work instead of taking the subway.",
                            "scope": ["transport"],
                        },
                    ],
                    "retrieved_constraints": [],
                    "judgments": [
                        {
                            "memory_id": "m_old",
                            "usage_status": "USABLE",
                            "reason": "Old memory supports the query.",
                            "blocked_by": [],
                            "replaced_by": [],
                        },
                        {
                            "memory_id": "m_new",
                            "usage_status": "USABLE",
                            "reason": "New memory is relevant.",
                            "blocked_by": [],
                            "replaced_by": [],
                        },
                    ],
                    "signals": [],
                    "write_decisions": [],
                    "verify_result": {"pass": True, "violations": [], "reason": ""},
                },
            }
        },
    }

    trace = runner._trace_sample(result)

    targets = trace["dims"]["dim1"]["retrieval"]["stale_targets"]
    assert targets["m_old"]["matched"] is True
    assert targets["m_old"]["usage_status"] == "USABLE"
    assert targets["m_new"]["matched"] is True
    assert targets["m_new"]["usage_status"] == "USABLE"
    assert trace["dims"]["dim1"]["likely_problem"] == "judge_missed_stale_update"


def test_run_stale_sample_reuses_haystack_snapshot_for_dim_queries(monkeypatch):
    calls = []

    class FakePipeline:
        def __init__(self, label):
            self.label = label

        def run(self, text):
            calls.append((self.label, text))
            return SimpleNamespace(
                answer=f"answer: {text}",
                query_info=None,
                signals=[],
                write_decisions=[],
                judgments=[],
                goal_plan=None,
                verify_result=None,
                structured_output={"version": 1},
            )

    def fake_make_pipeline(
        store_dir,
        use_llm_extractor=False,
        llm_client=None,
        use_llm_usability_judge=False,
        llm_usability_judge_client=None,
        use_llm_answer_generator=False,
        llm_answer_generator_client=None,
        use_llm_verifier=False,
        llm_verifier_client=None,
        max_judge_memories=None,
        force_query_intent=False,
        haystack_extractor_cache=None,
        **kwargs,
    ):
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "raw_interaction_log.jsonl").write_text("", encoding="utf-8")
        (store_dir / "ordinary_memory.json").write_text("[]\n", encoding="utf-8")
        (store_dir / "active_constraints.json").write_text("[]\n", encoding="utf-8")
        return FakePipeline(store_dir.name)

    monkeypatch.setattr(runner, "make_pipeline", fake_make_pipeline)

    result = run_stale_sample(make_sample())

    dim1_meta = result["target_model_meta"]["dim1_meta"]
    dim3_meta = result["target_model_meta"]["dim3_meta"]

    assert dim1_meta["query"] == "我现在还能吃重辣火锅吗？"
    assert dim3_meta["query"] == "帮我安排今晚吃什么。"
    labels = [label for label, _text in calls]
    assert labels.count("haystack") == 2
    assert labels.count("dim1") == 1
    assert labels.count("dim3") == 1
    assert "dim2" not in labels


def test_run_stale_sample_passes_stale_target_context_only_to_dim_pipelines(monkeypatch):
    contexts = []

    class FakePipeline:
        def run(self, text):
            return SimpleNamespace(
                answer=f"answer: {text}",
                query_info=None,
                signals=[],
                write_decisions=[],
                judgments=[],
                goal_plan=None,
                verify_result=None,
                structured_output={"version": 1},
            )

    def fake_make_pipeline(
        store_dir,
        stale_target_context=None,
        **kwargs,
    ):
        contexts.append((store_dir.name, stale_target_context))
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "raw_interaction_log.jsonl").write_text("", encoding="utf-8")
        (store_dir / "ordinary_memory.json").write_text("[]\n", encoding="utf-8")
        (store_dir / "active_constraints.json").write_text("[]\n", encoding="utf-8")
        return FakePipeline()

    monkeypatch.setattr(runner, "make_pipeline", fake_make_pipeline)

    run_stale_sample(make_sample())

    assert ("haystack", None) in contexts
    dim_contexts = {
        label: context
        for label, context in contexts
        if label.startswith("dim")
    }
    assert dim_contexts["dim1"] == {"m_old": "旧记忆", "m_new": "新记忆"}
    assert dim_contexts["dim3"] == {"m_old": "旧记忆", "m_new": "新记忆"}


def test_run_stale_sample_keeps_haystack_replay_write_only(monkeypatch):
    pipeline_configs = []

    class FakePipeline:
        def __init__(self, label):
            self.label = label

        def run(self, text):
            return SimpleNamespace(
                answer=f"answer: {text}",
                query_info=None,
                signals=[],
                write_decisions=[],
                judgments=[],
                goal_plan=None,
                verify_result=None,
                structured_output={"version": 1},
            )

    def fake_make_pipeline(
        store_dir,
        use_llm_extractor=False,
        llm_client=None,
        use_llm_usability_judge=False,
        llm_usability_judge_client=None,
        use_llm_answer_generator=False,
        llm_answer_generator_client=None,
        use_llm_verifier=False,
        llm_verifier_client=None,
        max_judge_memories=None,
        force_query_intent=False,
        haystack_extractor_cache=None,
        **kwargs,
    ):
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "raw_interaction_log.jsonl").write_text("", encoding="utf-8")
        (store_dir / "ordinary_memory.json").write_text("[]\n", encoding="utf-8")
        (store_dir / "active_constraints.json").write_text("[]\n", encoding="utf-8")
        pipeline_configs.append(
            {
                "label": store_dir.name,
                "use_llm_usability_judge": use_llm_usability_judge,
                "use_llm_answer_generator": use_llm_answer_generator,
                "use_llm_verifier": use_llm_verifier,
                "max_judge_memories": max_judge_memories,
            }
        )
        return FakePipeline(store_dir.name)

    monkeypatch.setattr(runner, "make_pipeline", fake_make_pipeline)

    run_stale_sample(
        make_sample(),
        use_llm_usability_judge=True,
        use_llm_answer_generator=True,
        use_llm_verifier=True,
        max_judge_memories=20,
    )

    haystack_config = next(config for config in pipeline_configs if config["label"] == "haystack")
    dim_configs = [config for config in pipeline_configs if config["label"].startswith("dim")]

    assert haystack_config["use_llm_usability_judge"] is False
    assert haystack_config["use_llm_answer_generator"] is False
    assert haystack_config["use_llm_verifier"] is False
    assert haystack_config["max_judge_memories"] is None
    assert dim_configs
    assert all(config["use_llm_usability_judge"] is True for config in dim_configs)
    assert all(config["use_llm_answer_generator"] is True for config in dim_configs)
    assert all(config["use_llm_verifier"] is True for config in dim_configs)


def test_run_stale_dataset_writes_output_file(tmp_path):
    dataset = tmp_path / "toy_MAIN.json"
    output = tmp_path / "answers.json"
    dataset.write_text(json.dumps([make_sample()], ensure_ascii=False), encoding="utf-8")

    results = run_stale_dataset(dataset, output, max_samples=1, verbose=True)

    assert len(results) == 1
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written[0]["uid"] == "sample_001"
    assert "target_model_responses" in written[0]
    assert "dim1_response" in written[0]["target_model_responses"]


def test_run_stale_dataset_writes_trace_file(tmp_path):
    dataset = tmp_path / "toy_MAIN.json"
    output = tmp_path / "answers.json"
    trace = tmp_path / "trace.json"
    dataset.write_text(json.dumps([make_sample()], ensure_ascii=False), encoding="utf-8")

    run_stale_dataset(dataset, output, max_samples=1, trace_path=trace)

    traced = json.loads(trace.read_text(encoding="utf-8"))
    assert traced[0]["uid"] == "sample_001"
    dim1 = traced[0]["dims"]["dim1"]
    assert "signal_types" in dim1
    assert "retrieval" in dim1
    assert "judge" in dim1
    assert "answer" in dim1
    assert "verifier" in dim1
    assert dim1["likely_problem"] in {
        "retrieval_empty",
        "judge_llm_call_failed",
        "judge_all_fallback",
        "judge_all_noise",
        "verifier_rejected_answer",
        "ok",
    }


def test_cached_llm_client_reuses_memory_and_disk_cache(tmp_path):
    class FakeClient:
        model = "fake-model"
        base_url = "https://example.test"

        def __init__(self):
            self.calls = 0

        def complete_json(self, prompt):
            self.calls += 1
            return {"signals": [{"content": prompt}]}

    cache_path = tmp_path / "llm_cache.json"
    client = FakeClient()
    cached = CachedLLMClient(client, cache_path)

    assert cached.complete_json("same prompt") == {"signals": [{"content": "same prompt"}]}
    assert cached.complete_json("same prompt") == {"signals": [{"content": "same prompt"}]}
    assert client.calls == 1
    assert cached.hits == 1
    assert cached.misses == 1

    second_cached = CachedLLMClient(client, cache_path)
    assert second_cached.complete_json("same prompt") == {"signals": [{"content": "same prompt"}]}
    assert client.calls == 1
    assert second_cached.hits == 1


def test_cached_llm_client_normalizes_uuid_and_timestamps(tmp_path):
    class FakeClient:
        model = "fake-model"
        base_url = "https://example.test"

        def __init__(self):
            self.calls = 0

        def complete_json(self, prompt):
            self.calls += 1
            return {"calls": self.calls}

    client = FakeClient()
    cached = CachedLLMClient(client, tmp_path / "llm_cache.json")
    first_prompt = (
        '{"memory_id":"11111111-1111-4111-8111-111111111111",'
        '"created_at":"2026-06-16T00:00:00+00:00"}'
    )
    second_prompt = (
        '{"memory_id":"22222222-2222-4222-8222-222222222222",'
        '"created_at":"2026-06-17T12:30:45.123456+00:00"}'
    )

    assert cached.complete_json(first_prompt) == {"calls": 1}
    assert cached.complete_json(second_prompt) == {"calls": 1}
    assert client.calls == 1
    assert cached.hits == 1
    assert cached.misses == 1


def test_run_stale_sample_dedupes_repeated_haystack_user_turns(monkeypatch):
    sample = make_sample()
    sample["haystack_session"].append(
        [
            {"role": "user", "content": "我喜欢重辣口味，尤其喜欢川菜和火锅。"},
            {"role": "assistant", "content": "重复。"},
        ]
    )
    calls = []

    class FakePipeline:
        def __init__(self, label):
            self.label = label

        def run(self, text):
            calls.append((self.label, text))
            return SimpleNamespace(
                answer=f"answer: {text}",
                query_info=None,
                signals=[],
                write_decisions=[],
                judgments=[],
                goal_plan=None,
                verify_result=None,
                structured_output={"version": 1},
            )

    def fake_make_pipeline(
        store_dir,
        use_llm_extractor=False,
        llm_client=None,
        use_llm_usability_judge=False,
        llm_usability_judge_client=None,
        use_llm_verifier=False,
        llm_verifier_client=None,
        max_judge_memories=None,
        force_query_intent=False,
        haystack_extractor_cache=None,
        **kwargs,
    ):
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "raw_interaction_log.jsonl").write_text("", encoding="utf-8")
        (store_dir / "ordinary_memory.json").write_text("[]\n", encoding="utf-8")
        (store_dir / "active_constraints.json").write_text("[]\n", encoding="utf-8")
        return FakePipeline(store_dir.name)

    monkeypatch.setattr(runner, "make_pipeline", fake_make_pipeline)

    run_stale_sample(sample)

    labels = [label for label, _text in calls]
    assert labels.count("haystack") == 2


def test_conditional_answer_verifier_skips_when_no_judgments_or_constraints():
    class FakeVerifier:
        def __init__(self):
            self.calls = 0

        def verify(self, **kwargs):
            self.calls += 1
            return {"pass": False}

    fake = FakeVerifier()
    result = ConditionalAnswerVerifier(fake).verify("answer", [], [])

    assert fake.calls == 0
    assert result["pass"] is True
    assert result["verifier_type"] == "skipped"


def test_conditional_answer_verifier_delegates_when_evidence_exists():
    class FakeVerifier:
        def __init__(self):
            self.calls = 0

        def verify(self, **kwargs):
            self.calls += 1
            return {"pass": True, "verifier_type": "fake"}

    fake = FakeVerifier()
    result = ConditionalAnswerVerifier(fake).verify("answer", [{"usage_status": "USABLE"}], [])

    assert fake.calls == 1
    assert result["verifier_type"] == "fake"


def test_query_intent_fallback_extractor_adds_query_signal_when_missing():
    class OrdinaryOnlyExtractor:
        def extract(self, text):
            return [
                MemorySignal(
                    signal_id="s1",
                    signal_type="ordinary_memory",
                    content="ordinary candidate",
                    subject="user",
                    scope=["food"],
                    confidence=0.8,
                    evidence_text=text,
                )
            ]

    signals = QueryIntentFallbackExtractor(OrdinaryOnlyExtractor()).extract("Can I eat tonight?")

    assert [signal.signal_type for signal in signals] == ["ordinary_memory", "query_intent"]
    assert signals[-1].scope == ["food"]


def test_query_intent_fallback_extractor_does_not_duplicate_existing_trigger():
    class QueryExtractor:
        def extract(self, text):
            return [
                MemorySignal(
                    signal_id="s1",
                    signal_type="query_intent",
                    content=text,
                    subject="user",
                    scope=["general"],
                    confidence=1.0,
                    evidence_text=text,
                )
            ]

    signals = QueryIntentFallbackExtractor(QueryExtractor()).extract("Can I eat tonight?")

    assert [signal.signal_type for signal in signals] == ["query_intent"]


def test_normalize_haystack_text_collapses_fullwidth_and_spaces():
    assert normalize_haystack_text("  Ａ  B\nC  ") == "A B C"


def test_cached_signal_extractor_reuses_normalized_haystack_turns(tmp_path):
    class FakeExtractor:
        def __init__(self):
            self.calls = 0

        def extract(self, text):
            self.calls += 1
            return [
                MemorySignal(
                    signal_id=f"s{self.calls}",
                    signal_type="ordinary_memory",
                    content=text.strip(),
                    subject="user",
                    scope=["general"],
                    confidence=0.8,
                    evidence_text=text,
                )
            ]

    base = FakeExtractor()
    cache = HaystackExtractorCache(tmp_path / "haystack_cache.json")
    extractor = CachedSignalExtractor(base, cache)

    first = extractor.extract("  hello   world  ")
    second = extractor.extract("hello world")

    assert base.calls == 1
    assert first[0].content == "hello   world"
    assert second[0].content == "hello   world"
    assert cache.hits == 1


def test_prewarm_haystack_cache_dedupes_unique_turns(tmp_path):
    class FakeLLMClient:
        model = "fake-model"
        base_url = "https://example.test"

        def __init__(self):
            self.calls = 0

        def complete_json(self, prompt):
            self.calls += 1
            return {
                "signals": [
                    {
                        "signal_type": "ordinary_memory",
                        "content": f"signal {self.calls}",
                        "subject": "user",
                        "scope": ["general"],
                        "confidence": 0.8,
                        "evidence_text": "evidence",
                    }
                ]
            }

    sample = make_sample()
    sample["haystack_session"].append(
        [
            {"role": "user", "content": "我喜欢重辣口味，尤其喜欢川菜和火锅。"},
        ]
    )
    client = FakeLLMClient()
    cache = HaystackExtractorCache(tmp_path / "haystack_cache.json")

    runner._prewarm_haystack_extractor_cache([sample], client, cache)

    assert client.calls == 2
    assert len(cache.cache) == 2


def test_prewarm_haystack_cache_renders_progress(tmp_path):
    class FakeLLMClient:
        model = "fake-model"
        base_url = "https://example.test"
        hits = 0
        misses = 0

        def __init__(self):
            self.calls = 0

        def complete_json(self, prompt):
            self.calls += 1
            self.misses += 1
            return {
                "signals": [
                    {
                        "signal_type": "ordinary_memory",
                        "content": f"signal {self.calls}",
                        "subject": "user",
                        "scope": ["general"],
                        "confidence": 0.8,
                        "evidence_text": "evidence",
                    }
                ]
            }

    sample = make_sample()
    client = FakeLLMClient()
    cache = HaystackExtractorCache(tmp_path / "haystack_cache.json")
    stream = StringIO()

    runner._prewarm_haystack_extractor_cache(
        [sample],
        client,
        cache,
        show_progress=True,
        stream=stream,
    )

    output = stream.getvalue()
    assert "prewarm" in output
    assert "2/2" in output
    assert "haystack_hits=" in output
    assert "haystack_misses=" in output
    assert "llm_misses=" in output


def test_limited_memory_retriever_limits_memories_and_constraints():
    memories = [
        SimpleNamespace(
            memory_id="m1",
            content="unrelated",
            subject="user",
            scope=[],
            status="active",
            confidence=0.9,
            updated_at="2026-06-16T00:00:00+00:00",
        ),
        SimpleNamespace(
            memory_id="m2",
            content="今晚火锅",
            subject="user",
            scope=[],
            status="active",
            confidence=0.9,
            updated_at="2026-06-16T00:00:00+00:00",
        ),
        SimpleNamespace(
            memory_id="m3",
            content="also unrelated",
            subject="user",
            scope=[],
            status="active",
            confidence=0.9,
            updated_at="2026-06-16T00:00:00+00:00",
        ),
    ]
    constraints = [
        SimpleNamespace(
            constraint_id="c1",
            content="今晚不能吃火锅",
            subject="user",
            scope=[],
            status="active",
            priority="high",
            strength="hard",
            confidence=0.9,
            updated_at="2026-06-16T00:00:00+00:00",
        ),
        SimpleNamespace(
            constraint_id="c2",
            content="预算最多五十元",
            subject="user",
            scope=[],
            status="active",
            priority="high",
            strength="hard",
            confidence=0.9,
            updated_at="2026-06-16T00:00:00+00:00",
        ),
    ]

    class FakeRetriever:
        def retrieve(self, query_info):
            return memories, constraints

    query_info = SimpleNamespace(
        query="今晚还能吃火锅吗",
        query_scope=[],
        resolved_subject="user",
    )
    limited_memories, constraints = LimitedMemoryRetriever(
        FakeRetriever(),
        max_memories=2,
        max_constraints=1,
    ).retrieve(query_info)

    assert [memory.memory_id for memory in limited_memories] == ["m2", "m1"]
    assert [constraint.constraint_id for constraint in constraints] == ["c1"]


def test_rank_memories_by_recency_uses_updated_at_descending():
    older = SimpleNamespace(
        memory_id="old",
        updated_at="2026-06-15T00:00:00+00:00",
    )
    newer = SimpleNamespace(
        memory_id="new",
        updated_at="2026-06-17T00:00:00+00:00",
    )

    ranked = rank_memories_by_recency([older, newer])

    assert [memory.memory_id for memory in ranked] == ["new", "old"]


def test_stale_aware_retriever_appends_recent_memory_to_limited_candidates():
    old_premise = SimpleNamespace(
        memory_id="old",
        content="User loves the energy in Seattle and finds it inspiring.",
        subject="user",
        scope=["location", "inspiration"],
        status="active",
        confidence=1.0,
        updated_at="2026-06-15T00:00:00+00:00",
    )
    distractor = SimpleNamespace(
        memory_id="distractor",
        content="User likes inspiring art in Seattle galleries.",
        subject="user",
        scope=["inspiration"],
        status="active",
        confidence=1.0,
        updated_at="2026-06-15T01:00:00+00:00",
    )
    new_state = SimpleNamespace(
        memory_id="new",
        content="The user appreciates the stillness in the evenings after a busy day.",
        subject="user",
        scope=["evenings"],
        status="active",
        confidence=1.0,
        updated_at="2026-06-17T00:00:00+00:00",
    )

    class FakeRetriever:
        def retrieve(self, query_info):
            return [old_premise, distractor, new_state], []

    query_info = SimpleNamespace(
        query="Based on the conversation history, does the user still find inspiration from the energy of Seattle?",
        query_scope=["conversation_history"],
        resolved_subject="user",
        query_intent="question_with_premise",
        possible_old_premises=[],
    )

    limited_memories, _ = LimitedMemoryRetriever(
        FakeRetriever(),
        max_memories=2,
        stale_aware_retrieval=True,
        recent_memory_window=1,
    ).retrieve(query_info)

    assert [memory.memory_id for memory in limited_memories] == ["old", "distractor", "new"]


def test_stale_aware_retriever_does_not_mix_recent_memory_for_plain_query():
    relevant = SimpleNamespace(
        memory_id="relevant",
        content="User likes spicy hotpot.",
        subject="user",
        scope=["food"],
        status="active",
        confidence=1.0,
        updated_at="2026-06-15T00:00:00+00:00",
    )
    new_state = SimpleNamespace(
        memory_id="new",
        content="The user appreciates the stillness in the evenings after a busy day.",
        subject="user",
        scope=["evenings"],
        status="active",
        confidence=1.0,
        updated_at="2026-06-17T00:00:00+00:00",
    )

    class FakeRetriever:
        def retrieve(self, query_info):
            return [new_state, relevant], []

    query_info = SimpleNamespace(
        query="What should the user eat tonight?",
        query_scope=["food"],
        resolved_subject="user",
        query_intent="explicit_query_intent",
        possible_old_premises=[],
    )

    limited_memories, _ = LimitedMemoryRetriever(
        FakeRetriever(),
        max_memories=1,
        stale_aware_retrieval=True,
        recent_memory_window=1,
    ).retrieve(query_info)

    assert [memory.memory_id for memory in limited_memories] == ["relevant"]


def test_rank_constraints_for_judge_uses_lexical_relevance():
    relevant = SimpleNamespace(
        constraint_id="c1",
        content="今晚不能吃火锅",
        subject="user",
        scope=["food"],
        status="active",
        priority="high",
        strength="hard",
        confidence=0.9,
        updated_at="2026-06-16T00:00:00+00:00",
    )
    unrelated = SimpleNamespace(
        constraint_id="c2",
        content="预算最多五十元",
        subject="user",
        scope=["finance"],
        status="active",
        priority="high",
        strength="hard",
        confidence=0.9,
        updated_at="2026-06-16T00:00:00+00:00",
    )
    query_info = SimpleNamespace(
        query="今晚还能吃火锅吗",
        query_scope=["food"],
        resolved_subject="user",
        query_intent="query_intent",
    )

    ranked = rank_constraints_for_judge(query_info, [unrelated, relevant], [])

    assert [constraint.constraint_id for constraint in ranked] == ["c1", "c2"]


def test_rank_constraints_for_judge_can_use_hybrid_embedding_similarity():
    class FakeEmbeddingEncoder:
        def encode(self, text):
            lowered = text.casefold()
            if "working from home" in lowered or "home office" in lowered:
                return [1.0, 0.0]
            if "budget" in lowered:
                return [0.0, 1.0]
            return [0.0, 0.0]

    relevant = SimpleNamespace(
        constraint_id="c1",
        content="The user cannot work from home today.",
        subject="user",
        scope=[],
        status="active",
        priority="medium",
        strength="hard",
        confidence=0.9,
        updated_at="2026-06-16T00:00:00+00:00",
    )
    unrelated = SimpleNamespace(
        constraint_id="c2",
        content="The user has a strict budget for dinner.",
        subject="user",
        scope=[],
        status="active",
        priority="high",
        strength="hard",
        confidence=0.9,
        updated_at="2026-06-16T00:00:00+00:00",
    )
    query_info = SimpleNamespace(
        query="Can the user still rely on the home office setup?",
        query_scope=[],
        resolved_subject="user",
        query_intent="query_intent",
    )

    ranked = rank_constraints_for_judge(
        query_info,
        [unrelated, relevant],
        [],
        ranker="hybrid",
        embedding_encoder=FakeEmbeddingEncoder(),
    )

    assert [constraint.constraint_id for constraint in ranked] == ["c1", "c2"]


def test_rank_memories_for_judge_uses_text_overlap_when_scopes_differ():
    unrelated = SimpleNamespace(
        memory_id="m1",
        content="用户喜欢科幻电影",
        subject="user",
        scope=["entertainment"],
        status="active",
        confidence=0.9,
        updated_at="2026-06-16T00:00:00+00:00",
    )
    relevant = SimpleNamespace(
        memory_id="m2",
        content="用户喜欢重辣火锅",
        subject="user",
        scope=["food"],
        status="active",
        confidence=0.9,
        updated_at="2026-06-15T00:00:00+00:00",
    )
    query_info = SimpleNamespace(
        query="今晚还能吃火锅吗",
        query_scope=["diet"],
        resolved_subject="user",
    )

    ranked = rank_memories_for_judge(query_info, [unrelated, relevant], [])

    assert [memory.memory_id for memory in ranked] == ["m2", "m1"]


def test_rank_memories_for_judge_uses_constraint_content_overlap():
    relevant = SimpleNamespace(
        memory_id="m1",
        content="用户周末喜欢长途自驾",
        subject="user",
        scope=["travel"],
        status="active",
        confidence=0.9,
        updated_at="2026-06-15T00:00:00+00:00",
    )
    unrelated = SimpleNamespace(
        memory_id="m2",
        content="用户喜欢安静咖啡馆",
        subject="user",
        scope=["preference"],
        status="active",
        confidence=0.9,
        updated_at="2026-06-16T00:00:00+00:00",
    )
    constraint = SimpleNamespace(content="医生说不能长时间自驾", scope=["health"])
    query_info = SimpleNamespace(
        query="周末怎么安排",
        query_scope=["planning"],
        resolved_subject="user",
    )

    ranked = rank_memories_for_judge(query_info, [unrelated, relevant], [constraint])

    assert [memory.memory_id for memory in ranked] == ["m1", "m2"]


def test_limited_memory_retriever_pins_stale_target_memories_outside_top_k():
    top_ranked = SimpleNamespace(
        memory_id="top",
        content="Can the user still answer the current question?",
        subject="user",
        scope=["general"],
        status="active",
        confidence=0.9,
        updated_at="2026-06-18T00:00:00+00:00",
    )
    old_memory = SimpleNamespace(
        memory_id="old",
        content="The user has been taking the subway to work every morning.",
        subject="user",
        scope=["transport"],
        status="active",
        confidence=0.8,
        updated_at="2026-06-15T00:00:00+00:00",
    )
    new_memory = SimpleNamespace(
        memory_id="new",
        content="The user now drives to work instead of taking the subway.",
        subject="user",
        scope=["transport"],
        status="active",
        confidence=0.8,
        updated_at="2026-06-16T00:00:00+00:00",
    )

    class FakeRetriever:
        def retrieve(self, query_info):
            return [top_ranked, old_memory, new_memory], []

    query_info = SimpleNamespace(
        query="Can the user still answer the current question?",
        query_scope=["general"],
        resolved_subject="user",
        query_intent="query_intent",
        possible_old_premises=[],
    )

    retriever = LimitedMemoryRetriever(
        FakeRetriever(),
        max_memories=1,
        stale_target_context={
            "m_old": "I take the subway to work every morning.",
            "m_new": "I now drive to work instead of taking the subway.",
        },
    )

    memories, _constraints = retriever.retrieve(query_info)

    memory_ids = [memory.memory_id for memory in memories]
    assert memory_ids[:2] == ["old", "new"]
    assert "top" in memory_ids


def test_run_stale_sample_passes_max_judge_memories_to_pipeline(monkeypatch):
    values = []

    class FakePipeline:
        def __init__(self, label):
            self.label = label

        def run(self, text):
            return SimpleNamespace(
                answer=f"answer: {text}",
                query_info=None,
                signals=[],
                write_decisions=[],
                judgments=[],
                goal_plan=None,
                verify_result=None,
                structured_output={"version": 1},
            )

    def fake_make_pipeline(
        store_dir,
        use_llm_extractor=False,
        llm_client=None,
        use_llm_usability_judge=False,
        llm_usability_judge_client=None,
        use_llm_verifier=False,
        llm_verifier_client=None,
        max_judge_memories=None,
        force_query_intent=False,
        haystack_extractor_cache=None,
        **kwargs,
    ):
        values.append((store_dir.name, max_judge_memories))
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "raw_interaction_log.jsonl").write_text("", encoding="utf-8")
        (store_dir / "ordinary_memory.json").write_text("[]\n", encoding="utf-8")
        (store_dir / "active_constraints.json").write_text("[]\n", encoding="utf-8")
        return FakePipeline(store_dir.name)

    monkeypatch.setattr(runner, "make_pipeline", fake_make_pipeline)

    run_stale_sample(make_sample(), max_judge_memories=20)

    assert values
    assert ("haystack", None) in values
    dim_values = [
        max_judge_memories
        for label, max_judge_memories in values
        if label.startswith("dim")
    ]
    assert dim_values
    assert set(dim_values) == {20}


def test_run_stale_sample_forces_query_intent_only_for_dim_queries(monkeypatch):
    values = []

    class FakePipeline:
        def run(self, text):
            return SimpleNamespace(
                answer=f"answer: {text}",
                query_info=None,
                signals=[],
                write_decisions=[],
                judgments=[],
                goal_plan=None,
                verify_result=None,
                structured_output={"version": 1},
            )

    def fake_make_pipeline(
        store_dir,
        use_llm_extractor=False,
        llm_client=None,
        use_llm_usability_judge=False,
        llm_usability_judge_client=None,
        use_llm_verifier=False,
        llm_verifier_client=None,
        max_judge_memories=None,
        force_query_intent=False,
        haystack_extractor_cache=None,
        **kwargs,
    ):
        values.append((store_dir.name, force_query_intent))
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "raw_interaction_log.jsonl").write_text("", encoding="utf-8")
        (store_dir / "ordinary_memory.json").write_text("[]\n", encoding="utf-8")
        (store_dir / "active_constraints.json").write_text("[]\n", encoding="utf-8")
        return FakePipeline()

    monkeypatch.setattr(runner, "make_pipeline", fake_make_pipeline)

    run_stale_sample(make_sample())

    assert ("haystack", False) in values
    assert ("dim1", True) in values
    assert ("dim3", True) in values


def test_run_stale_sample_uses_isolated_store_dirs_per_uid(monkeypatch):
    store_dirs = []

    class FakePipeline:
        def run(self, text):
            return SimpleNamespace(
                answer=f"answer: {text}",
                query_info=None,
                signals=[],
                write_decisions=[],
                judgments=[],
                goal_plan=None,
                verify_result=None,
                structured_output={"version": 1},
            )

    def fake_make_pipeline(
        store_dir,
        use_llm_extractor=False,
        llm_client=None,
        use_llm_usability_judge=False,
        llm_usability_judge_client=None,
        use_llm_verifier=False,
        llm_verifier_client=None,
        max_judge_memories=None,
        force_query_intent=False,
        haystack_extractor_cache=None,
        **kwargs,
    ):
        store_dirs.append(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "raw_interaction_log.jsonl").write_text("", encoding="utf-8")
        (store_dir / "ordinary_memory.json").write_text("[]\n", encoding="utf-8")
        (store_dir / "active_constraints.json").write_text("[]\n", encoding="utf-8")
        return FakePipeline()

    monkeypatch.setattr(runner, "make_pipeline", fake_make_pipeline)

    sample_a = make_sample()
    sample_b = make_sample()
    sample_b["uid"] = "sample_002"
    run_stale_sample(sample_a)
    run_stale_sample(sample_b)

    uid_dirs = {path.parent.name for path in store_dirs}
    assert {"sample_001", "sample_002"}.issubset(uid_dirs)


def test_run_stale_sample_supports_parallel_dims(monkeypatch):
    calls = []

    class FakePipeline:
        def __init__(self, label):
            self.label = label

        def run(self, text):
            calls.append((self.label, text))
            return SimpleNamespace(
                answer=f"answer: {text}",
                query_info=None,
                signals=[],
                write_decisions=[],
                judgments=[],
                goal_plan=None,
                verify_result=None,
                structured_output={"version": 1},
            )

    def fake_make_pipeline(
        store_dir,
        use_llm_extractor=False,
        llm_client=None,
        use_llm_usability_judge=False,
        llm_usability_judge_client=None,
        use_llm_verifier=False,
        llm_verifier_client=None,
        max_judge_memories=None,
        force_query_intent=False,
        haystack_extractor_cache=None,
        **kwargs,
    ):
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "raw_interaction_log.jsonl").write_text("", encoding="utf-8")
        (store_dir / "ordinary_memory.json").write_text("[]\n", encoding="utf-8")
        (store_dir / "active_constraints.json").write_text("[]\n", encoding="utf-8")
        return FakePipeline(store_dir.name)

    monkeypatch.setattr(runner, "make_pipeline", fake_make_pipeline)

    result = run_stale_sample(make_sample(), parallel_dims=True)

    assert result["target_model_responses"]["dim1_response"]
    assert result["target_model_responses"]["dim3_response"]
    assert {label for label, _ in calls}.issuperset({"haystack", "dim1", "dim3"})


def test_run_stale_dataset_resume_skips_completed_sample(tmp_path, monkeypatch):
    dataset = tmp_path / "toy_MAIN.json"
    output = tmp_path / "answers.json"
    dataset.write_text(json.dumps([make_sample()], ensure_ascii=False), encoding="utf-8")
    output.write_text(json.dumps([{"uid": "sample_001"}], ensure_ascii=False), encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("completed sample should be skipped")

    monkeypatch.setattr(runner, "run_stale_sample", fail_if_called)

    results = run_stale_dataset(dataset, output, max_samples=1, verbose=True)

    assert results == [{"uid": "sample_001"}]


def test_progress_bar_renders_status_and_cache_counts():
    stream = StringIO()
    llm_client = SimpleNamespace(hits=3, misses=2)
    progress = ProgressBar(total=10, enabled=True, width=10, stream=stream)

    progress.update(4, "running", "sample_004", processed=3, skipped=1, llm_client=llm_client)
    progress.finish()

    output = stream.getvalue()
    assert "4/10" in output
    assert "running" in output
    assert "uid=sample_004" in output
    assert "cache_hits=3" in output
    assert "cache_misses=2" in output

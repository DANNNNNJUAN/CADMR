from cadmr.pipeline import CADMRPipeline
from cadmr.schemas import MemorySignal, OrdinaryMemory
from cadmr.stores import ActiveConstraintStore, OrdinaryMemoryStore, RawInteractionLog


class MockLLMStyleExtractor:
    """Deterministic test extractor that simulates LLM-classified signals."""

    def extract(self, text):
        signals = []
        if "我喜欢重辣口味" in text:
            signals.append(
                self._signal(
                    "ordinary_memory",
                    "我喜欢重辣口味，尤其喜欢川菜和火锅。",
                    text,
                    scope=["diet", "preference"],
                )
            )
        if "医生说我接下来四周不能吃辣" in text:
            signals.append(
                self._signal(
                    "active_constraint",
                    "医生说我接下来四周不能吃辣",
                    text,
                    scope=["health", "diet"],
                )
            )
        if "今晚还能吃火锅" in text:
            signals.append(
                self._signal(
                    "query_intent",
                    "询问今晚是否还能吃火锅",
                    text,
                    scope=["diet"],
                )
            )
        if "我现在住在北京" in text:
            signals.append(
                self._signal("ordinary_memory", "我现在住在北京。", text, scope=["location"])
            )
        if "我已经搬到上海" in text:
            signals.append(
                self._signal("ordinary_memory", "我已经搬到上海了。", text, scope=["location"])
            )
        if "附近" in text and "餐厅" in text:
            signals.append(
                self._signal("query_intent", "询问当前附近餐厅", text, scope=["location", "diet"])
            )
        if "我只是轻度乳糖不耐受" in text and "既然" not in text:
            signals.append(
                self._signal("ordinary_memory", "我只是轻度乳糖不耐受。", text, scope=["health", "diet"])
            )
        if "我的猫刚从医院回来" in text:
            signals.append(
                self._signal("ordinary_memory", "我的猫刚从医院回来", text, subject="cat", scope=["health"])
            )
        if "兽医说饮食需要小心" in text:
            signals.append(
                self._signal(
                    "active_constraint",
                    "兽医说饮食需要小心",
                    text,
                    subject="cat",
                    scope=["health", "diet"],
                )
            )
        if "既然我只是轻度乳糖不耐受" in text:
            signals.append(
                self._signal("question_premise", "我只是轻度乳糖不耐受", text, scope=["health", "diet"])
            )
        if "这个长期记忆系统 demo" in text:
            signals.append(
                self._signal(
                    "ordinary_memory",
                    "这个长期记忆系统 demo 先给内部技术同事看，只要跑通原型就行。",
                    text,
                    scope=["work", "project"],
                )
            )
        if "给合作方和管理老师演示" in text:
            signals.append(
                self._signal(
                    "ordinary_memory",
                    "明天要给合作方和管理老师演示",
                    text,
                    scope=["work", "project"],
                )
            )
        if "不能展示真实用户数据" in text or "需要脱敏" in text:
            signals.append(
                self._signal(
                    "active_constraint",
                    "不能展示真实数据，需要脱敏",
                    text,
                    scope=["work", "project", "privacy", "safety"],
                )
            )
        if "演示方案" in text or "最稳妥" in text:
            signals.append(
                self._signal("query_intent", "请求安排最稳妥的演示方案", text, scope=["work", "project"])
            )
        return signals

    def _signal(self, signal_type, content, evidence_text, subject="user", scope=None):
        return MemorySignal(
            signal_id=f"test-{signal_type}-{len(content)}",
            signal_type=signal_type,
            content=content,
            subject=subject,
            scope=scope or ["general"],
            confidence=0.95,
            evidence_text=evidence_text,
        )


def make_pipeline(tmp_path, extractor=None):
    raw_log = RawInteractionLog(tmp_path / "raw_interaction_log.jsonl")
    ordinary_store = OrdinaryMemoryStore(tmp_path / "ordinary_memory.json")
    constraint_store = ActiveConstraintStore(tmp_path / "active_constraints.json")
    pipeline = CADMRPipeline(
        raw_log=raw_log,
        ordinary_store=ordinary_store,
        constraint_store=constraint_store,
        extractor=extractor,
    )
    return pipeline, raw_log, ordinary_store, constraint_store


def signal_types(result):
    return {signal.signal_type for signal in result.signals}


def decisions_by_signal_type(result):
    return {
        signal.signal_type: decision.decision
        for signal, decision in zip(result.signals, result.write_decisions, strict=True)
    }


def test_pipeline_writes_ordinary_memory(tmp_path):
    pipeline, raw_log, ordinary_store, constraint_store = make_pipeline(
        tmp_path,
        extractor=MockLLMStyleExtractor(),
    )

    result = pipeline.run("我喜欢重辣口味，尤其喜欢川菜和火锅。")

    assert len(raw_log.list_all()) == 1
    assert len(ordinary_store.list_all()) == 1
    assert len(constraint_store.list_all()) == 0
    assert "ordinary_memory" in signal_types(result)
    assert "WRITE_TO_ORDINARY_MEMORY" in [
        decision.decision for decision in result.write_decisions
    ]


def test_pipeline_writes_active_constraint(tmp_path):
    pipeline, raw_log, ordinary_store, constraint_store = make_pipeline(
        tmp_path,
        extractor=MockLLMStyleExtractor(),
    )

    result = pipeline.run("医生说我接下来四周不能吃辣。")

    assert len(raw_log.list_all()) == 1
    assert len(ordinary_store.list_all()) == 0
    assert len(constraint_store.list_all()) == 1
    assert "active_constraint" in signal_types(result)
    assert "WRITE_TO_ACTIVE_CONSTRAINT" in [
        decision.decision for decision in result.write_decisions
    ]


def test_pipeline_handles_constraint_plus_query(tmp_path):
    pipeline, raw_log, ordinary_store, constraint_store = make_pipeline(
        tmp_path,
        extractor=MockLLMStyleExtractor(),
    )

    result = pipeline.run("医生说我接下来四周不能吃辣，那今晚还能吃火锅吗？")

    assert len(raw_log.list_all()) == 1
    assert len(constraint_store.list_all()) == 1
    assert len(ordinary_store.list_all()) == 0
    assert {"active_constraint", "query_intent"}.issubset(signal_types(result))
    assert decisions_by_signal_type(result)["query_intent"] == "DO_NOT_WRITE"
    assert result.query_info is not None
    assert result.answer is not None


def test_pipeline_does_not_write_query_intent_to_structured_stores(tmp_path):
    pipeline, raw_log, ordinary_store, constraint_store = make_pipeline(
        tmp_path,
        extractor=MockLLMStyleExtractor(),
    )

    result = pipeline.run("今晚还能吃火锅吗？")

    assert len(raw_log.list_all()) == 1
    assert len(ordinary_store.list_all()) == 0
    assert len(constraint_store.list_all()) == 0
    assert result.query_info is not None


def test_pipeline_does_not_write_hypothetical_to_structured_stores(tmp_path):
    class HypotheticalOnlyExtractor:
        def extract(self, text):
            return [
                MemorySignal(
                    signal_id="s1",
                    signal_type="hypothetical",
                    content=text,
                    subject="user",
                    scope=["location", "transport"],
                    confidence=0.8,
                    evidence_text=text,
                )
            ]

    pipeline, raw_log, ordinary_store, constraint_store = make_pipeline(
        tmp_path,
        extractor=HypotheticalOnlyExtractor(),
    )

    result = pipeline.run("如果我以后搬去上海，应该怎么通勤？")

    assert len(raw_log.list_all()) == 1
    assert len(ordinary_store.list_all()) == 0
    assert len(constraint_store.list_all()) == 0
    assert decisions_by_signal_type(result)["hypothetical"] == "DO_NOT_WRITE"
    assert result.query_info is not None
    assert result.answer is not None


def test_pipeline_does_not_write_question_premise_to_structured_stores(tmp_path):
    class PremiseOnlyExtractor:
        def extract(self, text):
            return [
                MemorySignal(
                    signal_id="s1",
                    signal_type="question_premise",
                    content=text,
                    subject="user",
                    scope=["transport"],
                    confidence=0.8,
                    evidence_text=text,
                )
            ]

    pipeline, raw_log, ordinary_store, constraint_store = make_pipeline(
        tmp_path,
        extractor=PremiseOnlyExtractor(),
    )

    result = pipeline.run("既然我每天骑车上班，帮我规划路线。")

    assert len(raw_log.list_all()) == 1
    assert len(ordinary_store.list_all()) == 0
    assert len(constraint_store.list_all()) == 0
    assert decisions_by_signal_type(result)["question_premise"] == "VERIFY_ONLY"
    assert result.query_info is not None
    assert result.answer is not None


def test_pipeline_answers_question_premise_without_query_intent_signal(tmp_path):
    class PremiseOnlyExtractor:
        def extract(self, text):
            return [
                MemorySignal(
                    signal_id="s1",
                    signal_type="question_premise",
                    content=text,
                    subject="user",
                    scope=["diet", "preference"],
                    confidence=0.8,
                    evidence_text=text,
                )
            ]

    raw_log = RawInteractionLog(tmp_path / "raw_interaction_log.jsonl")
    ordinary_store = OrdinaryMemoryStore(tmp_path / "ordinary_memory.json")
    constraint_store = ActiveConstraintStore(tmp_path / "active_constraints.json")
    pipeline = CADMRPipeline(
        raw_log=raw_log,
        ordinary_store=ordinary_store,
        constraint_store=constraint_store,
        extractor=PremiseOnlyExtractor(),
    )

    result = pipeline.run("既然我喜欢火锅，今晚是不是应该去吃？")

    assert result.query_info is not None
    assert result.query_info.query_intent == "question_with_premise"
    assert result.answer is not None
    assert result.write_decisions[0].decision == "VERIFY_ONLY"


def test_pipeline_answers_uncertain_intention_without_query_intent_signal(tmp_path):
    class UncertainOnlyExtractor:
        def extract(self, text):
            return [
                MemorySignal(
                    signal_id="s1",
                    signal_type="uncertain_intention",
                    content=text,
                    subject="user",
                    scope=["work", "project"],
                    confidence=0.7,
                    evidence_text=text,
                )
            ]

    raw_log = RawInteractionLog(tmp_path / "raw_interaction_log.jsonl")
    ordinary_store = OrdinaryMemoryStore(tmp_path / "ordinary_memory.json")
    constraint_store = ActiveConstraintStore(tmp_path / "active_constraints.json")
    pipeline = CADMRPipeline(
        raw_log=raw_log,
        ordinary_store=ordinary_store,
        constraint_store=constraint_store,
        extractor=UncertainOnlyExtractor(),
    )

    result = pipeline.run("我可能想换研究方向，应该怎么安排？")

    assert result.query_info is not None
    assert result.query_info.query_intent == "uncertain_intention_query"
    assert result.answer is not None
    assert result.write_decisions[0].decision == "WRITE_TO_SHORT_TERM_BUFFER"


def test_pipeline_end_to_end_constrained_food_case(tmp_path):
    pipeline, _, ordinary_store, constraint_store = make_pipeline(
        tmp_path,
        extractor=MockLLMStyleExtractor(),
    )

    first_result = pipeline.run("我喜欢重辣口味，尤其喜欢川菜和火锅。")

    assert len(ordinary_store.list_all()) == 1
    assert first_result.answer is None

    second_result = pipeline.run("医生说我接下来四周不能吃辣，那今晚还能吃火锅吗？")

    assert len(constraint_store.list_all()) == 1
    assert second_result.query_info is not None
    assert any(
        judgment.usage_status == "CONSTRAINED"
        for judgment in second_result.judgments
    )
    assert second_result.answer is not None
    assert "current constraints" in second_result.answer.lower()
    assert "direct action guidance" in second_result.answer
    assert "Current constraints" in second_result.answer
    assert second_result.structured_output is not None
    assert second_result.structured_output["status_summary"]["CONSTRAINED"] >= 1
    assert second_result.structured_output["retrieved_constraints"]


def test_pipeline_does_not_auto_supersede_location_memory_without_llm_judgment(tmp_path):
    pipeline, _, ordinary_store, _ = make_pipeline(tmp_path, extractor=MockLLMStyleExtractor())

    pipeline.run("我现在住在北京。")
    pipeline.run("我已经搬到上海了。")

    memories = ordinary_store.list_all()
    beijing_memory = next(memory for memory in memories if "北京" in memory.content)
    shanghai_memory = next(memory for memory in memories if "上海" in memory.content)

    assert len(memories) >= 2
    assert beijing_memory.status == "active"
    assert shanghai_memory.status == "active"


def test_pipeline_location_case_does_not_infer_stale_without_llm_judgment(tmp_path):
    pipeline, _, ordinary_store, _ = make_pipeline(tmp_path, extractor=MockLLMStyleExtractor())

    pipeline.run("我现在住在北京。")
    pipeline.run("我已经搬到上海了。")
    result = pipeline.run("我附近有什么餐厅？")

    memories = ordinary_store.list_all()
    beijing_memory = next(memory for memory in memories if "北京" in memory.content)
    shanghai_memory = next(memory for memory in memories if "上海" in memory.content)

    judgment_by_memory_id = {
        judgment.memory_id: judgment for judgment in result.judgments
    }
    assert judgment_by_memory_id[beijing_memory.memory_id].usage_status == "USABLE"
    assert judgment_by_memory_id[shanghai_memory.memory_id].usage_status != "STALE"
    assert result.answer is not None
    assert "STALE" not in result.answer


def test_pipeline_end_to_end_cat_referent_case(tmp_path):
    pipeline, _, _, _ = make_pipeline(tmp_path, extractor=MockLLMStyleExtractor())

    pipeline.run("我只是轻度乳糖不耐受。")
    pipeline.run("我的猫刚从医院回来，兽医说饮食需要小心。")
    result = pipeline.run("既然我只是轻度乳糖不耐受，那它偶尔喝点牛奶应该没事吧？")

    assert result.query_info is not None
    assert result.query_info.resolved_subject == "user"
    assert result.answer is not None
    assert result.query_info.query_intent == "question_with_premise"


def test_pipeline_goal_level_demo_is_not_reconstructed_without_llm_component(tmp_path):
    pipeline, _, _, _ = make_pipeline(tmp_path, extractor=MockLLMStyleExtractor())

    pipeline.run("这个长期记忆系统 demo 先给内部技术同事看，只要跑通原型就行。")
    pipeline.run("明天要给合作方和管理老师演示，不能展示真实用户数据，需要脱敏。")
    result = pipeline.run("帮我安排一个最稳妥的演示方案。")

    assert result.goal_plan is not None
    assert result.goal_plan["needs_goal_reconstruction"] is False
    assert result.goal_plan["forbidden_actions"] == []
    assert result.goal_plan["required_plan_components"] == []
    assert result.answer is not None
    assert result.structured_output is not None
    assert result.structured_output["goal_plan"]["needs_goal_reconstruction"] is False
    assert result.structured_output["verify_result"]["needs_revision"] is False
    assert "required plan components" not in result.answer.lower()
    assert "forbidden actions" not in result.answer.lower()


def test_pipeline_passes_structured_output_to_answer_verifier(tmp_path):
    class CapturingVerifier:
        def __init__(self):
            self.structured_output = None

        def verify(self, answer, judgments, constraints, goal_plan=None, structured_output=None):
            self.structured_output = structured_output
            return {
                "pass": True,
                "violations": [],
                "missing_components": [],
                "needs_revision": False,
                "verifier_type": "fake",
            }

    raw_log = RawInteractionLog(tmp_path / "raw_interaction_log.jsonl")
    ordinary_store = OrdinaryMemoryStore(tmp_path / "ordinary_memory.json")
    constraint_store = ActiveConstraintStore(tmp_path / "active_constraints.json")
    verifier = CapturingVerifier()
    pipeline = CADMRPipeline(
        raw_log=raw_log,
        ordinary_store=ordinary_store,
        constraint_store=constraint_store,
        answer_verifier=verifier,
        extractor=MockLLMStyleExtractor(),
    )

    pipeline.run("我喜欢重辣口味，尤其喜欢川菜和火锅。")
    result = pipeline.run("医生说我接下来四周不能吃辣，那今晚还能吃火锅吗？")

    assert verifier.structured_output is not None
    assert verifier.structured_output["answer"] == result.answer
    assert verifier.structured_output["verify_result"] is None
    assert verifier.structured_output["status_summary"]["CONSTRAINED"] >= 1
    assert result.structured_output["verify_result"]["verifier_type"] == "fake"


def test_pipeline_includes_judge_diagnostics_in_structured_output(tmp_path):
    class DiagnosticJudge:
        def __init__(self):
            self.last_diagnostics = None

        def judge(self, query_info, memories, constraints):
            self.last_diagnostics = {
                "judge_type": "DiagnosticJudge",
                "batches_total": 1,
                "batches_succeeded": 0,
                "batches_failed": 1,
                "memories_total": len(memories),
                "judgments_from_llm": 0,
                "fallback_judgments": len(memories),
                "fallback_used": bool(memories),
                "fallback_memory_ids_sample": [memory.memory_id for memory in memories],
                "errors": [{"error_type": "ValueError", "message": "bad json"}],
            }
            return []

    raw_log = RawInteractionLog(tmp_path / "raw_interaction_log.jsonl")
    ordinary_store = OrdinaryMemoryStore(tmp_path / "ordinary_memory.json")
    constraint_store = ActiveConstraintStore(tmp_path / "active_constraints.json")
    ordinary_store.add(
        OrdinaryMemory(
            memory_id="m1",
            content="I like hot pot.",
            subject="user",
            scope=["diet"],
            stability="long_term",
            status="active",
            confidence=0.9,
            evidence_ids=["i1"],
            created_at="2026-06-16T00:00:00Z",
            updated_at="2026-06-16T00:00:00Z",
        )
    )
    judge = DiagnosticJudge()
    pipeline = CADMRPipeline(
        raw_log=raw_log,
        ordinary_store=ordinary_store,
        constraint_store=constraint_store,
        extractor=MockLLMStyleExtractor(),
        usability_judge=judge,
    )

    result = pipeline.run("今晚还能吃火锅吗？")

    assert result.structured_output is not None
    diagnostics = result.structured_output["judge_diagnostics"]
    assert diagnostics["judge_type"] == "DiagnosticJudge"
    assert diagnostics["batches_failed"] == 1
    assert diagnostics["errors"][0]["error_type"] == "ValueError"


def test_pipeline_revises_answer_when_verifier_reports_stale_or_constraint_violation(tmp_path):
    class StaticRetriever:
        def retrieve(self, query_info):
            return [], []

    class StaticJudge:
        last_diagnostics = None

        def judge(self, query_info, memories, constraints):
            return []

    class RevisingAnswerGenerator:
        def __init__(self):
            self.revision_contexts = []

        def generate(
            self,
            query_info,
            judgments,
            constraints,
            memories=None,
            revision_context=None,
        ):
            self.revision_contexts.append(revision_context)
            if revision_context:
                return "Revised answer using only current allowed evidence."
            return "Initial answer that relies on stale evidence."

    class TwoStepVerifier:
        def __init__(self):
            self.answers = []

        def verify(self, answer, judgments, constraints, goal_plan=None, structured_output=None):
            self.answers.append(answer)
            if len(self.answers) == 1:
                return {
                    "pass": False,
                    "violations": [
                        {
                            "type": "stale_memory_use",
                            "evidence": "relies on stale evidence based on an outdated home office routine",
                            "related_id": "m-old",
                        }
                    ],
                    "missing_components": [],
                    "needs_revision": True,
                    "reason": "The answer used stale memory as current advice.",
                    "verifier_type": "fake",
                }
            return {
                "pass": True,
                "violations": [],
                "missing_components": [],
                "needs_revision": False,
                "reason": "Revised answer passed.",
                "verifier_type": "fake",
            }

    raw_log = RawInteractionLog(tmp_path / "raw_interaction_log.jsonl")
    ordinary_store = OrdinaryMemoryStore(tmp_path / "ordinary_memory.json")
    constraint_store = ActiveConstraintStore(tmp_path / "active_constraints.json")
    answer_generator = RevisingAnswerGenerator()
    verifier = TwoStepVerifier()
    pipeline = CADMRPipeline(
        raw_log=raw_log,
        ordinary_store=ordinary_store,
        constraint_store=constraint_store,
        extractor=MockLLMStyleExtractor(),
        retriever=StaticRetriever(),
        usability_judge=StaticJudge(),
        answer_generator=answer_generator,
        answer_verifier=verifier,
    )

    result = pipeline.run("今晚还能吃火锅吗？")

    assert verifier.answers == [
        "Initial answer that relies on stale evidence.",
        "Revised answer using only current allowed evidence.",
    ]
    assert answer_generator.revision_contexts[0] is None
    assert answer_generator.revision_contexts[1]["violations"][0]["type"] == "stale_memory_use"
    assert answer_generator.revision_contexts[1]["do_not_mention"] == [
        "relies on stale evidence",
        "an outdated home office routine",
    ]
    assert result.answer == "Revised answer using only current allowed evidence."
    assert result.verify_result["pass"] is True
    assert result.verify_result["revision_attempted"] is True
    assert result.verify_result["revision_trigger_types"] == ["stale_memory_use"]
    assert result.verify_result["first_verify_result"]["pass"] is False
    assert result.structured_output["answer"] == result.answer
    assert result.structured_output["verify_result"]["revision_attempted"] is True

from cadmr.answer_generator import ConstrainedAnswerGenerator, LLMConstrainedAnswerGenerator
from cadmr.schemas import ActiveConstraint, MemoryJudgment, OrdinaryMemory, QueryInfo


def make_query_info(
    query: str = "Can I still have hotpot tonight?",
    query_scope: list[str] | None = None,
) -> QueryInfo:
    return QueryInfo(
        query=query,
        query_intent="unknown",
        query_scope=query_scope or ["diet", "health"],
        resolved_subject="user",
        requires_action=True,
        requires_plan=False,
        possible_old_premises=[],
    )


def make_constraint(content: str = "The doctor said the user cannot eat spicy food for the next four weeks.") -> ActiveConstraint:
    return ActiveConstraint(
        constraint_id="c1",
        content=content,
        subject="user",
        scope=["health", "diet"],
        priority="high",
        strength="hard",
        valid_time={},
        status="active",
        source="user_input",
        confidence=0.9,
        evidence_ids=["i1"],
        created_at="2026-06-16T00:00:00Z",
        updated_at="2026-06-16T00:00:00Z",
    )


def make_judgment(usage_status: str) -> MemoryJudgment:
    return MemoryJudgment(
        memory_id="m1",
        usage_status=usage_status,
        truth_status="still_true" if usage_status != "NOISE" else "unknown",
        actionability="blocked_or_limited" if usage_status == "CONSTRAINED" else "actionable",
        blocked_by=["c1"] if usage_status == "CONSTRAINED" else [],
        replaced_by=[],
        allowed_use="This memory can be used as current grounding.",
        forbidden_use="Do not use this memory as a directly actionable basis."
        if usage_status == "CONSTRAINED"
        else "",
        reason="Test judgment.",
    )


def make_memory(
    memory_id: str = "m1",
    content: str = "The doctor said I cannot drive for long periods for the next two weeks.",
    scope: list[str] | None = None,
) -> OrdinaryMemory:
    return OrdinaryMemory(
        memory_id=memory_id,
        content=content,
        subject="user",
        scope=scope or ["health", "driving"],
        stability="long_term",
        status="active",
        confidence=0.9,
        evidence_ids=["i1"],
        created_at="2026-06-16T00:00:00Z",
        updated_at="2026-06-16T00:00:00Z",
    )


def test_constrained_answer_reports_judgment_and_constraint():
    answer = ConstrainedAnswerGenerator().generate(
        make_query_info(),
        [make_judgment("CONSTRAINED")],
        [make_constraint()],
    )

    assert "current constraints" in answer.lower()
    assert "direct action guidance" in answer
    assert "cannot eat spicy food" in answer
    assert "recommend very spicy hotpot" not in answer.lower()


def test_constrained_answer_keeps_user_visible_fallback_english_with_chinese_source_text():
    answer = ConstrainedAnswerGenerator().generate(
        make_query_info(),
        [make_judgment("CONSTRAINED")],
        [make_constraint("医生说用户接下来四周不能吃辣")],
    )

    assert "A retrieved active constraint applies to this query" in answer
    assert "不能吃辣" not in answer


def test_constrained_answer_does_not_mine_constraint_text_from_memories():
    answer = ConstrainedAnswerGenerator().generate(
        make_query_info(
            "Can I still follow my habit and drive to Hangzhou this weekend?",
            query_scope=["travel", "driving", "health"],
        ),
        [make_judgment("CONSTRAINED")],
        [],
        [make_memory()],
    )

    assert "cannot drive for long periods" not in answer
    assert "no concrete constraint text was retrieved" in answer


def test_budget_restaurant_query_does_not_use_food_template_without_food_constraint():
    answer = ConstrainedAnswerGenerator().generate(
        make_query_info(
            "Since I like premium experiences, should you directly recommend the most expensive restaurant?",
            query_scope=["finance", "entertainment", "preference"],
        ),
        [make_judgment("CONSTRAINED")],
        [make_constraint("The user needs to control spending.")],
    )

    assert "control spending" in answer
    assert "clear broth" not in answer
    assert "tomato broth" not in answer
    assert "irritating food" not in answer


def test_constrained_answer_is_not_overridden_by_arrangement_word():
    answer = ConstrainedAnswerGenerator().generate(
        make_query_info("Help me arrange what would be suitable to eat tonight."),
        [make_judgment("CONSTRAINED")],
        [make_constraint()],
    )

    assert "current constraints" in answer.lower()
    assert "direct action guidance" in answer
    assert "anonymized examples" not in answer
    assert "memory trace" not in answer


def test_usable_answer():
    answer = ConstrainedAnswerGenerator().generate(
        make_query_info("How should I commute?"),
        [make_judgment("USABLE")],
        [],
    )

    assert "available relevant evidence" in answer
    assert "Action:" not in answer
    assert "Recommendation:" in answer


def test_no_judgments_answer():
    answer = ConstrainedAnswerGenerator().generate(make_query_info(), [], [])

    assert "not have enough relevant stored evidence" in answer
    assert "Answer:" in answer


def test_noise_only_answer():
    answer = ConstrainedAnswerGenerator().generate(
        make_query_info(),
        [make_judgment("NOISE")],
        [],
    )

    assert "do not have relevant stored evidence" in answer
    assert "current question only" in answer


def test_structured_answer_shape_for_stale_case():
    stale = make_judgment("STALE")
    stale.memory_id = "old"
    usable = make_judgment("USABLE")
    usable.memory_id = "new"
    structured = ConstrainedAnswerGenerator().build_structured_answer(
        make_query_info("Does the user still live in San Diego?", query_scope=["location"]),
        [stale, usable],
        [],
        [
            make_memory("old", "The user used to live in San Diego.", scope=["location"]),
            make_memory("new", "The user now enjoys time in the mountains.", scope=["location"]),
        ],
    )

    assert structured["answer"] == "I would not assume the older premise is still current."
    assert structured["evidence"] == ["The user now enjoys time in the mountains."]
    assert "action" not in structured
    assert "older premise" in structured["recommendation"]


def test_structured_answer_does_not_mechanically_rewrite_evidence_as_action():
    usable = make_judgment("USABLE")
    usable.memory_id = "relax"

    answer = ConstrainedAnswerGenerator().generate(
        make_query_info(
            "What would you recommend for a cozy evening?",
            query_scope=["evening", "relaxation"],
        ),
        [usable],
        [],
        [
            make_memory(
                "relax",
                "The user is interested in deep breathing exercises and progressive muscle relaxation.",
                scope=["evening", "relaxation"],
            )
        ],
    )

    assert "Action:" not in answer
    assert "deep breathing exercises and progressive muscle relaxation" in answer
    assert "Answer directly using the relevant evidence listed here" not in answer


def test_structured_answer_avoids_internal_status_language():
    stale = make_judgment("STALE")
    answer = ConstrainedAnswerGenerator().generate(
        make_query_info("Does the older premise still hold?"),
        [stale],
        [],
    )

    lowered = answer.lower()
    assert "recalled memories require caution" not in lowered
    assert "stale" not in lowered
    assert "noise" not in lowered
    assert "judgment" not in lowered


def test_stale_answer_lists_usable_content_not_memory_ids():
    stale = make_judgment("STALE")
    stale.memory_id = "stale-memory-id"
    usable = make_judgment("USABLE")
    usable.memory_id = "usable-memory-id"
    memories = [
        make_memory(
            memory_id="usable-memory-id",
            content="The user now lives in Xuhui, Shanghai.",
            scope=["location"],
        ),
        make_memory(
            memory_id="stale-memory-id",
            content="The user previously lived in Beijing.",
            scope=["location"],
        ),
    ]

    answer = ConstrainedAnswerGenerator().generate(
        make_query_info("What restaurants are nearby?", query_scope=["location"]),
        [stale, usable],
        [],
        memories,
    )

    assert "The user now lives in Xuhui, Shanghai" in answer
    assert "usable-memory-id" not in answer
    assert "stale-memory-id" not in answer
    assert "The user previously lived in Beijing" not in answer


def test_llm_answer_generator_returns_english_answer_from_structured_reasoning():
    class FakeLLMClient:
        def __init__(self):
            self.prompt = ""

        def complete_json(self, prompt):
            self.prompt = prompt
            return {
                "answer": (
                    "I would not assume the user still lives in San Diego. "
                    "The newer evidence points to time in the mountains, so mountain-friendly "
                    "outdoor activities would be safer than San Diego beach recommendations."
                )
            }

    client = FakeLLMClient()
    stale = make_judgment("STALE")
    stale.reason = "The San Diego premise is replaced by newer mountain evidence."
    memories = [
        make_memory(memory_id="m1", content="The user used to live in San Diego.", scope=["location"]),
        make_memory(memory_id="m2", content="The user now enjoys time in the mountains.", scope=["location"]),
    ]

    answer = LLMConstrainedAnswerGenerator(client).generate(
        make_query_info("Does the user still live in San Diego?", query_scope=["location"]),
        [stale],
        [],
        memories,
    )

    assert "not assume" in answer
    assert "mountains" in answer
    assert "CADMR" not in answer
    assert "Generate a concise, user-facing answer in English." in client.prompt
    assert "Use English as the default output language" in client.prompt
    assert "Do not mention CADMR" in client.prompt


def test_llm_answer_generator_hides_noise_and_suspended_memory_content_from_prompt():
    class FakeLLMClient:
        def __init__(self):
            self.prompt = ""

        def complete_json(self, prompt):
            self.prompt = prompt
            return {"answer": "Use only the visible current evidence."}

    usable = make_judgment("USABLE")
    usable.memory_id = "usable-memory-id"
    noise = make_judgment("NOISE")
    noise.memory_id = "noise-memory-id"
    suspended = make_judgment("SUSPENDED")
    suspended.memory_id = "suspended-memory-id"
    memories = [
        make_memory("usable-memory-id", "The user currently enjoys quiet evenings."),
        make_memory("noise-memory-id", "The user once asked about unrelated Seattle museums."),
        make_memory("suspended-memory-id", "The user's current home city is uncertain."),
    ]
    client = FakeLLMClient()

    LLMConstrainedAnswerGenerator(client).generate(
        make_query_info("What should the user do tonight?"),
        [usable, noise, suspended],
        [],
        memories,
    )

    assert "The user currently enjoys quiet evenings" in client.prompt
    assert "unrelated Seattle museums" not in client.prompt
    assert "current home city is uncertain" not in client.prompt
    assert '"NOISE": 1' in client.prompt
    assert '"SUSPENDED": 1' in client.prompt
    assert "Memory content from CONSTRAINED, STALE, SUSPENDED, and" in client.prompt


def test_llm_answer_generator_hides_constrained_and_stale_memory_content_from_prompt():
    class FakeLLMClient:
        def __init__(self):
            self.prompt = ""

        def complete_json(self, prompt):
            self.prompt = prompt
            return {"answer": "Use current evidence and respect restrictions."}

    usable = make_judgment("USABLE")
    usable.memory_id = "usable-memory-id"
    constrained = make_judgment("CONSTRAINED")
    constrained.memory_id = "constrained-memory-id"
    constrained.blocked_by = ["c1"]
    stale = make_judgment("STALE")
    stale.memory_id = "stale-memory-id"
    memories = [
        make_memory("usable-memory-id", "The user currently wants a quiet evening."),
        make_memory("constrained-memory-id", "The user wants to cook spicy curry."),
        make_memory("stale-memory-id", "The user used to live near Seattle event venues."),
    ]
    constraint = make_constraint("The user should avoid long cooking tonight.")
    client = FakeLLMClient()

    LLMConstrainedAnswerGenerator(client).generate(
        make_query_info("What should the user do tonight?"),
        [usable, constrained, stale],
        [constraint],
        memories,
    )

    assert "The user currently wants a quiet evening" in client.prompt
    assert "spicy curry" not in client.prompt
    assert "Seattle event venues" not in client.prompt
    assert "The user should avoid long cooking tonight." in client.prompt
    assert '"CONSTRAINED": 1' in client.prompt
    assert '"STALE": 1' in client.prompt
    assert "answer_control_judgments" in client.prompt


def test_llm_answer_generator_only_exposes_blocking_constraints():
    class FakeLLMClient:
        def __init__(self):
            self.prompt = ""

        def complete_json(self, prompt):
            self.prompt = prompt
            return {"answer": "Adapt to the relevant constraint."}

    constrained = make_judgment("CONSTRAINED")
    constrained.memory_id = "m1"
    constrained.blocked_by = ["c1"]
    relevant_constraint = make_constraint("The user cannot eat spicy food.")
    relevant_constraint.constraint_id = "c1"
    unrelated_constraint = make_constraint("The user needs to service their bike.")
    unrelated_constraint.constraint_id = "c2"
    client = FakeLLMClient()

    LLMConstrainedAnswerGenerator(client).generate(
        make_query_info(),
        [constrained],
        [relevant_constraint, unrelated_constraint],
        [make_memory("m1", "The user likes spicy hotpot.")],
    )

    assert "The user cannot eat spicy food." in client.prompt
    assert "service their bike" not in client.prompt


def test_llm_answer_generator_falls_back_when_client_fails():
    class FailingLLMClient:
        def complete_json(self, prompt):
            raise RuntimeError("network failed")

    answer = LLMConstrainedAnswerGenerator(FailingLLMClient()).generate(
        make_query_info(),
        [make_judgment("NOISE")],
        [],
    )

    assert "reliable basis" in answer
    assert "do not match" not in answer

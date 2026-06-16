from cadmr.answer_generator import ConstrainedAnswerGenerator
from cadmr.schemas import ActiveConstraint, MemoryJudgment, QueryInfo


def make_query_info(query: str = "今晚还能吃火锅吗？") -> QueryInfo:
    return QueryInfo(
        query=query,
        query_intent="unknown",
        query_scope=["diet", "health"],
        resolved_subject="user",
        requires_action=True,
        requires_plan=False,
        possible_old_premises=[],
    )


def make_constraint(content: str = "医生说用户接下来四周不能吃辣") -> ActiveConstraint:
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


def test_constrained_food_answer():
    answer = ConstrainedAnswerGenerator().generate(
        make_query_info(),
        [make_judgment("CONSTRAINED")],
        [make_constraint()],
    )

    assert "不建议" in answer or "不适合" in answer
    assert "不能吃辣" in answer
    assert any(word in answer for word in ["清淡", "清汤", "番茄锅", "菌汤"])
    assert "推荐重辣火锅" not in answer


def test_usable_answer():
    answer = ConstrainedAnswerGenerator().generate(
        make_query_info("我应该怎么通勤？"),
        [make_judgment("USABLE")],
        [],
    )

    assert "可以作为回答依据" in answer or "可用记忆" in answer


def test_no_judgments_answer():
    answer = ConstrainedAnswerGenerator().generate(make_query_info(), [], [])

    assert "没有找到可用的历史记忆或限制条件" in answer


def test_noise_only_answer():
    answer = ConstrainedAnswerGenerator().generate(
        make_query_info(),
        [make_judgment("NOISE")],
        [],
    )

    assert "不匹配" in answer
    assert "不应作为当前回答依据" in answer

from cadmr.schemas import ActiveConstraint, MemoryJudgment
from cadmr.verifier import AnswerVerifier


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


def make_constrained_judgment() -> MemoryJudgment:
    return MemoryJudgment(
        memory_id="m1",
        usage_status="CONSTRAINED",
        truth_status="still_true",
        actionability="blocked",
        blocked_by=["c1"],
        replaced_by=[],
        allowed_use="Use only as background.",
        forbidden_use="Do not use as direct action basis.",
        reason="Blocked by active constraint.",
    )


def test_constrained_spicy_food_answer_violation():
    result = AnswerVerifier().verify(
        "今晚可以吃麻辣火锅。",
        [make_constrained_judgment()],
        [make_constraint()],
    )

    assert result["pass"] is False
    assert result["violations"]


def test_constrained_spicy_food_safe_answer():
    result = AnswerVerifier().verify(
        "今晚不建议吃麻辣火锅，可以选择清淡饮食。",
        [make_constrained_judgment()],
        [make_constraint()],
    )

    assert result["pass"] is True


def test_goal_plan_missing_components():
    result = AnswerVerifier().verify(
        "建议做脱敏案例。",
        [],
        [],
        {"required_plan_components": ["脱敏案例", "备用方案", "风险控制"]},
    )

    assert result["pass"] is False
    assert "备用方案" in result["missing_components"]
    assert "风险控制" in result["missing_components"]

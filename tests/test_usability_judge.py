from cadmr.schemas import ActiveConstraint, OrdinaryMemory, QueryInfo
from cadmr.usability_judge import MemoryUsabilityJudge


def make_memory(
    memory_id: str = "m1",
    content: str = "用户当前在成都出差",
    subject: str = "user",
    scope: list[str] | None = None,
) -> OrdinaryMemory:
    return OrdinaryMemory(
        memory_id=memory_id,
        content=content,
        subject=subject,
        scope=scope or ["location", "transport"],
        stability="long_term",
        status="active",
        confidence=0.9,
        evidence_ids=["i1"],
        created_at="2026-06-16T00:00:00Z",
        updated_at="2026-06-16T00:00:00Z",
    )


def make_constraint(
    constraint_id: str = "c1",
    scope: list[str] | None = None,
    strength: str = "hard",
    status: str = "active",
) -> ActiveConstraint:
    return ActiveConstraint(
        constraint_id=constraint_id,
        content="医生说用户接下来四周不能吃辣",
        subject="user",
        scope=scope or ["health", "diet"],
        priority="high",
        strength=strength,
        valid_time={},
        status=status,
        source="user_input",
        confidence=0.9,
        evidence_ids=["i1"],
        created_at="2026-06-16T00:00:00Z",
        updated_at="2026-06-16T00:00:00Z",
    )


def make_query_info(
    query_scope: list[str],
    resolved_subject: str = "user",
    query: str = "今晚还能吃火锅吗？",
) -> QueryInfo:
    return QueryInfo(
        query=query,
        query_intent="unknown",
        query_scope=query_scope,
        resolved_subject=resolved_subject,
        requires_action=True,
        requires_plan=False,
        possible_old_premises=[],
    )


def test_usable_memory():
    memory = make_memory()
    query_info = make_query_info(["location"])

    judgments = MemoryUsabilityJudge().judge(query_info, [memory], [])

    assert judgments[0].usage_status == "USABLE"


def test_constrained_memory_food_case():
    memory = make_memory(
        content="用户喜欢重辣口味，尤其喜欢川菜和火锅",
        scope=["diet", "preference"],
    )
    constraint = make_constraint()
    query_info = make_query_info(["diet", "health"])

    judgments = MemoryUsabilityJudge().judge(query_info, [memory], [constraint])
    judgment = judgments[0]

    assert judgment.usage_status == "CONSTRAINED"
    assert judgment.truth_status == "still_true"
    assert judgment.actionability == "blocked"
    assert constraint.constraint_id in judgment.blocked_by
    assert judgment.forbidden_use


def test_subject_mismatch_noise():
    memory = make_memory(scope=["health", "diet"])
    query_info = make_query_info(["diet"], resolved_subject="cat")

    judgments = MemoryUsabilityJudge().judge(query_info, [memory], [])

    assert judgments[0].usage_status == "NOISE"


def test_scope_mismatch_noise():
    memory = make_memory(scope=["finance"])
    query_info = make_query_info(["diet"])

    judgments = MemoryUsabilityJudge().judge(query_info, [memory], [])

    assert judgments[0].usage_status == "NOISE"


def test_soft_constraint_requires_adjustment():
    memory = make_memory(scope=["diet", "preference"])
    constraint = make_constraint(strength="soft")
    query_info = make_query_info(["diet"])

    judgments = MemoryUsabilityJudge().judge(query_info, [memory], [constraint])

    assert judgments[0].usage_status == "CONSTRAINED"
    assert judgments[0].actionability == "requires_adjustment"


def test_stale_memory_judgment():
    memory = make_memory(scope=["location", "transport"])
    memory.status = "stale"
    query_info = make_query_info(["location"])

    judgments = MemoryUsabilityJudge().judge(query_info, [memory], [])

    assert judgments[0].usage_status == "STALE"
    assert judgments[0].actionability == "not_actionable"
    assert judgments[0].forbidden_use

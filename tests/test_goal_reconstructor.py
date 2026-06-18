from cadmr.goal_reconstructor import GoalReconstructor
from cadmr.schemas import ActiveConstraint, QueryInfo


def make_query_info(query: str) -> QueryInfo:
    return QueryInfo(
        query=query,
        query_intent="unknown",
        query_scope=["work", "project"],
        resolved_subject="user",
        requires_action=True,
        requires_plan=True,
        possible_old_premises=[],
    )


def make_constraint(content: str) -> ActiveConstraint:
    return ActiveConstraint(
        constraint_id="c1",
        content=content,
        subject="user",
        scope=["privacy", "safety", "work", "project"],
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


def test_external_demo_privacy_constraint_case_is_noop_without_llm_reconstructor():
    plan = GoalReconstructor().reconstruct(
        make_query_info("帮我安排一个最稳妥的演示方案。"),
        [],
        [make_constraint("明天给合作方和管理老师演示，不能展示真实数据，需要脱敏。")],
    )

    assert plan["needs_goal_reconstruction"] is False
    assert plan["required_plan_components"] == []
    assert plan["forbidden_actions"] == []
    assert plan["reason"] == "No goal reconstructor is configured."


def test_normal_plan_case_is_noop_without_llm_reconstructor():
    plan = GoalReconstructor().reconstruct(make_query_info("帮我安排一个学习计划。"), [], [])

    assert plan["needs_goal_reconstruction"] is False
    assert plan["required_plan_components"] == []


def test_arrangement_word_alone_does_not_trigger_goal_reconstruction():
    plan = GoalReconstructor().reconstruct(make_query_info("帮我安排今晚吃什么。"), [], [])

    assert plan["needs_goal_reconstruction"] is False


def test_no_goal_reconstruction_case():
    plan = GoalReconstructor().reconstruct(make_query_info("今晚还能吃火锅吗？"), [], [])

    assert plan["needs_goal_reconstruction"] is False

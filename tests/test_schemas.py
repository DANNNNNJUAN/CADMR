from cadmr.schemas import (
    ActiveConstraint,
    MemoryJudgment,
    MemorySignal,
    OrdinaryMemory,
    PipelineResult,
    QueryInfo,
    RawInteraction,
    WriteDecisionResult,
)


def test_raw_interaction_can_be_instantiated():
    interaction = RawInteraction(
        interaction_id="i1",
        timestamp="2026-06-16T00:00:00Z",
        speaker="user",
        text="I prefer quiet cafes.",
    )

    assert interaction.interaction_id == "i1"


def test_ordinary_memory_can_be_instantiated():
    memory = OrdinaryMemory(
        memory_id="m1",
        content="The user prefers quiet cafes.",
        subject="user",
        scope=["preference", "places"],
        stability="stable",
        status="active",
        confidence=0.9,
        evidence_ids=["i1"],
        created_at="2026-06-16T00:00:00Z",
        updated_at="2026-06-16T00:00:00Z",
    )

    assert memory.scope == ["preference", "places"]


def test_active_constraint_can_be_instantiated():
    constraint = ActiveConstraint(
        constraint_id="c1",
        content="The user has only 30 minutes.",
        subject="user",
        scope=["time"],
        priority="high",
        strength="hard",
        valid_time={"start": "2026-06-16T00:00:00Z", "end": None},
        status="active",
        source="user",
        confidence=0.95,
        evidence_ids=["i2"],
        created_at="2026-06-16T00:00:00Z",
        updated_at="2026-06-16T00:00:00Z",
    )

    assert constraint.priority == "high"


def test_memory_signal_can_be_instantiated():
    signal = MemorySignal(
        signal_id="s1",
        signal_type="ordinary_memory",
        content="The user likes morning workouts.",
        subject="user",
        scope=["preference", "fitness"],
        confidence=0.8,
        evidence_text="I like morning workouts.",
    )

    assert signal.signal_type == "ordinary_memory"


def test_write_decision_result_can_be_instantiated():
    decision = WriteDecisionResult(
        signal_id="s1",
        decision="WRITE_TO_ORDINARY_MEMORY",
        target_store="ordinary_memory",
        reason="Stable preference.",
    )

    assert decision.target_store == "ordinary_memory"


def test_query_info_can_be_instantiated():
    query_info = QueryInfo(
        query="Can I go running today?",
        query_intent="ask_for_action_advice",
        query_scope=["fitness", "health"],
        resolved_subject="user",
        requires_action=True,
        requires_plan=False,
        possible_old_premises=[],
    )

    assert query_info.requires_action is True


def test_memory_judgment_can_be_instantiated():
    judgment = MemoryJudgment(
        memory_id="m1",
        usage_status="CONSTRAINED",
        truth_status="plausible",
        actionability="limited",
        blocked_by=["c1"],
        replaced_by=[],
        allowed_use="Use as background preference.",
        forbidden_use="Do not recommend strenuous activity.",
        reason="A health constraint limits action advice.",
    )

    assert judgment.usage_status == "CONSTRAINED"


def test_pipeline_result_can_be_instantiated():
    result = PipelineResult(
        user_input="What should I eat?",
        signals=[],
        write_decisions=[],
        query_info=None,
        judgments=[],
        answer=None,
        structured_output={"version": 1, "judgments": []},
    )

    assert result.signals == []
    assert result.structured_output["version"] == 1

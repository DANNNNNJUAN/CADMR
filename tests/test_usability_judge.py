from cadmr.schemas import ActiveConstraint, OrdinaryMemory, QueryInfo
from cadmr.usability_judge import LLMUsabilityJudge, MemoryUsabilityJudge


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


def test_llm_usability_judge_can_constrain_without_scope_overlap():
    class FakeLLMClient:
        def __init__(self):
            self.prompt = ""

        def complete_json(self, prompt):
            self.prompt = prompt
            return {
                "judgments": [
                    {
                        "memory_id": "m1",
                        "usage_status": "CONSTRAINED",
                        "truth_status": "still_true",
                        "actionability": "blocked",
                        "blocked_by": ["c1"],
                        "replaced_by": [],
                        "allowed_use": "Use as background travel preference only.",
                        "forbidden_use": "Do not recommend long-distance driving.",
                        "reason": "The medical constraint limits long driving even though scopes differ.",
                    }
                ]
            }

    memory = make_memory(
        content="用户喜欢周末自驾游",
        scope=["travel"],
    )
    constraint = make_constraint(scope=["medical"])
    constraint.content = "医生说接下来两周不能长时间开车"
    query_info = make_query_info(["weekend_plan"], query="周末能不能自驾去杭州？")

    judge = LLMUsabilityJudge(FakeLLMClient())
    judgments = judge.judge(query_info, [memory], [constraint])

    assert judgments[0].usage_status == "CONSTRAINED"
    assert judgments[0].blocked_by == ["c1"]
    assert "long driving" in judgments[0].reason
    diagnostics = judge.last_diagnostics
    assert diagnostics["judge_type"] == "LLMUsabilityJudge"
    assert diagnostics["batches_succeeded"] == 1
    assert diagnostics["judgments_from_llm"] == 1
    assert diagnostics["fallback_used"] is False


def test_llm_usability_judge_prompt_distinguishes_stale_from_noise_and_constrained():
    class FakeLLMClient:
        def __init__(self):
            self.prompt = ""

        def complete_json(self, prompt):
            self.prompt = prompt
            return {"judgments": []}

    client = FakeLLMClient()
    memory = make_memory(
        content="The user lives in San Diego.",
        scope=["location"],
    )
    newer_memory = make_memory(
        memory_id="m2",
        content="The user now enjoys living in the mountains.",
        scope=["location"],
    )
    query_info = make_query_info(
        ["location"],
        query="Does the user still live in San Diego?",
    )

    LLMUsabilityJudge(client).judge(query_info, [memory, newer_memory], [])

    assert "Prefer STALE over NOISE" in client.prompt
    assert "Prefer STALE over CONSTRAINED" in client.prompt
    assert "newer/current replacement evidence" in client.prompt
    assert "old premise" in client.prompt
    assert "now enjoying the mountains" in client.prompt
    assert "sunny Saturday afternoon" in client.prompt
    assert "mark it SUSPENDED" in client.prompt
    assert "current action or recommendation questions" in client.prompt
    assert "prioritize that current query state" in client.prompt


def test_llm_usability_judge_includes_stale_target_pair_context():
    class FakeLLMClient:
        def __init__(self):
            self.prompt = ""

        def complete_json(self, prompt):
            self.prompt = prompt
            return {"judgments": []}

    client = FakeLLMClient()
    old_memory = make_memory(
        content="The user has been taking the subway to work every morning.",
        scope=["transport"],
    )
    new_memory = make_memory(
        memory_id="m2",
        content="The user now drives to work instead of taking the subway.",
        scope=["transport"],
    )
    query_info = make_query_info(
        ["transport"],
        query="Does the user still take the subway to work every morning?",
    )

    judge = LLMUsabilityJudge(
        client,
        stale_target_context={
            "m_old": "I take the subway to work every morning.",
            "m_new": "I now drive to work instead of taking the subway.",
        },
    )
    judge.judge(query_info, [old_memory, new_memory], [])

    assert "stale_target_context" in client.prompt
    assert "m_old" in client.prompt
    assert "m_new" in client.prompt
    assert "pair" in client.prompt
    assert old_memory.memory_id in client.prompt
    assert new_memory.memory_id in client.prompt
    assert "mark it noise" in client.prompt.lower()


def test_llm_usability_judge_falls_back_for_missing_memory_judgment():
    class FakeLLMClient:
        def complete_json(self, prompt):
            return {"judgments": []}

    memory = make_memory(scope=["diet", "preference"])
    constraint = make_constraint(scope=["diet", "health"])
    query_info = make_query_info(["diet"])

    judge = LLMUsabilityJudge(FakeLLMClient())
    judgments = judge.judge(query_info, [memory], [constraint])

    assert judgments[0].usage_status == "CONSTRAINED"
    assert judge.last_diagnostics["batches_total"] == 1
    assert judge.last_diagnostics["batches_succeeded"] == 1
    assert judge.last_diagnostics["batches_failed"] == 0
    assert judge.last_diagnostics["judgments_from_llm"] == 0
    assert judge.last_diagnostics["fallback_judgments"] == 1
    assert judge.last_diagnostics["fallback_used"] is True


def test_llm_usability_judge_batches_large_memory_sets():
    class FakeLLMClient:
        def __init__(self):
            self.calls = 0

        def complete_json(self, prompt):
            self.calls += 1
            return {"judgments": []}

    client = FakeLLMClient()
    memories = [make_memory(memory_id=f"m{index}", scope=["diet"]) for index in range(5)]
    query_info = make_query_info(["diet"])

    judge = LLMUsabilityJudge(client, max_memories_per_call=2)
    judgments = judge.judge(
        query_info,
        memories,
        [],
    )

    assert client.calls == 3
    assert len(judgments) == 5
    assert judge.last_diagnostics["batches_total"] == 3
    assert judge.last_diagnostics["batches_succeeded"] == 3
    assert len(judge.last_diagnostics["fallback_memory_ids_sample"]) == 5


def test_llm_usability_judge_falls_back_when_llm_json_parse_fails():
    class FailingLLMClient:
        def complete_json(self, prompt):
            raise ValueError("bad json")

    memory = make_memory(scope=["diet", "preference"])
    constraint = make_constraint(scope=["diet", "health"])
    query_info = make_query_info(["diet"])

    judge = LLMUsabilityJudge(FailingLLMClient())
    judgments = judge.judge(
        query_info,
        [memory],
        [constraint],
    )

    assert judgments[0].usage_status == "CONSTRAINED"
    assert judge.last_diagnostics["batches_total"] == 1
    assert judge.last_diagnostics["batches_succeeded"] == 0
    assert judge.last_diagnostics["batches_failed"] == 1
    assert judge.last_diagnostics["fallback_judgments"] == 1
    assert judge.last_diagnostics["fallback_used"] is True
    assert judge.last_diagnostics["errors"][0]["error_type"] == "ValueError"
    assert "bad json" in judge.last_diagnostics["errors"][0]["message"]

from cadmr.schemas import ActiveConstraint, MemoryJudgment
from cadmr.verifier import AnswerVerifier, LLMAnswerVerifier


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


def make_usable_judgment() -> MemoryJudgment:
    return MemoryJudgment(
        memory_id="m1",
        usage_status="USABLE",
        truth_status="still_true",
        actionability="actionable",
        blocked_by=[],
        replaced_by=[],
        allowed_use="Use as current grounding.",
        forbidden_use="",
        reason="Relevant current evidence.",
    )


def test_default_answer_verifier_is_noop_and_does_not_keyword_scan():
    result = AnswerVerifier().verify(
        "今晚可以吃麻辣火锅。",
        [make_constrained_judgment()],
        [make_constraint()],
    )

    assert result["pass"] is True
    assert result["violations"] == []
    assert result["verifier_type"] == "noop"


def test_constrained_spicy_food_safe_answer():
    result = AnswerVerifier().verify(
        "今晚不建议吃麻辣火锅，可以选择清淡饮食。",
        [make_constrained_judgment()],
        [make_constraint()],
    )

    assert result["pass"] is True


def test_default_answer_verifier_does_not_check_goal_plan_components():
    result = AnswerVerifier().verify(
        "建议做脱敏案例。",
        [],
        [],
        {"required_plan_components": ["脱敏案例", "备用方案", "风险控制"]},
    )

    assert result["pass"] is True
    assert result["missing_components"] == []


def test_llm_answer_verifier_uses_structured_output_and_normalizes_result():
    class FakeLLMClient:
        def __init__(self):
            self.prompt = ""

        def complete_json(self, prompt):
            self.prompt = prompt
            return {
                "pass": False,
                "violations": [
                    {
                        "type": "constraint_violation",
                        "evidence": "今晚可以吃麻辣火锅",
                        "related_id": "c1",
                    }
                ],
                "missing_components": [],
                "needs_revision": True,
                "reason": "Answer recommends a forbidden action.",
            }

    client = FakeLLMClient()
    structured_output = {
        "answer": "今晚可以吃麻辣火锅。",
        "judgments": [make_constrained_judgment().model_dump()],
        "retrieved_constraints": [make_constraint().model_dump()],
    }

    result = LLMAnswerVerifier(client).verify(
        "今晚可以吃麻辣火锅。",
        [make_constrained_judgment()],
        [make_constraint()],
        structured_output=structured_output,
    )

    assert result["pass"] is False
    assert result["needs_revision"] is True
    assert result["verifier_type"] == "llm"
    assert result["violations"][0]["type"] == "constraint_violation"
    assert "CADMR structured_output" in client.prompt
    assert "今晚可以吃麻辣火锅" in client.prompt
    assert "Treat the supplied judgments as authoritative" in client.prompt
    assert "you must not report using that memory as noise_use" in client.prompt


def test_llm_answer_verifier_pass_requires_no_violations_or_missing_components():
    class FakeLLMClient:
        def complete_json(self, prompt):
            return {
                "pass": True,
                "violations": [],
                "missing_components": ["备用方案"],
                "needs_revision": False,
                "reason": "Model claimed pass but reported a missing component.",
            }

    result = LLMAnswerVerifier(FakeLLMClient()).verify("建议做脱敏案例。", [], [])

    assert result["pass"] is False
    assert result["needs_revision"] is True
    assert result["missing_components"] == ["备用方案"]


def test_llm_answer_verifier_removes_usable_memory_false_violations():
    class FakeLLMClient:
        def complete_json(self, prompt):
            return {
                "pass": False,
                "violations": [
                    {
                        "type": "noise_use",
                        "evidence": "User enjoys mountains.",
                        "related_id": "m1",
                    },
                    {
                        "type": "stale_memory_use",
                        "evidence": "User enjoys mountains.",
                        "related_id": "m1",
                    },
                ],
                "missing_components": [],
                "needs_revision": True,
                "reason": "Incorrectly rejudged usable evidence.",
            }

    structured_output = {
        "answer": "Use mountain-friendly activities.",
        "judgments": [make_usable_judgment().model_dump()],
    }

    result = LLMAnswerVerifier(FakeLLMClient()).verify(
        "Use mountain-friendly activities.",
        [make_usable_judgment()],
        [],
        structured_output=structured_output,
    )

    assert result["pass"] is True
    assert result["needs_revision"] is False
    assert result["violations"] == []
    assert "removed because they contradicted" in result["reason"]
    assert "Incorrectly rejudged" not in result["reason"]


def test_llm_answer_verifier_allows_stale_rejection_language():
    class FakeLLMClient:
        def complete_json(self, prompt):
            return {
                "pass": False,
                "violations": [
                    {
                        "type": "stale_memory_use",
                        "evidence": "I would not assume the older premise is still current.",
                        "related_id": "m_old",
                    }
                ],
                "missing_components": [],
                "needs_revision": True,
                "reason": "Incorrectly flagged stale rejection.",
            }

    stale = make_constrained_judgment()
    stale.memory_id = "m_old"
    stale.usage_status = "STALE"
    stale.blocked_by = []
    structured_output = {
        "answer": "I would not assume the older premise is still current.",
        "judgments": [stale.model_dump()],
    }

    result = LLMAnswerVerifier(FakeLLMClient()).verify(
        "I would not assume the older premise is still current.",
        [stale],
        [],
        structured_output=structured_output,
    )

    assert result["pass"] is True
    assert result["violations"] == []
    assert "removed because they contradicted" in result["reason"]
    assert "Incorrectly flagged" not in result["reason"]


def test_llm_answer_verifier_returns_fallback_when_llm_json_parse_fails():
    class FailingLLMClient:
        def complete_json(self, prompt):
            raise ValueError("bad json")

    result = LLMAnswerVerifier(FailingLLMClient()).verify(
        "回答文本",
        [make_constrained_judgment()],
        [make_constraint()],
    )

    assert result["pass"] is False
    assert result["needs_revision"] is True
    assert result["verifier_type"] == "llm"
    assert "bad json" in result["reason"]

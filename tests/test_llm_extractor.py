from cadmr.extractor import LLMMemorySignalExtractor, MockLLMClient


def extract_with_response(text: str, response: dict):
    extractor = LLMMemorySignalExtractor(MockLLMClient(response))
    return extractor.extract(text)


def signal_types(signals):
    return {signal.signal_type for signal in signals}


def test_llm_extractor_prompt_is_english_and_discourages_query_as_constraint():
    extractor = LLMMemorySignalExtractor(MockLLMClient({"signals": []}))

    prompt = extractor._build_prompt("帮我安排一个方案。")

    assert "Extract CADMR memory signals" in prompt
    assert "query_intent" in prompt
    assert "is query_intent, not active_constraint" in prompt
    assert "从用户输入" not in prompt
    assert "用户输入" not in prompt


def test_llm_extractor_accepts_correct_ordinary_memory():
    signals = extract_with_response(
        "我喜欢重辣口味，尤其喜欢川菜和火锅。",
        {
            "signals": [
                {
                    "signal_type": "ordinary_memory",
                    "content": "用户喜欢重辣口味，尤其喜欢川菜和火锅",
                    "subject": "user",
                    "scope": ["diet", "preference"],
                    "confidence": 0.9,
                    "evidence_text": "我喜欢重辣口味，尤其喜欢川菜和火锅。",
                }
            ]
        },
    )

    assert len(signals) == 1
    assert signals[0].signal_type == "ordinary_memory"
    assert "diet" in signals[0].scope


def test_llm_extractor_accepts_constraint_and_query():
    signals = extract_with_response(
        "医生说我接下来四周不能吃辣，那今晚还能吃火锅吗？",
        {
            "signals": [
                {
                    "signal_type": "active_constraint",
                    "content": "用户接下来四周不能吃辣",
                    "subject": "user",
                    "scope": ["health", "diet"],
                    "confidence": 0.95,
                    "evidence_text": "医生说我接下来四周不能吃辣",
                },
                {
                    "signal_type": "query_intent",
                    "content": "用户询问今晚还能不能吃火锅",
                    "subject": "user",
                    "scope": ["diet"],
                    "confidence": 0.85,
                    "evidence_text": "那今晚还能吃火锅吗？",
                },
            ]
        },
    )

    assert {"active_constraint", "query_intent"}.issubset(signal_types(signals))
    constraint = next(signal for signal in signals if signal.signal_type == "active_constraint")
    assert {"health", "diet"}.intersection(constraint.scope)


def test_validator_does_not_repair_hypothetical_memory_pollution():
    signals = extract_with_response(
        "如果我以后搬去上海，应该怎么通勤？",
        {
            "signals": [
                {
                    "signal_type": "ordinary_memory",
                    "content": "用户以后会搬去上海",
                    "subject": "user",
                    "scope": ["location"],
                    "confidence": 0.9,
                    "evidence_text": "如果我以后搬去上海",
                }
            ]
        },
    )

    assert signals[0].signal_type == "ordinary_memory"


def test_validator_does_not_repair_question_premise_pollution():
    signals = extract_with_response(
        "既然我每天骑车上班，帮我规划路线。",
        {
            "signals": [
                {
                    "signal_type": "ordinary_memory",
                    "content": "用户每天骑车上班",
                    "subject": "user",
                    "scope": ["transport"],
                    "confidence": 0.9,
                    "evidence_text": "既然我每天骑车上班",
                }
            ]
        },
    )

    assert signals[0].signal_type == "ordinary_memory"


def test_validator_does_not_repair_uncertain_intention_pollution():
    signals = extract_with_response(
        "我可能想换研究方向。",
        {
            "signals": [
                {
                    "signal_type": "ordinary_memory",
                    "content": "用户想换研究方向",
                    "subject": "user",
                    "scope": ["work", "project"],
                    "confidence": 0.9,
                    "evidence_text": "我可能想换研究方向。",
                }
            ]
        },
    )

    assert signals[0].signal_type == "ordinary_memory"


def test_validator_drops_invalid_signal_type():
    signals = extract_with_response(
        "我喜欢火锅。",
        {"signals": [{"signal_type": "random_type", "content": "用户喜欢火锅"}]},
    )

    assert signals == []


def test_validator_fills_missing_fields():
    signals = extract_with_response(
        "我喜欢火锅。",
        {"signals": [{"signal_type": "ordinary_memory", "content": "用户喜欢火锅"}]},
    )

    assert len(signals) == 1
    assert signals[0].subject == "user"
    assert signals[0].scope == ["general"]
    assert signals[0].confidence == 0.8
    assert signals[0].evidence_text == "我喜欢火锅。"


def test_llm_extractor_returns_empty_signals_when_llm_json_parse_fails():
    class FailingLLMClient:
        def complete_json(self, prompt):
            raise ValueError("bad json")

    signals = LLMMemorySignalExtractor(FailingLLMClient()).extract("我喜欢火锅。")

    assert signals == []


def test_validator_normalizes_scope_format_without_semantic_aliasing():
    signals = extract_with_response(
        "我今晚还能按自己的偏好去吃重辣火锅吗？",
        {
            "signals": [
                {
                    "signal_type": "question_premise",
                    "content": "用户询问是否还能按偏好吃重辣火锅",
                    "subject": "user",
                    "scope": ["dinner", "preferences"],
                    "confidence": 0.8,
                    "evidence_text": "我今晚还能按自己的偏好去吃重辣火锅吗？",
                }
            ]
        },
    )

    assert signals[0].scope == ["dinner", "preferences"]


def test_validator_does_not_promote_constraint_like_ordinary_memory():
    signals = extract_with_response(
        "医生说我接下来两周不能长时间开车。",
        {
            "signals": [
                {
                    "signal_type": "ordinary_memory",
                    "content": "医生说用户接下来两周不能长时间开车",
                    "subject": "user",
                    "scope": ["medical", "driving"],
                    "confidence": 0.9,
                    "evidence_text": "医生说我接下来两周不能长时间开车。",
                }
            ]
        },
    )

    assert signals[0].signal_type == "ordinary_memory"


def test_validator_does_not_promote_budget_limit_to_active_constraint():
    signals = extract_with_response(
        "这周末娱乐预算最多 500 元。",
        {
            "signals": [
                {
                    "signal_type": "ordinary_memory",
                    "content": "用户这周末娱乐预算最多 500 元",
                    "subject": "user",
                    "scope": ["budget", "entertainment"],
                    "confidence": 0.9,
                    "evidence_text": "这周末娱乐预算最多 500 元。",
                }
            ]
        },
    )

    assert signals[0].signal_type == "ordinary_memory"
    assert signals[0].scope == ["budget", "entertainment"]


def test_llm_extractor_accepts_llm_classified_budget_constraint():
    signals = extract_with_response(
        "这周末娱乐预算最多 500 元。",
        {
            "signals": [
                {
                    "signal_type": "active_constraint",
                    "content": "用户这周末娱乐预算最多 500 元",
                    "subject": "user",
                    "scope": ["budget", "entertainment"],
                    "confidence": 0.9,
                    "evidence_text": "这周末娱乐预算最多 500 元。",
                }
            ]
        },
    )

    assert signals[0].signal_type == "active_constraint"
    assert signals[0].scope == ["budget", "entertainment"]


def test_validator_deduplicates_identical_signals():
    item = {
        "signal_type": "ordinary_memory",
        "content": "用户喜欢火锅",
        "subject": "user",
        "scope": ["diet", "preference"],
        "confidence": 0.9,
        "evidence_text": "我喜欢火锅。",
    }

    signals = extract_with_response("我喜欢火锅。", {"signals": [item, item]})

    assert len(signals) == 1

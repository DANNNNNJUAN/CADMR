from cadmr.extractor import RuleBasedMemorySignalExtractor, infer_scope


def signal_types(signals):
    return {signal.signal_type for signal in signals}


def merged_scopes(signals):
    return {scope for signal in signals for scope in signal.scope}


def test_infer_scope_merges_multiple_scope_groups():
    scopes = set(infer_scope("医生说我不能吃辣"))

    assert {"diet", "preference", "health"}.issubset(scopes)


def test_rule_based_extractor_detects_ordinary_preference():
    extractor = RuleBasedMemorySignalExtractor()
    signals = extractor.extract("我喜欢重辣口味，尤其喜欢川菜和火锅。")

    assert "ordinary_memory" in signal_types(signals)
    assert {"diet", "preference"}.intersection(merged_scopes(signals))


def test_rule_based_extractor_detects_health_constraint():
    extractor = RuleBasedMemorySignalExtractor()
    signals = extractor.extract("医生说我接下来四周不能吃辣。")

    assert "active_constraint" in signal_types(signals)
    assert {"health", "diet"}.intersection(merged_scopes(signals))


def test_rule_based_extractor_detects_query_intent():
    extractor = RuleBasedMemorySignalExtractor()
    signals = extractor.extract("今晚还能吃火锅吗？")

    assert "query_intent" in signal_types(signals)


def test_rule_based_extractor_detects_constraint_and_query():
    extractor = RuleBasedMemorySignalExtractor()
    signals = extractor.extract("医生说我接下来四周不能吃辣，那今晚还能吃火锅吗？")

    assert {"active_constraint", "query_intent"}.issubset(signal_types(signals))


def test_rule_based_extractor_detects_hypothetical():
    extractor = RuleBasedMemorySignalExtractor()
    signals = extractor.extract("如果我以后搬去上海，应该怎么通勤？")

    assert "hypothetical" in signal_types(signals)


def test_rule_based_extractor_detects_question_premise():
    extractor = RuleBasedMemorySignalExtractor()
    signals = extractor.extract("既然我每天骑车上班，帮我规划路线。")

    assert "question_premise" in signal_types(signals)
    assert "query_intent" in signal_types(signals)


def test_rule_based_extractor_detects_uncertain_intention():
    extractor = RuleBasedMemorySignalExtractor()
    signals = extractor.extract("我可能想换研究方向。")

    assert "uncertain_intention" in signal_types(signals)


def test_rule_based_extractor_detects_project_background_memory():
    extractor = RuleBasedMemorySignalExtractor()
    signals = extractor.extract("我正在准备长期记忆系统 demo。")

    assert "ordinary_memory" in signal_types(signals)
    assert {"work", "project"}.intersection(merged_scopes(signals))

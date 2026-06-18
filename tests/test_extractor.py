from cadmr.extractor import RuleBasedMemorySignalExtractor, infer_scope


def test_infer_scope_returns_general_without_keyword_rules():
    assert infer_scope("医生说我不能吃辣") == ["general"]
    assert infer_scope("我现在附近有什么餐厅可以考虑？") == ["general"]


def test_rule_based_extractor_is_noop_for_ordinary_preference():
    extractor = RuleBasedMemorySignalExtractor()

    assert extractor.extract("我喜欢重辣口味，尤其喜欢川菜和火锅。") == []


def test_rule_based_extractor_is_noop_for_constraints():
    extractor = RuleBasedMemorySignalExtractor()

    assert extractor.extract("医生说我接下来四周不能吃辣。") == []
    assert extractor.extract("这周末娱乐预算最多 500 元，需要控制支出。") == []
    assert extractor.extract("医生说我接下来两周不能长时间开车，需要避免久坐和疲劳驾驶。") == []


def test_rule_based_extractor_is_noop_for_query_and_premise_markers():
    extractor = RuleBasedMemorySignalExtractor()

    assert extractor.extract("今晚还能吃火锅吗？") == []
    assert extractor.extract("如果我以后搬去上海，应该怎么通勤？") == []
    assert extractor.extract("既然我每天骑车上班，帮我规划路线。") == []
    assert extractor.extract("我可能想换研究方向。") == []

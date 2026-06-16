from cadmr.pipeline import CADMRPipeline
from cadmr.stores import ActiveConstraintStore, OrdinaryMemoryStore, RawInteractionLog


def make_pipeline(tmp_path):
    raw_log = RawInteractionLog(tmp_path / "raw_interaction_log.jsonl")
    ordinary_store = OrdinaryMemoryStore(tmp_path / "ordinary_memory.json")
    constraint_store = ActiveConstraintStore(tmp_path / "active_constraints.json")
    pipeline = CADMRPipeline(
        raw_log=raw_log,
        ordinary_store=ordinary_store,
        constraint_store=constraint_store,
    )
    return pipeline, raw_log, ordinary_store, constraint_store


def signal_types(result):
    return {signal.signal_type for signal in result.signals}


def decisions_by_signal_type(result):
    return {
        signal.signal_type: decision.decision
        for signal, decision in zip(result.signals, result.write_decisions, strict=True)
    }


def test_pipeline_writes_ordinary_memory(tmp_path):
    pipeline, raw_log, ordinary_store, constraint_store = make_pipeline(tmp_path)

    result = pipeline.run("我喜欢重辣口味，尤其喜欢川菜和火锅。")

    assert len(raw_log.list_all()) == 1
    assert len(ordinary_store.list_all()) == 1
    assert len(constraint_store.list_all()) == 0
    assert "ordinary_memory" in signal_types(result)
    assert "WRITE_TO_ORDINARY_MEMORY" in [
        decision.decision for decision in result.write_decisions
    ]


def test_pipeline_writes_active_constraint(tmp_path):
    pipeline, raw_log, ordinary_store, constraint_store = make_pipeline(tmp_path)

    result = pipeline.run("医生说我接下来四周不能吃辣。")

    assert len(raw_log.list_all()) == 1
    assert len(ordinary_store.list_all()) == 0
    assert len(constraint_store.list_all()) == 1
    assert "active_constraint" in signal_types(result)
    assert "WRITE_TO_ACTIVE_CONSTRAINT" in [
        decision.decision for decision in result.write_decisions
    ]


def test_pipeline_handles_constraint_plus_query(tmp_path):
    pipeline, raw_log, ordinary_store, constraint_store = make_pipeline(tmp_path)

    result = pipeline.run("医生说我接下来四周不能吃辣，那今晚还能吃火锅吗？")

    assert len(raw_log.list_all()) == 1
    assert len(constraint_store.list_all()) == 1
    assert len(ordinary_store.list_all()) == 0
    assert {"active_constraint", "query_intent"}.issubset(signal_types(result))
    assert decisions_by_signal_type(result)["query_intent"] == "DO_NOT_WRITE"
    assert result.query_info is not None
    assert result.answer is not None


def test_pipeline_does_not_write_query_intent_to_structured_stores(tmp_path):
    pipeline, raw_log, ordinary_store, constraint_store = make_pipeline(tmp_path)

    result = pipeline.run("今晚还能吃火锅吗？")

    assert len(raw_log.list_all()) == 1
    assert len(ordinary_store.list_all()) == 0
    assert len(constraint_store.list_all()) == 0
    assert result.query_info is not None


def test_pipeline_does_not_write_hypothetical_to_structured_stores(tmp_path):
    pipeline, raw_log, ordinary_store, constraint_store = make_pipeline(tmp_path)

    result = pipeline.run("如果我以后搬去上海，应该怎么通勤？")

    assert len(raw_log.list_all()) == 1
    assert len(ordinary_store.list_all()) == 0
    assert len(constraint_store.list_all()) == 0
    assert decisions_by_signal_type(result)["hypothetical"] == "DO_NOT_WRITE"


def test_pipeline_does_not_write_question_premise_to_structured_stores(tmp_path):
    pipeline, raw_log, ordinary_store, constraint_store = make_pipeline(tmp_path)

    result = pipeline.run("既然我每天骑车上班，帮我规划路线。")

    assert len(raw_log.list_all()) == 1
    assert len(ordinary_store.list_all()) == 0
    assert len(constraint_store.list_all()) == 0
    assert decisions_by_signal_type(result)["question_premise"] == "VERIFY_ONLY"


def test_pipeline_end_to_end_constrained_food_case(tmp_path):
    pipeline, _, ordinary_store, constraint_store = make_pipeline(tmp_path)

    first_result = pipeline.run("我喜欢重辣口味，尤其喜欢川菜和火锅。")

    assert len(ordinary_store.list_all()) == 1
    assert first_result.answer is None

    second_result = pipeline.run("医生说我接下来四周不能吃辣，那今晚还能吃火锅吗？")

    assert len(constraint_store.list_all()) == 1
    assert second_result.query_info is not None
    assert any(
        judgment.usage_status == "CONSTRAINED"
        for judgment in second_result.judgments
    )
    assert second_result.answer is not None
    assert "不建议" in second_result.answer or "不适合" in second_result.answer
    assert "不能吃辣" in second_result.answer or "当前限制" in second_result.answer
    assert any(
        word in second_result.answer
        for word in ["清淡", "清汤", "番茄锅", "菌汤"]
    )


def test_pipeline_marks_superseded_location_memory_stale(tmp_path):
    pipeline, _, ordinary_store, _ = make_pipeline(tmp_path)

    pipeline.run("我现在住在北京。")
    pipeline.run("我已经搬到上海了。")

    memories = ordinary_store.list_all()
    beijing_memory = next(memory for memory in memories if "北京" in memory.content)
    shanghai_memory = next(memory for memory in memories if "上海" in memory.content)

    assert len(memories) >= 2
    assert beijing_memory.status == "stale"
    assert shanghai_memory.status == "active"


def test_pipeline_end_to_end_stale_location_case(tmp_path):
    pipeline, _, ordinary_store, _ = make_pipeline(tmp_path)

    pipeline.run("我现在住在北京。")
    pipeline.run("我已经搬到上海了。")
    result = pipeline.run("我附近有什么餐厅？")

    memories = ordinary_store.list_all()
    beijing_memory = next(memory for memory in memories if "北京" in memory.content)
    shanghai_memory = next(memory for memory in memories if "上海" in memory.content)

    judgment_by_memory_id = {
        judgment.memory_id: judgment for judgment in result.judgments
    }
    if beijing_memory.memory_id in judgment_by_memory_id:
        assert judgment_by_memory_id[beijing_memory.memory_id].usage_status == "STALE"
    assert judgment_by_memory_id[shanghai_memory.memory_id].usage_status != "STALE"
    assert result.answer is not None
    assert "北京" not in result.answer


def test_pipeline_end_to_end_cat_referent_case(tmp_path):
    pipeline, _, _, _ = make_pipeline(tmp_path)

    pipeline.run("我只是轻度乳糖不耐受。")
    pipeline.run("我的猫刚从医院回来，兽医说饮食需要小心。")
    result = pipeline.run("既然我只是轻度乳糖不耐受，那它偶尔喝点牛奶应该没事吧？")

    assert result.query_info is not None
    assert result.query_info.resolved_subject == "cat"
    for judgment in result.judgments:
        if judgment.memory_id:
            assert judgment.usage_status != "USABLE" or "乳糖" not in result.answer
    assert result.answer is not None
    assert "乳糖不耐受" not in result.answer


def test_pipeline_end_to_end_goal_level_demo(tmp_path):
    pipeline, _, _, _ = make_pipeline(tmp_path)

    pipeline.run("这个长期记忆系统 demo 先给内部技术同事看，只要跑通原型就行。")
    pipeline.run("明天要给合作方和管理老师演示，不能展示真实用户数据，需要脱敏。")
    result = pipeline.run("帮我安排一个最稳妥的演示方案。")

    assert result.goal_plan is not None
    assert result.goal_plan["needs_goal_reconstruction"] is True
    assert any("真实" in action for action in result.goal_plan["forbidden_actions"])
    assert "脱敏案例" in result.goal_plan["required_plan_components"]
    assert "备用方案" in result.goal_plan["required_plan_components"]
    assert "风险控制" in result.goal_plan["required_plan_components"]
    assert result.answer is not None
    assert "展示真实用户数据" not in result.answer or "不要展示真实用户数据" in result.answer

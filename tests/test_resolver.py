from cadmr.resolver import ReferentTopicResolver
from cadmr.schemas import OrdinaryMemory, RawInteraction


def make_interaction(text: str) -> RawInteraction:
    return RawInteraction(
        interaction_id="i1",
        timestamp="2026-06-16T00:00:00Z",
        speaker="user",
        text=text,
    )


def make_memory(content: str) -> OrdinaryMemory:
    return OrdinaryMemory(
        memory_id="m1",
        content=content,
        subject="user",
        scope=["work", "project"],
        stability="long_term",
        status="active",
        confidence=0.9,
        evidence_ids=["i1"],
        created_at="2026-06-16T00:00:00Z",
        updated_at="2026-06-16T00:00:00Z",
    )


def test_cat_referent_case_is_noop_without_llm_resolver():
    resolver = ReferentTopicResolver()

    result = resolver.resolve(
        "既然我只是轻度乳糖不耐受，那它偶尔喝点牛奶应该没事吧？",
        [make_interaction("我的猫刚从医院回来，兽医说饮食需要小心。")],
        [],
        [],
    )

    assert result["resolved_subject"] == "user"
    assert result["current_topic"] == "general"


def test_father_and_mother_subject_cases_are_noop_without_llm_resolver():
    resolver = ReferentTopicResolver()

    father = resolver.resolve("我爸应该怎么安排体检？", [], [], [])
    mother = resolver.resolve("母亲明天怎么去医院？", [], [], [])

    assert father["resolved_subject"] == "user"
    assert mother["resolved_subject"] == "user"


def test_topic_return_case_is_noop_without_llm_resolver():
    resolver = ReferentTopicResolver()

    result = resolver.resolve(
        "继续刚才那个方案。",
        [],
        [make_memory("长期记忆系统 demo 演示流程")],
        [],
    )

    assert result["topic_status"] == "active"
    assert result["current_topic"] == "general"


def test_default_user_case():
    result = ReferentTopicResolver().resolve("今晚吃什么？", [], [], [])

    assert result["resolved_subject"] == "user"
    assert result["topic_status"] == "active"

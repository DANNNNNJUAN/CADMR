from cadmr.retrieval import MemoryRetriever
from cadmr.schemas import ActiveConstraint, OrdinaryMemory, QueryInfo
from cadmr.stores import ActiveConstraintStore, OrdinaryMemoryStore


def make_memory(memory_id: str, scope: list[str], status: str = "active") -> OrdinaryMemory:
    return OrdinaryMemory(
        memory_id=memory_id,
        content=f"Memory {memory_id}",
        subject="user",
        scope=scope,
        stability="long_term",
        status=status,
        confidence=0.9,
        evidence_ids=["i1"],
        created_at="2026-06-16T00:00:00Z",
        updated_at="2026-06-16T00:00:00Z",
    )


def make_constraint(
    constraint_id: str,
    scope: list[str],
    status: str = "active",
    priority: str = "high",
    strength: str = "hard",
) -> ActiveConstraint:
    return ActiveConstraint(
        constraint_id=constraint_id,
        content=f"Constraint {constraint_id}",
        subject="user",
        scope=scope,
        priority=priority,
        strength=strength,
        valid_time={},
        status=status,
        source="user_input",
        confidence=0.9,
        evidence_ids=["i1"],
        created_at="2026-06-16T00:00:00Z",
        updated_at="2026-06-16T00:00:00Z",
    )


def make_query_info(query_scope: list[str]) -> QueryInfo:
    return QueryInfo(
        query="今晚还能吃火锅吗？",
        query_intent="unknown",
        query_scope=query_scope,
        resolved_subject="user",
        requires_action=True,
        requires_plan=False,
        possible_old_premises=[],
    )


def make_retriever(tmp_path):
    ordinary_store = OrdinaryMemoryStore(tmp_path / "ordinary_memory.json")
    constraint_store = ActiveConstraintStore(tmp_path / "active_constraints.json")
    return MemoryRetriever(ordinary_store, constraint_store), ordinary_store, constraint_store


def test_retrieve_memories_by_scope(tmp_path):
    retriever, ordinary_store, _ = make_retriever(tmp_path)
    memory1 = make_memory("m1", ["diet", "preference"])
    memory2 = make_memory("m2", ["finance"])
    ordinary_store.add(memory1)
    ordinary_store.add(memory2)

    results = retriever.retrieve_memories(make_query_info(["diet"]))

    assert results == [memory1]


def test_retrieve_constraints_by_scope(tmp_path):
    retriever, _, constraint_store = make_retriever(tmp_path)
    constraint1 = make_constraint("c1", ["health", "diet"])
    constraint2 = make_constraint("c2", ["finance"])
    constraint_store.add(constraint1)
    constraint_store.add(constraint2)

    results = retriever.retrieve_constraints(make_query_info(["diet"]))

    assert results == [constraint1]


def test_expired_constraint_not_returned(tmp_path):
    retriever, _, constraint_store = make_retriever(tmp_path)
    expired = make_constraint("c1", ["health", "diet"], status="expired")
    constraint_store.add(expired)

    results = retriever.retrieve_constraints(make_query_info(["diet"]))

    assert results == []


def test_retrieve_returns_tuple(tmp_path):
    retriever, ordinary_store, constraint_store = make_retriever(tmp_path)
    memory = make_memory("m1", ["diet"])
    constraint = make_constraint("c1", ["diet"])
    ordinary_store.add(memory)
    constraint_store.add(constraint)

    memories, constraints = retriever.retrieve(make_query_info(["diet"]))

    assert memories == [memory]
    assert constraints == [constraint]


def test_constraint_to_memory_backtracking(tmp_path):
    retriever, ordinary_store, constraint_store = make_retriever(tmp_path)
    memory = make_memory("m1", ["diet", "preference"])
    memory.content = "用户喜欢重辣火锅"
    constraint = make_constraint("c1", ["health", "diet"])
    constraint.content = "医生说四周不能吃辣"
    ordinary_store.add(memory)
    constraint_store.add(constraint)

    memories, constraints = retriever.retrieve(make_query_info(["health"]))

    assert constraint in constraints
    assert memory in memories

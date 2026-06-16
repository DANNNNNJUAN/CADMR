from cadmr.schemas import ActiveConstraint, OrdinaryMemory, RawInteraction
from cadmr.stores import ActiveConstraintStore, OrdinaryMemoryStore, RawInteractionLog


def make_raw_interaction(interaction_id: str, text: str) -> RawInteraction:
    return RawInteraction(
        interaction_id=interaction_id,
        timestamp="2026-06-16T00:00:00Z",
        speaker="user",
        text=text,
    )


def make_ordinary_memory(
    memory_id: str,
    scope: list[str],
    status: str = "active",
) -> OrdinaryMemory:
    return OrdinaryMemory(
        memory_id=memory_id,
        content=f"Memory {memory_id}",
        subject="user",
        scope=scope,
        stability="stable",
        status=status,
        confidence=0.9,
        evidence_ids=["i1"],
        created_at="2026-06-16T00:00:00Z",
        updated_at="2026-06-16T00:00:00Z",
    )


def make_active_constraint(
    constraint_id: str,
    scope: list[str],
    status: str = "active",
) -> ActiveConstraint:
    return ActiveConstraint(
        constraint_id=constraint_id,
        content=f"Constraint {constraint_id}",
        subject="user",
        scope=scope,
        priority="high",
        strength="hard",
        valid_time={"start": "2026-06-16T00:00:00Z", "end": None},
        status=status,
        source="user",
        confidence=0.9,
        evidence_ids=["i1"],
        created_at="2026-06-16T00:00:00Z",
        updated_at="2026-06-16T00:00:00Z",
    )


def test_raw_interaction_log_append_and_list_all(tmp_path):
    log = RawInteractionLog(tmp_path / "raw_interaction_log.jsonl")
    interaction = make_raw_interaction("i1", "I prefer tea.")

    log.append(interaction)

    assert log.list_all() == [interaction]


def test_raw_interaction_log_preserves_append_order(tmp_path):
    log = RawInteractionLog(tmp_path / "raw_interaction_log.jsonl")
    first = make_raw_interaction("i1", "First")
    second = make_raw_interaction("i2", "Second")

    log.append(first)
    log.append(second)

    assert log.list_all() == [first, second]


def test_raw_interaction_log_clear(tmp_path):
    log = RawInteractionLog(tmp_path / "raw_interaction_log.jsonl")
    log.append(make_raw_interaction("i1", "I prefer tea."))

    log.clear()

    assert log.list_all() == []


def test_ordinary_memory_store_add_and_list_all(tmp_path):
    store = OrdinaryMemoryStore(tmp_path / "ordinary_memory.json")
    memory = make_ordinary_memory("m1", ["diet"])

    store.add(memory)

    assert store.list_all() == [memory]


def test_ordinary_memory_store_get_active(tmp_path):
    store = OrdinaryMemoryStore(tmp_path / "ordinary_memory.json")
    active = make_ordinary_memory("m1", ["diet"], status="active")
    suspended = make_ordinary_memory("m2", ["diet"], status="suspended")

    store.add(active)
    store.add(suspended)

    assert store.get_active() == [active]


def test_ordinary_memory_store_search_by_scope(tmp_path):
    store = OrdinaryMemoryStore(tmp_path / "ordinary_memory.json")
    diet = make_ordinary_memory("m1", ["diet", "preference"])
    travel = make_ordinary_memory("m2", ["travel"])

    store.add(diet)
    store.add(travel)

    assert store.search_by_scope(["diet"]) == [diet]


def test_ordinary_memory_store_clear(tmp_path):
    store = OrdinaryMemoryStore(tmp_path / "ordinary_memory.json")
    store.add(make_ordinary_memory("m1", ["diet"]))

    store.clear()

    assert store.list_all() == []


def test_ordinary_memory_update(tmp_path):
    store = OrdinaryMemoryStore(tmp_path / "ordinary_memory.json")
    memory = make_ordinary_memory("m1", ["location"])
    store.add(memory)

    updated = memory.model_copy(update={"content": "Updated memory"})
    store.update(updated)

    assert store.list_all() == [updated]


def test_ordinary_memory_mark_stale(tmp_path):
    store = OrdinaryMemoryStore(tmp_path / "ordinary_memory.json")
    memory = make_ordinary_memory("m1", ["location"])
    store.add(memory)

    store.mark_stale("m1")

    memories = store.list_all()
    assert len(memories) == 1
    assert memories[0].status == "stale"
    assert store.get_active() == []


def test_active_constraint_store_add_and_list_all(tmp_path):
    store = ActiveConstraintStore(tmp_path / "active_constraints.json")
    constraint = make_active_constraint("c1", ["health"])

    store.add(constraint)

    assert store.list_all() == [constraint]


def test_active_constraint_store_get_active(tmp_path):
    store = ActiveConstraintStore(tmp_path / "active_constraints.json")
    active = make_active_constraint("c1", ["health"], status="active")
    expired = make_active_constraint("c2", ["health"], status="expired")

    store.add(active)
    store.add(expired)

    assert store.get_active() == [active]


def test_active_constraint_store_search_by_scope(tmp_path):
    store = ActiveConstraintStore(tmp_path / "active_constraints.json")
    health = make_active_constraint("c1", ["health", "diet"])
    privacy = make_active_constraint("c2", ["privacy"])

    store.add(health)
    store.add(privacy)

    assert store.search_by_scope(["health"]) == [health]


def test_active_constraint_store_clear(tmp_path):
    store = ActiveConstraintStore(tmp_path / "active_constraints.json")
    store.add(make_active_constraint("c1", ["health"]))

    store.clear()

    assert store.list_all() == []

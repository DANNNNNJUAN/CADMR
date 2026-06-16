from cadmr.schemas import MemorySignal
from cadmr.write_gate import MemoryWriteGate


def make_signal(
    signal_type: str,
    confidence: float = 0.8,
    content: str = "The user prefers tea.",
    signal_id: str = "s1",
    evidence_text: str = "I prefer tea.",
) -> MemorySignal:
    return MemorySignal(
        signal_id=signal_id,
        signal_type=signal_type,
        content=content,
        subject="user",
        scope=["preference"],
        confidence=confidence,
        evidence_text=evidence_text,
    )


def test_ordinary_memory_high_confidence_writes_to_ordinary_memory():
    result = MemoryWriteGate().decide(
        make_signal("ordinary_memory", confidence=0.85)
    )

    assert result.decision == "WRITE_TO_ORDINARY_MEMORY"
    assert result.target_store == "ordinary_memory"


def test_ordinary_memory_low_confidence_writes_to_short_term_buffer():
    result = MemoryWriteGate().decide(
        make_signal("ordinary_memory", confidence=0.4)
    )

    assert result.decision == "WRITE_TO_SHORT_TERM_BUFFER"
    assert result.target_store == "short_term_buffer"


def test_active_constraint_high_confidence_writes_to_active_constraint():
    result = MemoryWriteGate().decide(
        make_signal("active_constraint", confidence=0.9)
    )

    assert result.decision == "WRITE_TO_ACTIVE_CONSTRAINT"
    assert result.target_store == "active_constraint"


def test_active_constraint_low_confidence_writes_to_short_term_buffer():
    result = MemoryWriteGate().decide(
        make_signal("active_constraint", confidence=0.5)
    )

    assert result.decision == "WRITE_TO_SHORT_TERM_BUFFER"
    assert result.target_store == "short_term_buffer"


def test_query_intent_is_not_written():
    result = MemoryWriteGate().decide(make_signal("query_intent"))

    assert result.decision == "DO_NOT_WRITE"
    assert result.target_store is None


def test_hypothetical_is_not_written():
    result = MemoryWriteGate().decide(make_signal("hypothetical"))

    assert result.decision == "DO_NOT_WRITE"
    assert result.target_store is None


def test_question_premise_is_verify_only():
    result = MemoryWriteGate().decide(make_signal("question_premise"))

    assert result.decision == "VERIFY_ONLY"
    assert result.target_store is None


def test_uncertain_intention_writes_to_short_term_buffer():
    result = MemoryWriteGate().decide(make_signal("uncertain_intention"))

    assert result.decision == "WRITE_TO_SHORT_TERM_BUFFER"
    assert result.target_store == "short_term_buffer"


def test_empty_content_is_not_written():
    result = MemoryWriteGate().decide(
        make_signal("ordinary_memory", content="   ")
    )

    assert result.decision == "DO_NOT_WRITE"
    assert result.target_store is None
    assert result.reason == "Empty content should not be written."


def test_missing_evidence_is_not_rejected_but_noted():
    result = MemoryWriteGate().decide(
        make_signal("ordinary_memory", evidence_text="   ")
    )

    assert result.decision == "WRITE_TO_ORDINARY_MEMORY"
    assert "Evidence text is missing." in result.reason


def test_decide_many_preserves_order_and_signal_ids():
    signals = [
        make_signal("ordinary_memory", signal_id="s1"),
        make_signal("active_constraint", signal_id="s2", confidence=0.9),
        make_signal("query_intent", signal_id="s3"),
    ]

    results = MemoryWriteGate().decide_many(signals)

    assert len(results) == 3
    assert [result.signal_id for result in results] == ["s1", "s2", "s3"]
    assert [result.decision for result in results] == [
        "WRITE_TO_ORDINARY_MEMORY",
        "WRITE_TO_ACTIVE_CONSTRAINT",
        "DO_NOT_WRITE",
    ]

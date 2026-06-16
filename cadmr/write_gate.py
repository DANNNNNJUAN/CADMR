"""Write gate for routing memory signals to the right persistence target."""

from cadmr.schemas import MemorySignal, WriteDecisionResult


class MemoryWriteGate:
    """Decides whether and where a memory signal should be written."""

    def decide(self, signal: MemorySignal) -> WriteDecisionResult:
        evidence_note = " Evidence text is missing." if not signal.evidence_text.strip() else ""

        if not signal.content.strip():
            return WriteDecisionResult(
                signal_id=signal.signal_id,
                decision="DO_NOT_WRITE",
                target_store=None,
                reason="Empty content should not be written." + evidence_note,
            )

        signal_type = signal.signal_type

        if signal_type == "ordinary_memory":
            if signal.confidence < 0.5:
                return WriteDecisionResult(
                    signal_id=signal.signal_id,
                    decision="WRITE_TO_SHORT_TERM_BUFFER",
                    target_store="short_term_buffer",
                    reason=(
                        "Confidence is low, so the ordinary memory signal should be "
                        "kept in a short-term buffer instead of long-term memory."
                    )
                    + evidence_note,
                )
            return WriteDecisionResult(
                signal_id=signal.signal_id,
                decision="WRITE_TO_ORDINARY_MEMORY",
                target_store="ordinary_memory",
                reason=(
                    "Clear facts, stable preferences, habits, or long-term context can "
                    "be written to the ordinary memory store."
                )
                + evidence_note,
            )

        if signal_type == "active_constraint":
            if signal.confidence < 0.6:
                return WriteDecisionResult(
                    signal_id=signal.signal_id,
                    decision="WRITE_TO_SHORT_TERM_BUFFER",
                    target_store="short_term_buffer",
                    reason="Constraint confidence is insufficient, so it should wait for confirmation."
                    + evidence_note,
                )
            return WriteDecisionResult(
                signal_id=signal.signal_id,
                decision="WRITE_TO_ACTIVE_CONSTRAINT",
                target_store="active_constraint",
                reason="Current limits or high-priority constraints should be written to the active constraint store."
                + evidence_note,
            )

        if signal_type == "query_intent":
            return WriteDecisionResult(
                signal_id=signal.signal_id,
                decision="DO_NOT_WRITE",
                target_store=None,
                reason="Query intent is only for answering and is not a long-term fact or current constraint."
                + evidence_note,
            )

        if signal_type == "hypothetical":
            return WriteDecisionResult(
                signal_id=signal.signal_id,
                decision="DO_NOT_WRITE",
                target_store=None,
                reason="Hypothetical or counterfactual content must not be promoted to structured memory."
                + evidence_note,
            )

        if signal_type == "question_premise":
            return WriteDecisionResult(
                signal_id=signal.signal_id,
                decision="VERIFY_ONLY",
                target_store=None,
                reason="A premise embedded in a question should be verified before being treated as active memory."
                + evidence_note,
            )

        if signal_type == "uncertain_intention":
            return WriteDecisionResult(
                signal_id=signal.signal_id,
                decision="WRITE_TO_SHORT_TERM_BUFFER",
                target_store="short_term_buffer",
                reason="Uncertain intentions can be buffered but should not become stable long-term memory."
                + evidence_note,
            )

        return WriteDecisionResult(
            signal_id=signal.signal_id,
            decision="DO_NOT_WRITE",
            target_store=None,
            reason="Unknown signal_type should not be written." + evidence_note,
        )

    def decide_many(self, signals: list[MemorySignal]) -> list[WriteDecisionResult]:
        return [self.decide(signal) for signal in signals]

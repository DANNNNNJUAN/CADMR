"""Rule-based memory usability judgment."""

from cadmr.schemas import ActiveConstraint, MemoryJudgment, OrdinaryMemory, QueryInfo


def _has_scope_overlap(left: list[str], right: list[str]) -> bool:
    return bool(set(left).intersection(right))


class MemoryUsabilityJudge:
    """Judges how memories may be safely used for a query."""

    def judge(
        self,
        query_info: QueryInfo,
        memories: list[OrdinaryMemory],
        constraints: list[ActiveConstraint],
    ) -> list[MemoryJudgment]:
        return [
            self._judge_memory(query_info, memory, constraints)
            for memory in memories
        ]

    def _judge_memory(
        self,
        query_info: QueryInfo,
        memory: OrdinaryMemory,
        constraints: list[ActiveConstraint],
    ) -> MemoryJudgment:
        if not memory.content.strip():
            return MemoryJudgment(
                memory_id=memory.memory_id,
                usage_status="NOISE",
                truth_status="unknown",
                actionability="not_relevant",
                blocked_by=[],
                replaced_by=[],
                allowed_use="",
                forbidden_use="Do not use this memory for the current query.",
                reason="Empty memory content.",
            )

        if memory.status == "stale":
            return MemoryJudgment(
                memory_id=memory.memory_id,
                usage_status="STALE",
                truth_status="historically_true",
                actionability="not_actionable",
                blocked_by=[],
                replaced_by=[],
                allowed_use="This memory may be mentioned only as historical context.",
                forbidden_use="Do not use this stale memory as current grounding.",
                reason="Memory has been superseded by newer memory.",
            )

        if memory.subject != query_info.resolved_subject:
            return self._noise_judgment(
                memory,
                reason="Memory subject does not match query subject.",
            )

        if query_info.query_scope and not _has_scope_overlap(
            memory.scope,
            query_info.query_scope,
        ):
            return self._noise_judgment(
                memory,
                reason="Memory scope is not relevant to the current query.",
            )

        blocking_constraints = [
            constraint
            for constraint in constraints
            if constraint.status == "active"
            and constraint.subject == memory.subject
            and _has_scope_overlap(constraint.scope, memory.scope)
        ]
        if blocking_constraints:
            actionability = self._constraint_actionability(blocking_constraints)
            return MemoryJudgment(
                memory_id=memory.memory_id,
                usage_status="CONSTRAINED",
                truth_status="still_true",
                actionability=actionability,
                blocked_by=[
                    constraint.constraint_id for constraint in blocking_constraints
                ],
                replaced_by=[],
                allowed_use=(
                    "This memory may be acknowledged as historical or stable preference, "
                    "but it must be used under the active constraint."
                ),
                forbidden_use=self._constraint_forbidden_use(blocking_constraints),
                reason="Memory is still true but currently blocked or limited by active constraints.",
            )

        return MemoryJudgment(
            memory_id=memory.memory_id,
            usage_status="USABLE",
            truth_status="still_true",
            actionability="actionable",
            blocked_by=[],
            replaced_by=[],
            allowed_use="This memory can be used as current grounding.",
            forbidden_use="",
            reason="Memory is relevant and not blocked by active constraints.",
        )

    def _noise_judgment(self, memory: OrdinaryMemory, reason: str) -> MemoryJudgment:
        return MemoryJudgment(
            memory_id=memory.memory_id,
            usage_status="NOISE",
            truth_status="unknown",
            actionability="not_relevant",
            blocked_by=[],
            replaced_by=[],
            allowed_use="",
            forbidden_use="Do not use this memory for the current query.",
            reason=reason,
        )

    def _constraint_actionability(self, constraints: list[ActiveConstraint]) -> str:
        if any(
            constraint.strength == "hard" and constraint.priority == "high"
            for constraint in constraints
        ):
            return "blocked"
        if any(constraint.strength == "hard" for constraint in constraints):
            return "limited"
        return "requires_adjustment"

    def _constraint_forbidden_use(self, constraints: list[ActiveConstraint]) -> str:
        if any(constraint.strength == "hard" for constraint in constraints):
            return "Do not use this memory as direct action basis."
        return "Do not use this memory without adjusting for the active soft constraint."

"""Rule-based memory retrieval."""

from cadmr.schemas import ActiveConstraint, OrdinaryMemory, QueryInfo
from cadmr.stores import ActiveConstraintStore, OrdinaryMemoryStore


def _has_scope_overlap(left: list[str], right: list[str]) -> bool:
    return bool(set(left).intersection(right))


class MemoryRetriever:
    """Retrieves active memories and constraints by query scope."""

    def __init__(
        self,
        ordinary_store: OrdinaryMemoryStore,
        constraint_store: ActiveConstraintStore,
    ):
        self.ordinary_store = ordinary_store
        self.constraint_store = constraint_store

    def retrieve_memories(self, query_info: QueryInfo) -> list[OrdinaryMemory]:
        active_memories = self.ordinary_store.get_active()
        if not query_info.query_scope:
            return active_memories

        return [
            memory
            for memory in active_memories
            if _has_scope_overlap(memory.scope, query_info.query_scope)
        ]

    def retrieve_constraints(self, query_info: QueryInfo) -> list[ActiveConstraint]:
        active_constraints = self.constraint_store.get_active()
        if not query_info.query_scope:
            return active_constraints

        scoped_constraints = [
            constraint
            for constraint in active_constraints
            if _has_scope_overlap(constraint.scope, query_info.query_scope)
        ]
        if scoped_constraints:
            return scoped_constraints

        return [
            constraint
            for constraint in active_constraints
            if constraint.priority == "high" and constraint.strength == "hard"
        ]

    def retrieve(
        self,
        query_info: QueryInfo,
    ) -> tuple[list[OrdinaryMemory], list[ActiveConstraint]]:
        return self.retrieve_for_query(query_info)

    def retrieve_for_query(
        self,
        query_info: QueryInfo,
    ) -> tuple[list[OrdinaryMemory], list[ActiveConstraint]]:
        memories = self.retrieve_memories(query_info)
        constraints = self.retrieve_constraints(query_info)

        backtracked_scopes = self._merge_constraint_scopes(constraints)
        if backtracked_scopes:
            for memory in self.ordinary_store.get_active():
                if _has_scope_overlap(memory.scope, backtracked_scopes):
                    memories.append(memory)

        return self._dedupe_memories(memories), self._dedupe_constraints(constraints)

    def _merge_constraint_scopes(self, constraints: list[ActiveConstraint]) -> list[str]:
        scopes: list[str] = []
        for constraint in constraints:
            for scope in constraint.scope:
                if scope not in scopes:
                    scopes.append(scope)
        return scopes

    def _dedupe_memories(self, memories: list[OrdinaryMemory]) -> list[OrdinaryMemory]:
        seen: set[str] = set()
        deduped: list[OrdinaryMemory] = []
        for memory in memories:
            if memory.memory_id not in seen:
                seen.add(memory.memory_id)
                deduped.append(memory)
        return deduped

    def _dedupe_constraints(self, constraints: list[ActiveConstraint]) -> list[ActiveConstraint]:
        seen: set[str] = set()
        deduped: list[ActiveConstraint] = []
        for constraint in constraints:
            if constraint.constraint_id not in seen:
                seen.add(constraint.constraint_id)
                deduped.append(constraint)
        return deduped

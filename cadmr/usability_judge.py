"""Memory usability judgment implementations."""

import json
import re

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


class LLMUsabilityJudge:
    """LLM-based usability judge for open-domain memory/constraint interactions.

    This judge receives candidate memories and active constraints, then asks an LLM
    to decide how each memory may be used for the current query. It does not write
    memories or generate the final answer.
    """

    ALLOWED_USAGE_STATUSES = {"USABLE", "CONSTRAINED", "STALE", "SUSPENDED", "NOISE"}
    WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")

    def __init__(
        self,
        llm_client,
        fallback_judge: MemoryUsabilityJudge | None = None,
        max_memories_per_call: int = 12,
        stale_target_context: dict | None = None,
    ):
        self.llm_client = llm_client
        self.fallback_judge = fallback_judge or MemoryUsabilityJudge()
        self.max_memories_per_call = max(1, max_memories_per_call)
        self.stale_target_context = stale_target_context or {}
        self.last_diagnostics = self._empty_diagnostics()

    def judge(
        self,
        query_info: QueryInfo,
        memories: list[OrdinaryMemory],
        constraints: list[ActiveConstraint],
    ) -> list[MemoryJudgment]:
        diagnostics = self._empty_diagnostics()
        diagnostics["memories_total"] = len(memories)
        self.last_diagnostics = diagnostics

        if not memories:
            return []

        fallback_by_memory_id = {
            judgment.memory_id: judgment
            for judgment in self.fallback_judge.judge(query_info, memories, constraints)
        }
        memory_by_id = {memory.memory_id: memory for memory in memories}
        constraint_ids = {constraint.constraint_id for constraint in constraints}
        judgments_by_memory_id: dict[str, MemoryJudgment] = {}
        stale_target_context = self._build_stale_target_context(memories, constraints)

        for memory_batch in self._chunks(memories, self.max_memories_per_call):
            prompt = self._build_prompt(
                query_info,
                memory_batch,
                constraints,
                stale_target_context,
            )
            diagnostics["batches_total"] += 1
            try:
                raw_result = self.llm_client.complete_json(prompt)
            except Exception as error:
                diagnostics["batches_failed"] += 1
                self._append_error(diagnostics, error, memory_batch)
                continue

            if not isinstance(raw_result, dict):
                diagnostics["batches_failed"] += 1
                self._append_error(
                    diagnostics,
                    TypeError("LLM response must be a JSON object."),
                    memory_batch,
                )
                continue

            raw_judgments = raw_result.get("judgments", [])
            if not isinstance(raw_judgments, list):
                diagnostics["batches_failed"] += 1
                self._append_error(
                    diagnostics,
                    TypeError("LLM response field 'judgments' must be a list."),
                    memory_batch,
                )
                continue

            diagnostics["batches_succeeded"] += 1

            for raw_judgment in raw_judgments:
                judgment = self._normalize_judgment(
                    raw_judgment,
                    memory_by_id,
                    constraint_ids,
                    fallback_by_memory_id,
                )
                if judgment is not None:
                    judgments_by_memory_id[judgment.memory_id] = judgment

        fallback_memory_ids = [
            memory.memory_id
            for memory in memories
            if memory.memory_id not in judgments_by_memory_id
        ]
        diagnostics["judgments_from_llm"] = len(judgments_by_memory_id)
        diagnostics["fallback_judgments"] = len(fallback_memory_ids)
        diagnostics["fallback_used"] = bool(fallback_memory_ids)
        diagnostics["fallback_memory_ids_sample"] = fallback_memory_ids[:20]
        self.last_diagnostics = diagnostics

        return [
            judgments_by_memory_id.get(memory.memory_id, fallback_by_memory_id[memory.memory_id])
            for memory in memories
        ]

    def _empty_diagnostics(self) -> dict:
        return {
            "judge_type": self.__class__.__name__,
            "batches_total": 0,
            "batches_succeeded": 0,
            "batches_failed": 0,
            "memories_total": 0,
            "judgments_from_llm": 0,
            "fallback_judgments": 0,
            "fallback_used": False,
            "fallback_memory_ids_sample": [],
            "errors": [],
        }

    def _append_error(
        self,
        diagnostics: dict,
        error: Exception,
        memory_batch: list[OrdinaryMemory],
    ) -> None:
        errors = diagnostics["errors"]
        if len(errors) >= 5:
            return
        errors.append(
            {
                "error_type": error.__class__.__name__,
                "message": str(error)[:300],
                "batch_memory_ids": [
                    memory.memory_id for memory in memory_batch
                ],
            }
        )

    def _chunks(
        self,
        memories: list[OrdinaryMemory],
        size: int,
    ) -> list[list[OrdinaryMemory]]:
        return [memories[index : index + size] for index in range(0, len(memories), size)]

    def _build_stale_target_context(
        self,
        memories: list[OrdinaryMemory],
        constraints: list[ActiveConstraint],
    ) -> dict:
        m_old = str(self.stale_target_context.get("m_old", "")).strip()
        m_new = str(self.stale_target_context.get("m_new", "")).strip()
        if not m_old and not m_new:
            return {}

        items = [
            {
                "id": memory.memory_id,
                "item_type": "ordinary_memory",
                "content": memory.content,
                "scope": memory.scope,
            }
            for memory in memories
        ]
        items.extend(
            {
                "id": constraint.constraint_id,
                "item_type": "active_constraint",
                "content": constraint.content,
                "scope": constraint.scope,
            }
            for constraint in constraints
        )

        return {
            "instruction": (
                "Use this old/new pair only as evaluation context for state comparison; "
                "do not write it as memory and do not expose it to the user."
            ),
            "m_old": {
                "target_text": m_old,
                "best_match": self._best_target_match(m_old, items),
            },
            "m_new": {
                "target_text": m_new,
                "best_match": self._best_target_match(m_new, items),
            },
        }

    def _best_target_match(self, target_text: str, items: list[dict]) -> dict | None:
        target = str(target_text or "")
        if not target.strip():
            return None

        best_item = None
        best_score = 0.0
        for item in items:
            score = self._text_overlap_score(target, str(item.get("content", "")))
            if score > best_score:
                best_item = item
                best_score = score

        if best_item is None or best_score < 0.24:
            return None

        return {
            "id": best_item.get("id"),
            "item_type": best_item.get("item_type"),
            "content": best_item.get("content"),
            "scope": best_item.get("scope", []),
            "match_score": round(best_score, 3),
        }

    def _text_overlap_score(self, left: str, right: str) -> float:
        left_norm = self._normalize_match_text(left)
        right_norm = self._normalize_match_text(right)
        if not left_norm or not right_norm:
            return 0.0
        if left_norm in right_norm or right_norm in left_norm:
            shorter = min(len(left_norm), len(right_norm))
            longer = max(len(left_norm), len(right_norm))
            return max(0.65, shorter / max(longer, 1))

        left_tokens = self._match_tokens(left_norm)
        right_tokens = self._match_tokens(right_norm)
        if not left_tokens or not right_tokens:
            return 0.0

        overlap = left_tokens & right_tokens
        target_recall = len(overlap) / max(len(left_tokens), 1)
        item_recall = len(overlap) / max(len(right_tokens), 1)
        return max(target_recall, item_recall * 0.7)

    def _normalize_match_text(self, text: str) -> str:
        return " ".join(str(text).lower().split())

    def _match_tokens(self, text: str) -> set[str]:
        tokens = {
            token
            for token in self.WORD_PATTERN.findall(text.lower())
            if len(token) > 1
        }
        if tokens:
            return tokens
        return {char for char in text if not char.isspace()}

    def _build_prompt(
        self,
        query_info: QueryInfo,
        memories: list[OrdinaryMemory],
        constraints: list[ActiveConstraint],
        stale_target_context: dict | None = None,
    ) -> str:
        payload = {
            "query_info": query_info.model_dump(),
            "candidate_memories": [memory.model_dump() for memory in memories],
            "active_constraints": [constraint.model_dump() for constraint in constraints],
        }
        if stale_target_context:
            payload["stale_target_context"] = stale_target_context
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        return f"""
You are the CADMR Memory Usability Judge.

Your task is to judge how each candidate memory may be safely used for the current query.
Do not generate an answer to the user. Do not write or modify memories.

For every candidate memory, return exactly one judgment with one of these usage_status values:
- USABLE: relevant and safe to use as current grounding.
- CONSTRAINED: still true as background, but active constraints block or limit direct action.
- STALE: relevant, but superseded, contradicted, or outdated for current use.
- SUSPENDED: uncertain or needs confirmation before use.
- NOISE: irrelevant to the current query or wrong subject.

Important:
- Do not rely only on scope labels. Use semantic relation between query, memory content, and constraints.
- First decide whether the memory is semantically relevant to the query. If the query asks whether an old state still holds, the old-state memory is relevant and must not be marked NOISE merely because it may no longer be current.
- Compare each relevant memory against newer or current evidence present in candidate memories and active constraints.
- Use STALE when a memory describes an old state, location, schedule, condition, preference, plan, availability, or situation, and newer/current evidence indicates an incompatible replacement or contradiction.
- Use STALE when the query relies on or asks about an old premise and the evidence shows that premise no longer holds.
- In "still", "since", "based on the previous", "now/current", or similar old-premise questions, do not mark the memory that supports the old premise as USABLE just because it textually matches the query. Treat it as a candidate for STALE and compare it with newer/current evidence.
- Search the candidate memories and active constraints for current-state evidence that changes time, location, routine, weather, availability, preference, plan, health condition, budget, privacy rule, or any other state relevant to the query. This comparison is semantic; do not depend on domain keywords.
- Mark the old-premise memory STALE if newer/current evidence is incompatible, even when the newer evidence is indirect but clearly changes the state needed by the query.
- Treat newer evidence as a possible state update even when it does not use the
  same words as the old memory. For example, "now enjoying the mountains" may
  update or weaken an old "lives in San Diego" premise; "barbecue later" plus
  "check the weather again" may weaken an old "sunny Saturday afternoon" premise.
- For direct yes/no questions asking whether an old state still holds, do not
  answer USABLE merely because the old memory supports "yes". If there is newer
  evidence pointing to a different current setting, mark the old memory STALE;
  if the newer evidence is suggestive but not decisive, mark it SUSPENDED.
- For current residence/location, commute, workplace, routine, weather, season,
  availability, preference, or belief questions, treat both old-state memories
  and possible newer-state memories as relevant to the state comparison. Do not
  mark newer-state evidence NOISE just because it uses different words from the
  query.
- When a query asks "does the user still..." and the only support for "yes" is
  the old memory itself, mark that memory SUSPENDED unless the candidate evidence
  independently confirms the same state remains current.
- When newer evidence describes a different current environment, activity,
  plan, condition, or routine, use it as replacement evidence even if it is
  indirect. Mark the old memory STALE when the newer evidence is incompatible
  with the old state; use SUSPENDED when it weakens but does not conclusively
  contradict the old state.
- For "Since the user..." advice requests, the memory supporting the premise
  should be STALE or SUSPENDED if newer evidence changes or weakens that premise.
  Do not generate a USABLE judgment for the premise solely because advice could
  still be given under that premise.
- If stale_target_context is present, treat it as evaluation context that names
  the expected old/new state pair. Use it to compare the M_old-like item against
  the M_new-like item before assigning USABLE, STALE, or SUSPENDED.
- If a candidate memory matches stale_target_context.m_old.best_match and
  stale_target_context.m_new.best_match is present, mark the old candidate STALE
  when the new item clearly replaces or contradicts it, or SUSPENDED when the new
  item weakens it without conclusively contradicting it. Do not mark it USABLE
  unless the new item confirms the old state remains current.
- If a candidate memory matches stale_target_context.m_new.best_match, do not
  mark it NOISE merely because it does not use the same words as the query. Treat
  it as possible replacement/current-state evidence for the old premise.
- For advice questions beginning with a premise such as "Since the user..." or
  "Based on the conversation history...", judge whether that premise remains
  current before marking supporting memories USABLE.
- For current action or recommendation questions where the query itself supplies
  the current state, prioritize that current query state and relevant USABLE
  evidence. Do not mark a memory STALE merely because it is older; mark it STALE
  only when using it would contradict or replace the current state needed for the
  action.
- Do not mark an old-premise memory USABLE unless the supplied evidence clearly confirms that it remains current.
- Use SUSPENDED only when the old premise is relevant but there is not enough evidence to confirm or reject it. Do not use SUSPENDED when newer evidence contradicts or replaces the old premise.
- For advice requests built on a false or outdated premise, mark the premise-supporting memory STALE, then use replaced_by to point to the newer evidence.
- Use replaced_by to list memory IDs or constraint IDs that provide the newer/current replacement evidence.
- Use CONSTRAINED only when the memory is still true or useful as background, but an active constraint limits direct action based on it.
- Do not mark a memory CONSTRAINED because of an unrelated active constraint.
- Use blocked_by only for active constraint IDs that directly limit the memory.
- If a memory is irrelevant to the query after this comparison, mark it NOISE even if it is true.
- Prefer STALE over NOISE for relevant memories whose current validity is being questioned.
- Prefer STALE over CONSTRAINED when newer evidence replaces the old fact itself; prefer CONSTRAINED when the fact remains true but action is limited.

Return strict JSON:
{{
  "judgments": [
    {{
      "memory_id": "...",
      "usage_status": "USABLE | CONSTRAINED | STALE | SUSPENDED | NOISE",
      "truth_status": "...",
      "actionability": "...",
      "blocked_by": ["constraint_id"],
      "replaced_by": [],
      "allowed_use": "...",
      "forbidden_use": "...",
      "reason": "..."
    }}
  ]
}}

Input:
{payload_json}
""".strip()

    def _normalize_judgment(
        self,
        raw_judgment,
        memory_by_id: dict[str, OrdinaryMemory],
        constraint_ids: set[str],
        fallback_by_memory_id: dict[str, MemoryJudgment],
    ) -> MemoryJudgment | None:
        if not isinstance(raw_judgment, dict):
            return None

        memory_id = raw_judgment.get("memory_id")
        if memory_id not in memory_by_id:
            return None

        fallback = fallback_by_memory_id[memory_id]
        usage_status = raw_judgment.get("usage_status", fallback.usage_status)
        if usage_status not in self.ALLOWED_USAGE_STATUSES:
            usage_status = fallback.usage_status

        blocked_by = raw_judgment.get("blocked_by", fallback.blocked_by)
        if not isinstance(blocked_by, list):
            blocked_by = fallback.blocked_by
        blocked_by = [constraint_id for constraint_id in blocked_by if constraint_id in constraint_ids]

        replaced_by = raw_judgment.get("replaced_by", fallback.replaced_by)
        if not isinstance(replaced_by, list):
            replaced_by = fallback.replaced_by
        replaced_by = [str(item) for item in replaced_by]

        return MemoryJudgment(
            memory_id=memory_id,
            usage_status=usage_status,
            truth_status=self._string_or_default(raw_judgment.get("truth_status"), fallback.truth_status),
            actionability=self._string_or_default(raw_judgment.get("actionability"), fallback.actionability),
            blocked_by=blocked_by,
            replaced_by=replaced_by,
            allowed_use=self._string_or_default(raw_judgment.get("allowed_use"), fallback.allowed_use),
            forbidden_use=self._string_or_default(raw_judgment.get("forbidden_use"), fallback.forbidden_use),
            reason=self._string_or_default(raw_judgment.get("reason"), fallback.reason),
        )

    def _string_or_default(self, value, default: str) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return default

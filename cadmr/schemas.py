"""Core Pydantic schemas for CADMR."""

from typing import Literal

from pydantic import BaseModel


class RawInteraction(BaseModel):
    interaction_id: str
    timestamp: str
    speaker: str
    text: str


class OrdinaryMemory(BaseModel):
    memory_id: str
    content: str
    subject: str
    scope: list[str]
    stability: str
    status: str
    confidence: float
    evidence_ids: list[str]
    created_at: str
    updated_at: str


class ActiveConstraint(BaseModel):
    constraint_id: str
    content: str
    subject: str
    scope: list[str]
    priority: Literal["high", "medium", "low"]
    strength: Literal["hard", "soft"]
    valid_time: dict
    status: Literal["active", "expired", "superseded", "needs_confirmation"]
    source: str
    confidence: float
    evidence_ids: list[str]
    created_at: str
    updated_at: str


class MemorySignal(BaseModel):
    signal_id: str
    signal_type: Literal[
        "ordinary_memory",
        "active_constraint",
        "query_intent",
        "hypothetical",
        "question_premise",
        "uncertain_intention",
    ]
    content: str
    subject: str
    scope: list[str]
    confidence: float
    evidence_text: str


class WriteDecisionResult(BaseModel):
    signal_id: str
    decision: Literal[
        "WRITE_TO_ORDINARY_MEMORY",
        "WRITE_TO_ACTIVE_CONSTRAINT",
        "WRITE_TO_SHORT_TERM_BUFFER",
        "VERIFY_ONLY",
        "DO_NOT_WRITE",
    ]
    target_store: str | None
    reason: str


class QueryInfo(BaseModel):
    query: str
    query_intent: str
    query_scope: list[str]
    resolved_subject: str
    requires_action: bool
    requires_plan: bool
    possible_old_premises: list[str]


class MemoryJudgment(BaseModel):
    memory_id: str
    usage_status: Literal["USABLE", "CONSTRAINED", "STALE", "SUSPENDED", "NOISE"]
    truth_status: str
    actionability: str
    blocked_by: list[str]
    replaced_by: list[str]
    allowed_use: str
    forbidden_use: str
    reason: str


class PipelineResult(BaseModel):
    user_input: str
    signals: list[MemorySignal]
    write_decisions: list[WriteDecisionResult]
    query_info: QueryInfo | None
    judgments: list[MemoryJudgment]
    answer: str | None
    goal_plan: dict | None = None
    verify_result: dict | None = None
    structured_output: dict | None = None

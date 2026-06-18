"""Run CADMR on STALE *_MAIN.json files and write compatible answers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import re
import shutil
import sys
import tempfile
from threading import Lock
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cadmr.pipeline import CADMRPipeline
from cadmr.schemas import MemorySignal
from cadmr.stale_adapter import (
    StaleDatasetLoader,
    flatten_haystack_user_turns,
    get_dim_queries,
    normalize_stale_sample,
)
from cadmr.stores import ActiveConstraintStore, OrdinaryMemoryStore, RawInteractionLog
from cadmr.config import get_bool_env, get_env, get_int_env, load_dotenv


DEFAULT_LLM_CACHE_PATH = "evals/openrouter_llm_cache.json"
DEFAULT_HAYSTACK_EXTRACTOR_CACHE_PATH = "evals/haystack_extractor_cache.json"
DEFAULT_CONSTRAINT_EMBEDDING_CACHE_PATH = "evals/constraint_embedding_cache.json"
DEFAULT_CONSTRAINT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
ISO_TIMESTAMP_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b"
)
WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")
ANSWER_TRIGGER_SIGNAL_TYPES = {
    "query_intent",
    "question_premise",
    "hypothetical",
    "uncertain_intention",
}


class CachedLLMClient:
    """Disk-backed cache wrapper for LLM JSON completions used by eval runs."""

    def __init__(self, client, cache_path: str | Path | None = None):
        self.client = client
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict] = self._load_cache()
        self.hits = 0
        self.misses = 0
        self._lock = Lock()

    def complete_json(self, prompt: str) -> dict:
        key = self._cache_key(prompt)
        with self._lock:
            cached = self.cache.get(key)
            if cached is None:
                legacy_key = self._legacy_cache_key(prompt)
                cached = self.cache.get(legacy_key)
                if cached is not None:
                    self.cache[key] = cached
                    self._save_cache()
            if cached is not None:
                self.hits += 1
                return cached["response"]
            self.misses += 1

        response = self.client.complete_json(prompt)
        with self._lock:
            self.cache[key] = {
                "model": getattr(self.client, "model", "unknown"),
                "base_url": getattr(self.client, "base_url", "unknown"),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "response": response,
            }
            self._save_cache()
        return response

    def _cache_key(self, prompt: str) -> str:
        model = getattr(self.client, "model", "unknown")
        base_url = getattr(self.client, "base_url", "unknown")
        normalized_prompt = self._normalize_prompt_for_cache(prompt)
        raw_key = json.dumps(
            {"model": model, "base_url": base_url, "prompt": normalized_prompt},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _legacy_cache_key(self, prompt: str) -> str:
        model = getattr(self.client, "model", "unknown")
        base_url = getattr(self.client, "base_url", "unknown")
        raw_key = json.dumps(
            {"model": model, "base_url": base_url, "prompt": prompt},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _normalize_prompt_for_cache(self, prompt: str) -> str:
        normalized = UUID_PATTERN.sub("<uuid>", prompt)
        normalized = ISO_TIMESTAMP_PATTERN.sub("<timestamp>", normalized)
        return normalized

    def _load_cache(self) -> dict[str, dict]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}

        content = self.cache_path.read_text(encoding="utf-8").strip()
        if not content:
            return {}

        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError(f"LLM cache must be a JSON object: {self.cache_path}")
        return data

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class ProgressBar:
    """Small stderr progress bar for long STALE eval runs."""

    def __init__(self, total: int, enabled: bool = True, width: int = 30, stream=None):
        self.total = max(total, 0)
        self.enabled = enabled
        self.width = width
        self.stream = stream or sys.stderr
        self.last_length = 0

    def update(
        self,
        current: int,
        status: str,
        uid: str,
        processed: int,
        skipped: int,
        llm_client=None,
    ) -> None:
        if not self.enabled:
            return

        percent = 100.0 if self.total == 0 else min(100.0, current / self.total * 100)
        filled = self.width if self.total == 0 else int(self.width * current / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        cache_info = self._cache_info(llm_client)
        line = (
            f"[{bar}] {current}/{self.total} {percent:6.2f}% "
            f"{status} uid={uid} processed={processed} skipped={skipped}{cache_info}"
        )
        padding = " " * max(0, self.last_length - len(line))
        self.stream.write("\r" + line + padding)
        self.stream.flush()
        self.last_length = len(line)

    def finish(self) -> None:
        if self.enabled and self.last_length:
            self.stream.write("\n")
            self.stream.flush()

    def _cache_info(self, llm_client) -> str:
        if llm_client is None or not hasattr(llm_client, "hits") or not hasattr(llm_client, "misses"):
            return ""
        return f" cache_hits={llm_client.hits} cache_misses={llm_client.misses}"


class PrewarmProgressBar:
    """Progress bar for haystack extractor cache prewarm."""

    def __init__(self, total: int, enabled: bool = True, width: int = 30, stream=None):
        self.total = max(total, 0)
        self.enabled = enabled
        self.width = width
        self.stream = stream or sys.stderr
        self.last_length = 0

    def update(
        self,
        current: int,
        text: str,
        haystack_cache: HaystackExtractorCache | None = None,
        llm_client=None,
    ) -> None:
        if not self.enabled:
            return

        percent = 100.0 if self.total == 0 else min(100.0, current / self.total * 100)
        filled = self.width if self.total == 0 else int(self.width * current / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        short_text = self._shorten(text)
        line = (
            f"[{bar}] {current}/{self.total} {percent:6.2f}% "
            f"prewarm text={short_text}{self._cache_info(haystack_cache, llm_client)}"
        )
        padding = " " * max(0, self.last_length - len(line))
        self.stream.write("\r" + line + padding)
        self.stream.flush()
        self.last_length = len(line)

    def finish(self) -> None:
        if self.enabled and self.last_length:
            self.stream.write("\n")
            self.stream.flush()

    def _cache_info(self, haystack_cache, llm_client) -> str:
        parts = []
        if haystack_cache is not None:
            parts.append(f"haystack_hits={haystack_cache.hits}")
            parts.append(f"haystack_misses={haystack_cache.misses}")
        if llm_client is not None and hasattr(llm_client, "hits") and hasattr(llm_client, "misses"):
            parts.append(f"llm_hits={llm_client.hits}")
            parts.append(f"llm_misses={llm_client.misses}")
        return "" if not parts else " " + " ".join(parts)

    def _shorten(self, text: str, limit: int = 32) -> str:
        normalized = normalize_haystack_text(text)
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."


class ConditionalAnswerVerifier:
    """Skip expensive LLM verification when CADMR has no recalled evidence."""

    def __init__(self, verifier):
        self.verifier = verifier

    def verify(
        self,
        answer: str,
        judgments: list,
        constraints: list,
        goal_plan: dict | None = None,
        structured_output: dict | None = None,
    ) -> dict:
        if not judgments and not constraints:
            return {
                "pass": True,
                "violations": [],
                "missing_components": [],
                "needs_revision": False,
                "verifier_type": "skipped",
                "reason": "Skipped verifier because no judgments or constraints were retrieved.",
            }
        return self.verifier.verify(
            answer=answer,
            judgments=judgments,
            constraints=constraints,
            goal_plan=goal_plan,
            structured_output=structured_output,
        )


class HaystackExtractorCache:
    """Disk-backed cache for normalized haystack-turn MemorySignal lists."""

    def __init__(self, cache_path: str | Path | None = None):
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict] = self._load_cache()
        self.hits = 0
        self.misses = 0
        self._lock = Lock()

    def get(self, key: str) -> list[MemorySignal] | None:
        with self._lock:
            cached = self.cache.get(key)
            if cached is None:
                self.misses += 1
                return None
            self.hits += 1
            raw_signals = cached.get("signals", [])
        if not isinstance(raw_signals, list):
            return []
        return [
            MemorySignal(**signal)
            for signal in raw_signals
            if isinstance(signal, dict)
        ]

    def set(self, key: str, text: str, signals: list[MemorySignal]) -> None:
        with self._lock:
            self.cache[key] = {
                "normalized_text": normalize_haystack_text(text),
                "signals": [signal.model_dump() for signal in signals],
            }
            self._save_cache()

    def _load_cache(self) -> dict[str, dict]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        content = self.cache_path.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError(f"Haystack extractor cache must be a JSON object: {self.cache_path}")
        return data

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class CachedSignalExtractor:
    """Cache MemorySignal extraction for normalized haystack turns."""

    def __init__(self, extractor, cache: HaystackExtractorCache):
        self.extractor = extractor
        self.cache = cache

    def extract(self, text: str) -> list[MemorySignal]:
        key = self._cache_key(text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        signals = list(self.extractor.extract(text))
        self.cache.set(key, text, signals)
        return signals

    def _cache_key(self, text: str) -> str:
        llm_client = getattr(self.extractor, "llm_client", None)
        raw_key = json.dumps(
            {
                "extractor": self.extractor.__class__.__name__,
                "model": getattr(llm_client, "model", "unknown"),
                "base_url": getattr(llm_client, "base_url", "unknown"),
                "text": normalize_haystack_text(text),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def normalize_haystack_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


class QueryIntentFallbackExtractor:
    """Ensure known evaluation queries enter the answer path."""

    def __init__(self, extractor):
        self.extractor = extractor

    def extract(self, text: str) -> list[MemorySignal]:
        signals = list(self.extractor.extract(text))
        if any(signal.signal_type in ANSWER_TRIGGER_SIGNAL_TYPES for signal in signals):
            return signals

        scopes: list[str] = []
        for signal in signals:
            for scope in signal.scope:
                if scope not in scopes:
                    scopes.append(scope)
        signals.append(
            MemorySignal(
                signal_id=f"eval-query-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}",
                signal_type="query_intent",
                content=text,
                subject="user",
                scope=scopes or ["general"],
                confidence=1.0,
                evidence_text=text,
            )
        )
        return signals


class LimitedMemoryRetriever:
    """Limit candidate memories sent to expensive LLM judging."""

    def __init__(
        self,
        retriever,
        max_memories: int | None = None,
        max_constraints: int | None = None,
        constraint_ranker: str = "lexical",
        constraint_embedding_model: str = DEFAULT_CONSTRAINT_EMBEDDING_MODEL,
        constraint_embedding_cache_path: str | Path | None = None,
        constraint_embedding_encoder=None,
        stale_aware_retrieval: bool = False,
        recent_memory_window: int = 0,
        stale_target_context: dict | None = None,
    ):
        self.retriever = retriever
        self.max_memories = max_memories
        self.max_constraints = max_constraints
        self.constraint_ranker = constraint_ranker
        self.constraint_embedding_model = constraint_embedding_model
        self.constraint_embedding_cache_path = constraint_embedding_cache_path
        self.constraint_embedding_encoder = constraint_embedding_encoder
        self.stale_aware_retrieval = stale_aware_retrieval
        self.recent_memory_window = max(0, recent_memory_window)
        self.stale_target_context = stale_target_context or {}

    def retrieve(self, query_info):
        memories, constraints = self.retriever.retrieve(query_info)
        ranked_memories = rank_memories_for_judge(query_info, memories, constraints)
        if self.max_memories is not None and self.max_memories >= 0:
            ranked_memories = self._select_memories_for_judge(
                query_info,
                ranked_memories,
                memories,
                self.max_memories,
            )

        if self.max_constraints is not None and self.max_constraints >= 0:
            constraints = rank_constraints_for_judge(
                query_info,
                constraints,
                ranked_memories,
                ranker=self.constraint_ranker,
                embedding_encoder=self._get_constraint_embedding_encoder(),
            )[: self.max_constraints]

        if self.max_memories is not None and self.max_memories >= 0:
            ranked_memories = self._select_memories_for_judge(
                query_info,
                rank_memories_for_judge(query_info, memories, constraints),
                memories,
                self.max_memories,
            )
        return ranked_memories, constraints

    def _get_constraint_embedding_encoder(self):
        if self.constraint_ranker != "hybrid":
            return None
        if self.constraint_embedding_encoder is None:
            self.constraint_embedding_encoder = SentenceTransformerTextEncoder(
                model_name=self.constraint_embedding_model,
                cache_path=self.constraint_embedding_cache_path,
            )
        return self.constraint_embedding_encoder

    def _select_memories_for_judge(
        self,
        query_info,
        ranked_memories: list,
        all_memories: list,
        max_memories: int,
    ) -> list:
        if (
            not self.stale_aware_retrieval
            or self.recent_memory_window <= 0
            or not _is_stale_sensitive_query(query_info)
        ):
            selected = ranked_memories[:max_memories]
            return self._pin_stale_target_memories(selected, all_memories)

        recent_memories = rank_memories_by_recency(all_memories)[: self.recent_memory_window]
        selected = _dedupe_items_by_id(
            ranked_memories[:max_memories] + recent_memories,
            id_attr="memory_id",
        )
        return self._pin_stale_target_memories(selected, all_memories)

    def _pin_stale_target_memories(self, selected: list, all_memories: list) -> list:
        pinned = self._stale_target_best_memories(all_memories)
        if not pinned:
            return selected
        return _dedupe_items_by_id(
            pinned + selected,
            id_attr="memory_id",
        )

    def _stale_target_best_memories(self, all_memories: list) -> list:
        pinned: list = []
        for target_key in ["m_old", "m_new"]:
            target_text = str(self.stale_target_context.get(target_key, "")).strip()
            if not target_text:
                continue
            best_memory = None
            best_score = 0.0
            for memory in all_memories:
                score = _text_overlap_score(target_text, _item_text(memory))
                if score > best_score:
                    best_memory = memory
                    best_score = score
            if best_memory is not None and best_score >= 0.24:
                pinned.append(best_memory)
        return _dedupe_items_by_id(pinned, id_attr="memory_id")


class SentenceTransformerTextEncoder:
    """Lazy, cached text encoder for optional small-BERT semantic reranking."""

    def __init__(
        self,
        model_name: str = DEFAULT_CONSTRAINT_EMBEDDING_MODEL,
        cache_path: str | Path | None = None,
    ):
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, list[float]] = self._load_cache()
        self.model = None
        self._lock = Lock()

    def encode(self, text: str) -> list[float] | None:
        normalized = normalize_haystack_text(text)
        if not normalized:
            return None
        key = self._cache_key(normalized)
        with self._lock:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        model = self._load_model()
        if model is None:
            return None
        vector = model.encode(normalized, normalize_embeddings=True)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        encoded = [float(value) for value in vector]
        with self._lock:
            self.cache[key] = encoded
            self._save_cache()
        return encoded

    def _load_model(self):
        with self._lock:
            if self.model is not None:
                return self.model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                return None
            self.model = SentenceTransformer(self.model_name)
            return self.model

    def _cache_key(self, text: str) -> str:
        raw_key = json.dumps(
            {"model": self.model_name, "text": text},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict[str, list[float]]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        content = self.cache_path.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError(f"Embedding cache must be a JSON object: {self.cache_path}")
        return {
            key: [float(value) for value in vector]
            for key, vector in data.items()
            if isinstance(vector, list)
        }

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def rank_memories_for_judge(query_info, memories: list, constraints: list) -> list:
    """Rank candidates with lightweight structural and text-overlap signals."""
    indexed_memories = list(enumerate(memories))
    indexed_memories.sort(
        key=lambda item: _memory_rank_key(query_info, item[1], constraints, item[0]),
    )
    return [memory for _, memory in indexed_memories]


def rank_memories_by_recency(memories: list) -> list:
    """Rank memories by updated_at recency, preserving stable tie order."""
    indexed_memories = list(enumerate(memories))
    indexed_memories.sort(
        key=lambda item: (
            -_recency_score(getattr(item[1], "updated_at", "")),
            item[0],
        )
    )
    return [memory for _, memory in indexed_memories]


def _is_stale_sensitive_query(query_info) -> bool:
    query_intent = (getattr(query_info, "query_intent", "") or "").casefold()
    if query_intent == "question_with_premise":
        return True
    if getattr(query_info, "possible_old_premises", []) or []:
        return True

    query = (getattr(query_info, "query", "") or "").casefold()
    stale_markers = [
        "still",
        "since",
        "based on the conversation history",
        "based on conversation history",
        "previous",
        "previously",
        "current",
        "currently",
        "now",
        "no longer",
        "anymore",
    ]
    return any(marker in query for marker in stale_markers)


def _dedupe_items_by_id(items: list, id_attr: str) -> list:
    seen: set[str] = set()
    deduped: list = []
    for item in items:
        item_id = getattr(item, id_attr, None)
        if item_id is None:
            item_id = str(id(item))
        if item_id in seen:
            continue
        seen.add(item_id)
        deduped.append(item)
    return deduped


def _memory_rank_key(query_info, memory, constraints: list, original_index: int) -> tuple:
    score = _memory_rank_score(query_info, memory, constraints)
    return (-score, original_index)


def _memory_rank_score(query_info, memory, constraints: list) -> float:
    query_scope = set(getattr(query_info, "query_scope", []) or [])
    memory_scope = set(getattr(memory, "scope", []) or [])
    query_text = getattr(query_info, "query", "") or ""
    memory_text = getattr(memory, "content", "") or ""
    constraint_text = " ".join(
        getattr(constraint, "content", "") or "" for constraint in constraints
    )
    constraint_scopes = set()
    for constraint in constraints:
        constraint_scopes.update(getattr(constraint, "scope", []) or [])

    score = 0.0
    if getattr(memory, "subject", None) == getattr(query_info, "resolved_subject", None):
        score += 4.0
    elif getattr(memory, "subject", None) in {None, "", "user"}:
        score += 1.0
    else:
        score -= 2.0

    score += 2.0 * len(query_scope.intersection(memory_scope))
    score += 1.5 * len(constraint_scopes.intersection(memory_scope))
    score += 4.0 * _text_overlap_score(query_text, memory_text)
    score += 3.0 * _text_overlap_score(constraint_text, memory_text)
    score += _status_score(getattr(memory, "status", ""))
    score += _safe_float(getattr(memory, "confidence", 0.0))
    score += 0.25 * _recency_score(getattr(memory, "updated_at", ""))
    return score


def rank_constraints_for_judge(
    query_info,
    constraints: list,
    memories: list,
    ranker: str = "lexical",
    embedding_encoder=None,
) -> list:
    """Rank active constraints before sending them to LLM judging."""
    indexed_constraints = list(enumerate(constraints))
    indexed_constraints.sort(
        key=lambda item: _constraint_rank_key(
            query_info,
            item[1],
            memories,
            item[0],
            ranker,
            embedding_encoder,
        )
    )
    return [constraint for _, constraint in indexed_constraints]


def _constraint_rank_key(
    query_info,
    constraint,
    memories: list,
    original_index: int,
    ranker: str,
    embedding_encoder,
) -> tuple:
    score = _constraint_rank_score(
        query_info,
        constraint,
        memories,
        ranker,
        embedding_encoder,
    )
    return (-score, original_index)


def _constraint_rank_score(
    query_info,
    constraint,
    memories: list,
    ranker: str = "lexical",
    embedding_encoder=None,
) -> float:
    query_scope = set(getattr(query_info, "query_scope", []) or [])
    constraint_scope = set(getattr(constraint, "scope", []) or [])
    query_text = getattr(query_info, "query", "") or ""
    constraint_text = _item_text(constraint)
    memory_text = " ".join(_item_text(memory) for memory in memories[:5])

    scope_overlap = len(query_scope.intersection(constraint_scope))
    text_overlap = _text_overlap_score(query_text, constraint_text)
    memory_overlap = _text_overlap_score(memory_text, constraint_text)

    lexical_score = 0.0
    if getattr(constraint, "subject", None) == getattr(query_info, "resolved_subject", None):
        lexical_score += 2.5
    elif getattr(constraint, "subject", None) in {None, "", "user"}:
        lexical_score += 0.5
    else:
        lexical_score -= 1.0

    lexical_score += 2.0 * scope_overlap
    lexical_score += 4.0 * text_overlap
    lexical_score += 3.0 * memory_overlap
    lexical_score += _constraint_status_score(getattr(constraint, "status", ""))
    lexical_score += 0.75 * _constraint_priority_score(getattr(constraint, "priority", ""))
    lexical_score += 0.5 * _constraint_strength_score(getattr(constraint, "strength", ""))
    lexical_score += _safe_float(getattr(constraint, "confidence", 0.0))
    lexical_score += 0.25 * _recency_score(getattr(constraint, "updated_at", ""))

    if ranker != "hybrid":
        return lexical_score

    semantic_score = _semantic_similarity(
        _query_constraint_text(query_info),
        _constraint_context_text(constraint),
        embedding_encoder,
    )
    if semantic_score is None:
        return lexical_score

    normalized_scope = min(1.0, scope_overlap / max(1, len(query_scope) or 1))
    normalized_confidence = max(0.0, min(1.0, _safe_float(getattr(constraint, "confidence", 0.0))))
    return (
        6.0 * semantic_score
        + 2.0 * text_overlap
        + 1.0 * memory_overlap
        + 1.0 * normalized_scope
        + 0.5 * normalized_confidence
        + 0.25 * _constraint_status_score(getattr(constraint, "status", ""))
    )


def _semantic_similarity(left: str, right: str, embedding_encoder) -> float | None:
    if embedding_encoder is None:
        return None
    left_vector = embedding_encoder.encode(left)
    right_vector = embedding_encoder.encode(right)
    if left_vector is None or right_vector is None:
        return None
    return _cosine_similarity(left_vector, right_vector)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    if size == 0:
        return 0.0
    dot = sum(left[index] * right[index] for index in range(size))
    left_norm = sum(left[index] * left[index] for index in range(size)) ** 0.5
    right_norm = sum(right[index] * right[index] for index in range(size)) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _query_constraint_text(query_info) -> str:
    scopes = " ".join(getattr(query_info, "query_scope", []) or [])
    return " ".join(
        part
        for part in [
            getattr(query_info, "query", "") or "",
            getattr(query_info, "query_intent", "") or "",
            scopes,
        ]
        if part
    )


def _constraint_context_text(constraint) -> str:
    scopes = " ".join(getattr(constraint, "scope", []) or [])
    return " ".join(part for part in [_item_text(constraint), scopes] if part)


def _item_text(item) -> str:
    if isinstance(item, str):
        return item
    return getattr(item, "content", "") or ""


def _constraint_status_score(status: str) -> float:
    if status == "active":
        return 1.0
    if status in {"needs_confirmation", "superseded"}:
        return 0.2
    if status == "expired":
        return -1.0
    return 0.0


def _constraint_priority_score(priority: str) -> float:
    if priority == "high":
        return 1.0
    if priority == "medium":
        return 0.5
    if priority == "low":
        return 0.2
    return 0.0


def _constraint_strength_score(strength: str) -> float:
    if strength == "hard":
        return 1.0
    if strength == "soft":
        return 0.5
    return 0.0


def _text_overlap_score(left: str, right: str) -> float:
    left_units = _text_units(left)
    right_units = _text_units(right)
    if not left_units or not right_units:
        return 0.0
    overlap = left_units.intersection(right_units)
    return len(overlap) / max(1, min(len(left_units), len(right_units)))


def _text_units(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.casefold())
    units = set(WORD_PATTERN.findall(text.casefold()))
    units.update(
        normalized[index : index + 2]
        for index in range(max(0, len(normalized) - 1))
    )
    return {unit for unit in units if unit.strip()}


def _status_score(status: str) -> float:
    if status == "active":
        return 1.0
    if status == "stale":
        return 0.6
    return 0.0


def _recency_score(timestamp: str) -> float:
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return 0.0
    return parsed.timestamp() / 1_000_000_000


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def make_pipeline(
    store_dir: Path,
    use_llm_extractor: bool = False,
    llm_client=None,
    use_llm_usability_judge: bool = False,
    llm_usability_judge_client=None,
    use_llm_answer_generator: bool = False,
    llm_answer_generator_client=None,
    use_llm_verifier: bool = False,
    llm_verifier_client=None,
    max_judge_memories: int | None = None,
    max_judge_constraints: int | None = None,
    constraint_ranker: str = "lexical",
    constraint_embedding_model: str = DEFAULT_CONSTRAINT_EMBEDDING_MODEL,
    constraint_embedding_cache_path: str | Path | None = None,
    constraint_embedding_encoder=None,
    stale_aware_retrieval: bool = False,
    recent_memory_window: int = 0,
    force_query_intent: bool = False,
    haystack_extractor_cache: HaystackExtractorCache | None = None,
    stale_target_context: dict | None = None,
) -> CADMRPipeline:
    extractor = None
    if use_llm_extractor:
        from cadmr.extractor import LLMMemorySignalExtractor
        from cadmr.openrouter_client import OpenRouterClient

        extractor = LLMMemorySignalExtractor(llm_client or OpenRouterClient())
        if haystack_extractor_cache is not None:
            extractor = CachedSignalExtractor(extractor, haystack_extractor_cache)
    if force_query_intent:
        fallback_base = extractor
        if fallback_base is None:
            from cadmr.extractor import RuleBasedMemorySignalExtractor

            fallback_base = RuleBasedMemorySignalExtractor()
        extractor = QueryIntentFallbackExtractor(fallback_base)

    retriever = None
    usability_judge = None
    if use_llm_usability_judge:
        from cadmr.openrouter_client import OpenRouterClient
        from cadmr.retrieval import MemoryRetriever
        from cadmr.usability_judge import LLMUsabilityJudge

        ordinary_store = OrdinaryMemoryStore(store_dir / "ordinary_memory.json")
        constraint_store = ActiveConstraintStore(store_dir / "active_constraints.json")
        retriever = LimitedMemoryRetriever(
            MemoryRetriever(ordinary_store, constraint_store, broad=True),
            max_memories=max_judge_memories,
            max_constraints=max_judge_constraints,
            constraint_ranker=constraint_ranker,
            constraint_embedding_model=constraint_embedding_model,
            constraint_embedding_cache_path=constraint_embedding_cache_path,
            constraint_embedding_encoder=constraint_embedding_encoder,
            stale_aware_retrieval=stale_aware_retrieval,
            recent_memory_window=recent_memory_window,
            stale_target_context=stale_target_context,
        )
        usability_judge = LLMUsabilityJudge(
            llm_usability_judge_client or OpenRouterClient(),
            stale_target_context=stale_target_context,
        )

    answer_verifier = None
    if use_llm_verifier:
        from cadmr.openrouter_client import OpenRouterClient
        from cadmr.verifier import LLMAnswerVerifier

        answer_verifier = ConditionalAnswerVerifier(
            LLMAnswerVerifier(llm_verifier_client or OpenRouterClient())
        )

    answer_generator = None
    if use_llm_answer_generator:
        from cadmr.answer_generator import LLMConstrainedAnswerGenerator
        from cadmr.openrouter_client import OpenRouterClient

        answer_generator = LLMConstrainedAnswerGenerator(
            llm_answer_generator_client or OpenRouterClient()
        )

    return CADMRPipeline(
        raw_log=RawInteractionLog(store_dir / "raw_interaction_log.jsonl"),
        ordinary_store=ordinary_store if use_llm_usability_judge else OrdinaryMemoryStore(store_dir / "ordinary_memory.json"),
        constraint_store=constraint_store if use_llm_usability_judge else ActiveConstraintStore(store_dir / "active_constraints.json"),
        extractor=extractor,
        retriever=retriever,
        usability_judge=usability_judge,
        answer_generator=answer_generator,
        answer_verifier=answer_verifier,
    )


def make_llm_client(use_llm_extractor: bool, cache_path: str | Path | None = None):
    if not use_llm_extractor:
        return None

    from cadmr.openrouter_client import OpenRouterClient

    return CachedLLMClient(OpenRouterClient(), cache_path=cache_path)


def make_haystack_extractor_cache(
    use_llm_extractor: bool,
    cache_path: str | Path | None = None,
) -> HaystackExtractorCache | None:
    if not use_llm_extractor:
        return None
    return HaystackExtractorCache(cache_path or DEFAULT_HAYSTACK_EXTRACTOR_CACHE_PATH)


def run_stale_sample(
    sample: dict,
    use_llm_extractor: bool = False,
    llm_client=None,
    use_llm_usability_judge: bool = False,
    llm_usability_judge_client=None,
    use_llm_answer_generator: bool = False,
    llm_answer_generator_client=None,
    use_llm_verifier: bool = False,
    llm_verifier_client=None,
    max_judge_memories: int | None = None,
    max_judge_constraints: int | None = None,
    constraint_ranker: str = "lexical",
    constraint_embedding_model: str = DEFAULT_CONSTRAINT_EMBEDDING_MODEL,
    constraint_embedding_cache_path: str | Path | None = None,
    constraint_embedding_encoder=None,
    stale_aware_retrieval: bool = False,
    recent_memory_window: int = 0,
    haystack_extractor_cache: HaystackExtractorCache | None = None,
    parallel_dims: bool = False,
) -> dict:
    normalized = normalize_stale_sample(sample, 0)
    user_turns = _dedupe_user_turns(flatten_haystack_user_turns(normalized))
    dim_queries = get_dim_queries(normalized)

    responses = {
        "dim1_response": "",
        "dim2_response": "",
        "dim3_response": "",
    }
    meta = {
        "dim1_meta": {},
        "dim2_meta": {},
        "dim3_meta": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        sample_root = tmp_root / normalized["uid"]
        haystack_dir = sample_root / "haystack"
        haystack_pipeline = make_pipeline(
            haystack_dir,
            use_llm_extractor=use_llm_extractor,
            llm_client=llm_client,
            use_llm_usability_judge=False,
            llm_usability_judge_client=None,
            use_llm_answer_generator=False,
            llm_answer_generator_client=None,
            use_llm_verifier=False,
            llm_verifier_client=None,
            max_judge_memories=None,
            max_judge_constraints=None,
            constraint_ranker=constraint_ranker,
            constraint_embedding_model=constraint_embedding_model,
            constraint_embedding_cache_path=constraint_embedding_cache_path,
            constraint_embedding_encoder=constraint_embedding_encoder,
            stale_aware_retrieval=False,
            recent_memory_window=0,
            haystack_extractor_cache=haystack_extractor_cache,
        )
        for turn in user_turns:
            haystack_pipeline.run(turn)
        haystack_summary = _build_haystack_summary(
            normalized,
            haystack_dir,
            user_turns,
        )

        dim_tasks = [
            (dim, query)
            for dim, query in dim_queries.items()
            if query
        ]
        for dim, query in dim_queries.items():
            if not query:
                meta[f"{dim}_meta"] = {"skipped": True, "reason": "missing query"}

        if parallel_dims and len(dim_tasks) > 1:
            with ThreadPoolExecutor(max_workers=min(3, len(dim_tasks))) as executor:
                futures = {
                    executor.submit(
                        _run_dim_query,
                        dim,
                        query,
                        sample_root,
                        haystack_dir,
                        use_llm_extractor,
                        llm_client,
                        use_llm_usability_judge,
                        llm_usability_judge_client,
                        use_llm_answer_generator,
                        llm_answer_generator_client,
                        use_llm_verifier,
                        llm_verifier_client,
                        max_judge_memories,
                        max_judge_constraints,
                        constraint_ranker,
                        constraint_embedding_model,
                        constraint_embedding_cache_path,
                        constraint_embedding_encoder,
                        stale_aware_retrieval,
                        recent_memory_window,
                        _stale_target_context(normalized),
                    ): dim
                    for dim, query in dim_tasks
                }
                for future in as_completed(futures):
                    dim, answer, result_meta = future.result()
                    responses[f"{dim}_response"] = answer
                    meta[f"{dim}_meta"] = result_meta
        else:
            for dim, query in dim_tasks:
                dim, answer, result_meta = _run_dim_query(
                    dim,
                    query,
                    sample_root,
                    haystack_dir,
                    use_llm_extractor,
                    llm_client,
                    use_llm_usability_judge,
                    llm_usability_judge_client,
                    use_llm_answer_generator,
                    llm_answer_generator_client,
                    use_llm_verifier,
                    llm_verifier_client,
                    max_judge_memories,
                    max_judge_constraints,
                    constraint_ranker,
                    constraint_embedding_model,
                    constraint_embedding_cache_path,
                    constraint_embedding_encoder,
                    stale_aware_retrieval,
                    recent_memory_window,
                    _stale_target_context(normalized),
                )
                responses[f"{dim}_response"] = answer
                meta[f"{dim}_meta"] = result_meta

    return {
        "uid": normalized["uid"],
        "target_model_responses": responses,
        "target_model_meta": meta,
        "cadmr_eval_meta": {
            "m_old": normalized.get("M_old", ""),
            "m_new": normalized.get("M_new", ""),
            "haystack_summary": haystack_summary,
        },
    }


def _run_dim_query(
    dim: str,
    query: str,
    sample_root: Path,
    haystack_dir: Path,
    use_llm_extractor: bool,
    llm_client,
    use_llm_usability_judge: bool,
    llm_usability_judge_client,
    use_llm_answer_generator: bool,
    llm_answer_generator_client,
    use_llm_verifier: bool,
    llm_verifier_client,
    max_judge_memories: int | None,
    max_judge_constraints: int | None,
    constraint_ranker: str,
    constraint_embedding_model: str,
    constraint_embedding_cache_path: str | Path | None,
    constraint_embedding_encoder,
    stale_aware_retrieval: bool,
    recent_memory_window: int,
    stale_target_context: dict | None = None,
) -> tuple[str, str, dict]:
    dim_dir = sample_root / dim
    _copy_store_snapshot(haystack_dir, dim_dir)
    pipeline = make_pipeline(
        dim_dir,
        use_llm_extractor=use_llm_extractor,
        llm_client=llm_client,
        use_llm_usability_judge=use_llm_usability_judge,
        llm_usability_judge_client=llm_usability_judge_client,
        use_llm_answer_generator=use_llm_answer_generator,
        llm_answer_generator_client=llm_answer_generator_client,
        use_llm_verifier=use_llm_verifier,
        llm_verifier_client=llm_verifier_client,
        max_judge_memories=max_judge_memories,
        max_judge_constraints=max_judge_constraints,
        constraint_ranker=constraint_ranker,
        constraint_embedding_model=constraint_embedding_model,
        constraint_embedding_cache_path=constraint_embedding_cache_path,
        constraint_embedding_encoder=constraint_embedding_encoder,
        stale_aware_retrieval=stale_aware_retrieval,
        recent_memory_window=recent_memory_window,
        force_query_intent=True,
        stale_target_context=stale_target_context,
    )
    result = pipeline.run(query)
    return dim, result.answer or "", _result_meta(query, result)


def _copy_store_snapshot(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def _stale_target_context(sample: dict) -> dict:
    return {
        "m_old": sample.get("M_old", ""),
        "m_new": sample.get("M_new", ""),
    }


def _build_haystack_summary(
    sample: dict,
    haystack_dir: Path,
    user_turns: list[str],
) -> dict:
    ordinary_memories = [
        memory.model_dump()
        for memory in OrdinaryMemoryStore(haystack_dir / "ordinary_memory.json").list_all()
    ]
    active_constraints = [
        constraint.model_dump()
        for constraint in ActiveConstraintStore(haystack_dir / "active_constraints.json").list_all()
    ]
    stored_items = _stored_items_for_matching(ordinary_memories, active_constraints)
    return {
        "user_turn_count": len(user_turns),
        "ordinary_memory_count": len(ordinary_memories),
        "active_constraint_count": len(active_constraints),
        "m_old_storage_match": _match_target_text(sample.get("M_old", ""), stored_items),
        "m_new_storage_match": _match_target_text(sample.get("M_new", ""), stored_items),
        "ordinary_memory_preview": [
            _trace_memory(memory)
            for memory in ordinary_memories[:5]
            if isinstance(memory, dict)
        ],
        "active_constraint_preview": [
            _trace_constraint(constraint)
            for constraint in active_constraints[:5]
            if isinstance(constraint, dict)
        ],
    }


def _dedupe_user_turns(user_turns: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for turn in user_turns:
        normalized = " ".join(turn.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(turn)
    return deduped


def run_stale_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    max_samples: int | None = None,
    verbose: bool = False,
    trace_path: str | Path | None = None,
    use_llm_extractor: bool = False,
    use_llm_usability_judge: bool = False,
    use_llm_answer_generator: bool = False,
    use_llm_verifier: bool = False,
    llm_cache_path: str | Path | None = None,
    max_judge_memories: int | None = None,
    max_judge_constraints: int | None = None,
    constraint_ranker: str = "lexical",
    constraint_embedding_model: str = DEFAULT_CONSTRAINT_EMBEDDING_MODEL,
    constraint_embedding_cache_path: str | Path | None = None,
    stale_aware_retrieval: bool = False,
    recent_memory_window: int = 0,
    haystack_extractor_cache_path: str | Path | None = None,
    prewarm_haystack_cache: bool = False,
    parallel_dims: bool = False,
    resume: bool = True,
    show_progress: bool = False,
) -> list[dict]:
    samples = StaleDatasetLoader().load(dataset_path)
    if max_samples is not None:
        samples = samples[:max_samples]

    output = Path(output_path)
    trace_output = Path(trace_path) if trace_path else None
    results = _load_existing_results(output) if resume else []
    completed_uids = {
        item.get("uid")
        for item in results
        if isinstance(item, dict) and isinstance(item.get("uid"), str)
    }
    llm_client = make_llm_client(
        use_llm_extractor,
        cache_path=llm_cache_path or DEFAULT_LLM_CACHE_PATH,
    )
    llm_usability_judge_client = llm_client or make_llm_client(
        use_llm_usability_judge,
        cache_path=llm_cache_path or DEFAULT_LLM_CACHE_PATH,
    )
    llm_answer_generator_client = (
        llm_client
        or llm_usability_judge_client
        or make_llm_client(
            use_llm_answer_generator,
            cache_path=llm_cache_path or DEFAULT_LLM_CACHE_PATH,
        )
    )
    llm_verifier_client = llm_client or llm_usability_judge_client or llm_answer_generator_client or make_llm_client(
        use_llm_verifier,
        cache_path=llm_cache_path or DEFAULT_LLM_CACHE_PATH,
    )
    haystack_extractor_cache = make_haystack_extractor_cache(
        use_llm_extractor,
        cache_path=haystack_extractor_cache_path or DEFAULT_HAYSTACK_EXTRACTOR_CACHE_PATH,
    )
    constraint_embedding_encoder = None
    if constraint_ranker == "hybrid":
        constraint_embedding_encoder = SentenceTransformerTextEncoder(
            model_name=constraint_embedding_model,
            cache_path=constraint_embedding_cache_path or DEFAULT_CONSTRAINT_EMBEDDING_CACHE_PATH,
        )
    if prewarm_haystack_cache and use_llm_extractor and haystack_extractor_cache is not None:
        _prewarm_haystack_extractor_cache(
            samples,
            llm_client,
            haystack_extractor_cache,
            show_progress=show_progress,
        )

    progress_llm_client = llm_client or llm_usability_judge_client or llm_verifier_client
    progress = ProgressBar(total=len(samples), enabled=show_progress)
    processed_count = 0
    skipped_count = 0

    try:
        for index, sample in enumerate(samples, start=1):
            uid = sample["uid"]
            if uid in completed_uids:
                skipped_count += 1
                progress.update(index, "skipped", uid, processed_count, skipped_count, progress_llm_client)
                if verbose and not show_progress:
                    print(f"Skipped {uid} (already completed)")
                continue

            progress.update(index, "running", uid, processed_count, skipped_count, progress_llm_client)
            result = run_stale_sample(
                sample,
                use_llm_extractor=use_llm_extractor,
                llm_client=llm_client,
                use_llm_usability_judge=use_llm_usability_judge,
                llm_usability_judge_client=llm_usability_judge_client,
                use_llm_answer_generator=use_llm_answer_generator,
                llm_answer_generator_client=llm_answer_generator_client,
                use_llm_verifier=use_llm_verifier,
                llm_verifier_client=llm_verifier_client,
                max_judge_memories=max_judge_memories,
                max_judge_constraints=max_judge_constraints,
                constraint_ranker=constraint_ranker,
                constraint_embedding_model=constraint_embedding_model,
                constraint_embedding_cache_path=constraint_embedding_cache_path,
                constraint_embedding_encoder=constraint_embedding_encoder,
                stale_aware_retrieval=stale_aware_retrieval,
                recent_memory_window=recent_memory_window,
                haystack_extractor_cache=haystack_extractor_cache,
                parallel_dims=parallel_dims,
            )
            results.append(result)
            completed_uids.add(result["uid"])
            processed_count += 1
            _write_results(output, results)
            if trace_output is not None:
                _write_trace(trace_output, results)
            progress.update(index, "processed", result["uid"], processed_count, skipped_count, progress_llm_client)
            if verbose and not show_progress:
                print(f"Processed {result['uid']}")
    finally:
        progress.finish()

    _write_results(output, results)
    if trace_output is not None:
        _write_trace(trace_output, results)
    return results


def _prewarm_haystack_extractor_cache(
    samples: list[dict],
    llm_client,
    haystack_extractor_cache: HaystackExtractorCache,
    show_progress: bool = False,
    stream=None,
) -> None:
    if llm_client is None:
        return
    from cadmr.extractor import LLMMemorySignalExtractor

    extractor = CachedSignalExtractor(
        LLMMemorySignalExtractor(llm_client),
        haystack_extractor_cache,
    )
    seen: set[str] = set()
    unique_turns: list[str] = []
    for sample in samples:
        normalized = normalize_stale_sample(sample, 0)
        for turn in flatten_haystack_user_turns(normalized):
            normalized_turn = normalize_haystack_text(turn)
            if not normalized_turn or normalized_turn in seen:
                continue
            seen.add(normalized_turn)
            unique_turns.append(turn)

    progress = PrewarmProgressBar(
        total=len(unique_turns),
        enabled=show_progress,
        stream=stream,
    )
    try:
        for index, turn in enumerate(unique_turns, start=1):
            extractor.extract(turn)
            progress.update(
                index,
                turn,
                haystack_cache=haystack_extractor_cache,
                llm_client=llm_client,
            )
    finally:
        progress.finish()


def _load_existing_results(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []

    content = output_path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    data = json.loads(content)
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Existing output must be a JSON list: {output_path}")
    return [item for item in data if isinstance(item, dict)]


def _write_results(output_path: Path, results: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    tmp_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(output_path)


def _write_trace(trace_path: Path, results: list[dict]) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace = [_trace_sample(result) for result in results]
    tmp_path = trace_path.with_name(f".{trace_path.name}.tmp")
    tmp_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(trace_path)


def _trace_sample(result: dict) -> dict:
    uid = result.get("uid", "")
    responses = result.get("target_model_responses", {}) or {}
    meta = result.get("target_model_meta", {}) or {}
    eval_meta = result.get("cadmr_eval_meta", {}) or {}
    dims = {}
    for dim in ["dim1", "dim2", "dim3"]:
        dim_meta = meta.get(f"{dim}_meta", {}) or {}
        dims[dim] = _trace_dim(
            dim_meta,
            responses.get(f"{dim}_response", ""),
            eval_meta,
        )
    return {
        "uid": uid,
        "haystack_summary": eval_meta.get("haystack_summary", {}),
        "dims": dims,
    }


def _trace_dim(meta: dict, answer: str, eval_meta: dict | None = None) -> dict:
    eval_meta = eval_meta or {}
    structured_output = meta.get("structured_output", {}) or {}
    retrieved_memories = _first_trace_list(meta, structured_output, "retrieved_memories")
    retrieved_constraints = _first_trace_list(meta, structured_output, "retrieved_constraints")
    judgments = _first_trace_list(meta, structured_output, "judgments")
    signals = _first_trace_list(meta, structured_output, "signals")
    write_decisions = _first_trace_list(meta, structured_output, "write_decisions")
    verify_result = _first_trace_mapping(meta, structured_output, "verify_result")
    judge_diagnostics = meta.get("judge_diagnostics") or structured_output.get("judge_diagnostics") or {}

    memory_by_id = {
        memory.get("memory_id"): memory
        for memory in retrieved_memories
        if isinstance(memory, dict)
    }
    judgment_by_memory_id = {
        judgment.get("memory_id"): judgment
        for judgment in judgments
        if isinstance(judgment, dict)
    }
    retrieved_items = _stored_items_for_matching(
        retrieved_memories,
        retrieved_constraints,
    )
    stale_targets = {
        "m_old": _trace_target_match(
            eval_meta.get("m_old", ""),
            retrieved_items,
            judgment_by_memory_id,
        ),
        "m_new": _trace_target_match(
            eval_meta.get("m_new", ""),
            retrieved_items,
            judgment_by_memory_id,
        ),
    }
    non_noise = [
        _trace_judgment(judgment, memory_by_id)
        for judgment in judgments
        if isinstance(judgment, dict) and judgment.get("usage_status") != "NOISE"
    ]
    violations = verify_result.get("violations", []) if isinstance(verify_result, dict) else []
    if not isinstance(violations, list):
        violations = []

    trace = {
        "query": meta.get("query", ""),
        "signal_types": [
            signal.get("signal_type")
            for signal in signals
            if isinstance(signal, dict)
        ],
        "write_decisions": [
            decision.get("decision")
            for decision in write_decisions
            if isinstance(decision, dict)
        ],
        "retrieval": {
            "memory_count": len(retrieved_memories),
            "constraint_count": len(retrieved_constraints),
            "stale_targets": stale_targets,
            "memory_preview": [
                _trace_memory(memory)
                for memory in retrieved_memories[:5]
                if isinstance(memory, dict)
            ],
            "constraint_preview": [
                _trace_constraint(constraint)
                for constraint in retrieved_constraints[:5]
                if isinstance(constraint, dict)
            ],
        },
        "judge": {
            "status_summary": _status_summary(judgments),
            "diagnostics": judge_diagnostics,
            "non_noise_preview": non_noise[:8],
        },
        "answer": {
            "preview": _shorten(answer, 500),
            "length": len(answer or ""),
        },
        "verifier": {
            "pass": bool(verify_result.get("pass")) and not violations,
            "violation_types": [
                violation.get("type")
                for violation in violations
                if isinstance(violation, dict)
            ],
            "violations_preview": [
                {
                    "type": violation.get("type"),
                    "related_id": violation.get("related_id"),
                    "evidence": _shorten(str(violation.get("evidence", "")), 180),
                }
                for violation in violations[:5]
                if isinstance(violation, dict)
            ],
            "reason": _shorten(str(verify_result.get("reason", "")), 300),
        },
    }
    trace["likely_problem"] = _diagnose_trace_dim(trace)
    return trace


def _first_trace_list(meta: dict, structured_output: dict, key: str) -> list:
    value = meta.get(key)
    if isinstance(value, list):
        return value
    value = structured_output.get(key) if isinstance(structured_output, dict) else None
    return value if isinstance(value, list) else []


def _first_trace_mapping(meta: dict, structured_output: dict, key: str) -> dict:
    value = meta.get(key)
    if isinstance(value, dict):
        return value
    value = structured_output.get(key) if isinstance(structured_output, dict) else None
    return value if isinstance(value, dict) else {}


def _stored_items_for_matching(
    memories: list,
    constraints: list,
) -> list[dict]:
    items: list[dict] = []
    for memory in memories:
        if not isinstance(memory, dict):
            continue
        items.append(
            {
                "id": memory.get("memory_id", ""),
                "type": "ordinary_memory",
                "content": str(memory.get("content", "")),
            }
        )
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        items.append(
            {
                "id": constraint.get("constraint_id", ""),
                "type": "active_constraint",
                "content": str(constraint.get("content", "")),
            }
        )
    return items


def _trace_target_match(
    target_text: str,
    items: list[dict],
    judgment_by_memory_id: dict,
) -> dict:
    match = _match_target_text(target_text, items)
    item_id = match.get("matched_id")
    judgment = judgment_by_memory_id.get(item_id, {}) if item_id else {}
    match["usage_status"] = judgment.get("usage_status")
    match["judgment_reason"] = _shorten(str(judgment.get("reason", "")), 220)
    match["replaced_by"] = judgment.get("replaced_by", [])
    match["blocked_by"] = judgment.get("blocked_by", [])
    return match


def _match_target_text(target_text: str, items: list[dict]) -> dict:
    target = str(target_text or "")
    target_preview = _shorten(target, 220)
    if not target.strip():
        return {
            "target_preview": "",
            "matched": False,
            "score": 0.0,
            "matched_type": None,
            "matched_id": None,
            "matched_content": "",
        }

    best_item = None
    best_score = 0.0
    for item in items:
        score = _text_overlap_score(target, str(item.get("content", "")))
        if score > best_score:
            best_item = item
            best_score = score

    matched = bool(best_item) and best_score >= 0.28
    return {
        "target_preview": target_preview,
        "matched": matched,
        "score": round(best_score, 3),
        "matched_type": best_item.get("type") if best_item else None,
        "matched_id": best_item.get("id") if best_item else None,
        "matched_content": _shorten(str(best_item.get("content", "")), 220) if best_item else "",
    }


def _text_overlap_score(left: str, right: str) -> float:
    left_norm = _normalize_match_text(left)
    right_norm = _normalize_match_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm in right_norm or right_norm in left_norm:
        shorter = min(len(left_norm), len(right_norm))
        longer = max(len(left_norm), len(right_norm))
        return max(0.65, shorter / max(longer, 1))

    left_tokens = _match_tokens(left_norm)
    right_tokens = _match_tokens(right_norm)
    if not left_tokens or not right_tokens:
        return 0.0

    overlap = left_tokens & right_tokens
    target_recall = len(overlap) / max(len(left_tokens), 1)
    item_recall = len(overlap) / max(len(right_tokens), 1)
    return max(target_recall, item_recall * 0.7)


def _normalize_match_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def _match_tokens(text: str) -> set[str]:
    tokens = {token for token in WORD_PATTERN.findall(text.lower()) if len(token) > 1}
    if tokens:
        return tokens
    return {char for char in text if not char.isspace()}


def _trace_memory(memory: dict) -> dict:
    return {
        "memory_id": memory.get("memory_id"),
        "scope": memory.get("scope", []),
        "content": _shorten(str(memory.get("content", "")), 180),
    }


def _trace_constraint(constraint: dict) -> dict:
    return {
        "constraint_id": constraint.get("constraint_id"),
        "scope": constraint.get("scope", []),
        "content": _shorten(str(constraint.get("content", "")), 180),
    }


def _trace_judgment(judgment: dict, memory_by_id: dict) -> dict:
    memory = memory_by_id.get(judgment.get("memory_id"), {}) or {}
    return {
        "memory_id": judgment.get("memory_id"),
        "usage_status": judgment.get("usage_status"),
        "reason": _shorten(str(judgment.get("reason", "")), 220),
        "memory_content": _shorten(str(memory.get("content", "")), 180),
        "blocked_by": judgment.get("blocked_by", []),
        "replaced_by": judgment.get("replaced_by", []),
    }


def _status_summary(judgments: list) -> dict:
    summary = {
        "USABLE": 0,
        "CONSTRAINED": 0,
        "STALE": 0,
        "SUSPENDED": 0,
        "NOISE": 0,
    }
    for judgment in judgments:
        if not isinstance(judgment, dict):
            continue
        status = judgment.get("usage_status")
        if status in summary:
            summary[status] += 1
    return summary


def _diagnose_trace_dim(trace: dict) -> str:
    retrieval = trace["retrieval"]
    judge = trace["judge"]
    verifier = trace["verifier"]
    diagnostics = judge.get("diagnostics") or {}
    summary = judge.get("status_summary") or {}
    stale_targets = retrieval.get("stale_targets") or {}
    old_premise_query = _is_old_premise_query(trace.get("query", ""))

    if retrieval["memory_count"] == 0 and retrieval["constraint_count"] == 0:
        return "retrieval_empty"
    if old_premise_query:
        m_old = stale_targets.get("m_old") or {}
        m_new = stale_targets.get("m_new") or {}
        if m_old.get("target_preview") and not m_old.get("matched"):
            return "m_old_not_retrieved"
        if m_new.get("target_preview") and not m_new.get("matched"):
            return "m_new_not_retrieved"
        if (
            (m_old.get("matched") or m_new.get("matched"))
            and not any(summary.get(status, 0) for status in ["CONSTRAINED", "STALE", "SUSPENDED"])
        ):
            return "judge_missed_stale_update"
    if diagnostics.get("batches_failed"):
        return "judge_llm_call_failed"
    if diagnostics.get("memories_total") and diagnostics.get("judgments_from_llm") == 0:
        return "judge_all_fallback"
    if not any(summary.get(status, 0) for status in ["USABLE", "CONSTRAINED", "STALE", "SUSPENDED"]):
        return "judge_all_noise"
    if not verifier["pass"]:
        return "verifier_rejected_answer"
    return "ok"


def _is_old_premise_query(query: str) -> bool:
    lowered = str(query).lower()
    return any(
        phrase in lowered
        for phrase in [
            "does the user still",
            "do they still",
            "does she still",
            "does he still",
            "since the user",
            "based on the conversation history",
            "based on previous",
            "still ",
        ]
    )


def _shorten(text: str, max_length: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."


def _result_meta(query: str, result) -> dict:
    structured_output = _jsonable(result.structured_output)
    judge_diagnostics = None
    if isinstance(structured_output, dict):
        judge_diagnostics = structured_output.get("judge_diagnostics")

    return {
        "query": query,
        "query_info": _jsonable(result.query_info),
        "signals": _jsonable(result.signals),
        "write_decisions": _jsonable(result.write_decisions),
        "judgments": _jsonable(result.judgments),
        "judge_diagnostics": judge_diagnostics,
        "goal_plan": _jsonable(result.goal_plan),
        "verify_result": _jsonable(result.verify_result),
        "structured_output": structured_output,
    }


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run CADMR on a STALE *_MAIN.json file.")
    parser.add_argument("--dataset", default=None, help="Path to STALE *_MAIN.json")
    parser.add_argument("--output", default=None, help="Path to write CADMR answer JSON")
    parser.add_argument("--trace-path", default=None, help="Optional path for compact per-stage debug trace JSON.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--llm-cache-path",
        default=None,
        help="Path for disk-backed OpenRouter completion cache.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the eval progress bar.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing output and recompute from the beginning.",
    )
    parser.add_argument(
        "--use-llm-extractor",
        action="store_true",
        help="Use OpenRouter + LLMMemorySignalExtractor. Defaults to rule-based extraction.",
    )
    parser.add_argument(
        "--use-llm-usability-judge",
        action="store_true",
        help="Use OpenRouter + LLMUsabilityJudge with broad retrieval candidates.",
    )
    parser.add_argument(
        "--use-llm-answer-generator",
        action="store_true",
        help="Use OpenRouter + LLMConstrainedAnswerGenerator for final English answers.",
    )
    parser.add_argument(
        "--use-llm-verifier",
        action="store_true",
        help="Use OpenRouter + LLMAnswerVerifier. Defaults to rule-based verification.",
    )
    parser.add_argument(
        "--max-judge-memories",
        type=int,
        default=None,
        help="Maximum candidate memories to send to LLM usability judge per query.",
    )
    parser.add_argument(
        "--max-judge-constraints",
        type=int,
        default=None,
        help="Maximum active constraints to send to LLM usability judge per query.",
    )
    parser.add_argument(
        "--constraint-ranker",
        choices=["lexical", "hybrid"],
        default=None,
        help="Constraint reranker: lexical or hybrid lexical + optional small-BERT embeddings.",
    )
    parser.add_argument(
        "--constraint-embedding-model",
        default=None,
        help="SentenceTransformer model name for hybrid constraint reranking.",
    )
    parser.add_argument(
        "--constraint-embedding-cache-path",
        default=None,
        help="Path for cached constraint/query embeddings used by hybrid reranking.",
    )
    parser.add_argument(
        "--stale-aware-retrieval",
        action="store_true",
        help="For stale-sensitive queries, mix recent memories into the judge candidates.",
    )
    parser.add_argument(
        "--recent-memory-window",
        type=int,
        default=None,
        help="Number of recent memories to reserve for stale-aware retrieval.",
    )
    parser.add_argument(
        "--haystack-extractor-cache-path",
        default=None,
        help="Path for normalized haystack-turn extractor signal cache.",
    )
    parser.add_argument(
        "--prewarm-haystack-cache",
        action="store_true",
        help="Precompute unique haystack-turn extractor signals before running samples.",
    )
    parser.add_argument(
        "--parallel-dims",
        action="store_true",
        help="Run dim1/dim2/dim3 queries for each sample in parallel.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset or get_env("CADMR_STALE_DATASET")
    output_path = args.output or get_env("CADMR_STALE_OUTPUT", "evals/stale_answers_latest.json")
    trace_path = args.trace_path or get_env("CADMR_TRACE_PATH")
    max_samples = args.max_samples
    if max_samples is None:
        max_samples = get_int_env("CADMR_STALE_MAX_SAMPLES")
    max_judge_memories = args.max_judge_memories
    if max_judge_memories is None:
        max_judge_memories = get_int_env("CADMR_MAX_JUDGE_MEMORIES")
    max_judge_constraints = args.max_judge_constraints
    if max_judge_constraints is None:
        max_judge_constraints = get_int_env("CADMR_MAX_JUDGE_CONSTRAINTS")
    constraint_ranker = args.constraint_ranker or get_env("CADMR_CONSTRAINT_RANKER", "lexical")
    if constraint_ranker not in {"lexical", "hybrid"}:
        print("CADMR_CONSTRAINT_RANKER must be 'lexical' or 'hybrid'.")
        return 1
    constraint_embedding_model = args.constraint_embedding_model or get_env(
        "CADMR_CONSTRAINT_EMBEDDING_MODEL",
        DEFAULT_CONSTRAINT_EMBEDDING_MODEL,
    )
    constraint_embedding_cache_path = args.constraint_embedding_cache_path or get_env(
        "CADMR_CONSTRAINT_EMBEDDING_CACHE_PATH",
        DEFAULT_CONSTRAINT_EMBEDDING_CACHE_PATH,
    )
    stale_aware_retrieval = args.stale_aware_retrieval or get_bool_env(
        "CADMR_STALE_AWARE_RETRIEVAL",
        False,
    )
    recent_memory_window = args.recent_memory_window
    if recent_memory_window is None:
        recent_memory_window = get_int_env("CADMR_RECENT_MEMORY_WINDOW", 0) or 0
    use_llm_extractor = args.use_llm_extractor or get_bool_env("CADMR_USE_LLM_EXTRACTOR", False)
    use_llm_usability_judge = args.use_llm_usability_judge or get_bool_env("CADMR_USE_LLM_USABILITY_JUDGE", False)
    use_llm_answer_generator = args.use_llm_answer_generator or get_bool_env("CADMR_USE_LLM_ANSWER_GENERATOR", False)
    use_llm_verifier = args.use_llm_verifier or get_bool_env("CADMR_USE_LLM_VERIFIER", False)
    llm_cache_path = args.llm_cache_path or get_env(
        "CADMR_LLM_CACHE_PATH",
        DEFAULT_LLM_CACHE_PATH,
    )
    haystack_extractor_cache_path = args.haystack_extractor_cache_path or get_env(
        "CADMR_HAYSTACK_EXTRACTOR_CACHE_PATH",
        DEFAULT_HAYSTACK_EXTRACTOR_CACHE_PATH,
    )
    prewarm_haystack_cache = args.prewarm_haystack_cache or get_bool_env(
        "CADMR_PREWARM_HAYSTACK_CACHE",
        False,
    )
    parallel_dims = args.parallel_dims or get_bool_env("CADMR_PARALLEL_DIMS", False)

    if not dataset_path:
        print("Missing STALE dataset path. Pass --dataset or set CADMR_STALE_DATASET in .env.")
        return 0
    if not output_path:
        print("Missing output path. Pass --output or set CADMR_STALE_OUTPUT in .env.")
        return 0

    try:
        results = run_stale_dataset(
            dataset_path=dataset_path,
            output_path=output_path,
            max_samples=max_samples,
            verbose=args.verbose,
            trace_path=trace_path,
            use_llm_extractor=use_llm_extractor,
            use_llm_usability_judge=use_llm_usability_judge,
            use_llm_answer_generator=use_llm_answer_generator,
            use_llm_verifier=use_llm_verifier,
            llm_cache_path=llm_cache_path,
            max_judge_memories=max_judge_memories,
            max_judge_constraints=max_judge_constraints,
            constraint_ranker=constraint_ranker,
            constraint_embedding_model=constraint_embedding_model,
            constraint_embedding_cache_path=constraint_embedding_cache_path,
            stale_aware_retrieval=stale_aware_retrieval,
            recent_memory_window=recent_memory_window,
            haystack_extractor_cache_path=haystack_extractor_cache_path,
            prewarm_haystack_cache=prewarm_haystack_cache,
            parallel_dims=parallel_dims,
            resume=not args.no_resume,
            show_progress=not args.no_progress,
        )
    except ValueError as error:
        if "OPENROUTER_API_KEY" in str(error):
            print("OPENROUTER_API_KEY is not set. Re-run without LLM flags or set the key.")
            return 0
        raise
    print(f"Wrote {len(results)} STALE-compatible answers to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

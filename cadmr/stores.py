"""JSON-backed store implementations for CADMR memory data."""

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from cadmr.schemas import ActiveConstraint, OrdinaryMemory, RawInteraction


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _dump_model(model: BaseModel) -> dict:
    return model.model_dump()


class RawInteractionLog:
    """Append-only JSONL store for raw interactions."""

    def __init__(self, path: str | Path = "data/raw_interaction_log.jsonl") -> None:
        self.path = Path(path)
        _ensure_parent(self.path)
        self.path.touch(exist_ok=True)

    def append(self, interaction: RawInteraction) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(_dump_model(interaction), ensure_ascii=False) + "\n")

    def list_all(self) -> list[RawInteraction]:
        interactions: list[RawInteraction] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if stripped:
                    interactions.append(RawInteraction(**json.loads(stripped)))
        return interactions

    def clear(self) -> None:
        self.path.write_text("", encoding="utf-8")


class _JsonListStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        _ensure_parent(self.path)
        if not self.path.exists():
            self._write_raw([])

    def _read_raw(self) -> list[dict]:
        if not self.path.exists():
            self._write_raw([])
            return []

        content = self.path.read_text(encoding="utf-8").strip()
        if not content:
            return []

        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {self.path}")
        return data

    def _write_raw(self, items: list[dict]) -> None:
        self.path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def clear(self) -> None:
        self._write_raw([])


class OrdinaryMemoryStore(_JsonListStore):
    """JSON list store for ordinary memories."""

    def __init__(self, path: str | Path = "data/ordinary_memory.json") -> None:
        super().__init__(path)

    def add(self, memory: OrdinaryMemory) -> None:
        items = self._read_raw()
        items.append(_dump_model(memory))
        self._write_raw(items)

    def update(self, memory: OrdinaryMemory) -> None:
        items = self._read_raw()
        for index, item in enumerate(items):
            if item.get("memory_id") == memory.memory_id:
                items[index] = _dump_model(memory)
                self._write_raw(items)
                return
        raise KeyError(f"Memory not found: {memory.memory_id}")

    def mark_stale(self, memory_id: str) -> None:
        memories = self.list_all()
        for memory in memories:
            if memory.memory_id == memory_id:
                memory.status = "stale"
                memory.updated_at = datetime.now(UTC).isoformat()
                self.update(memory)
                return
        raise KeyError(f"Memory not found: {memory_id}")

    def list_all(self) -> list[OrdinaryMemory]:
        return [OrdinaryMemory(**item) for item in self._read_raw()]

    def get_active(self) -> list[OrdinaryMemory]:
        return [memory for memory in self.list_all() if memory.status == "active"]

    def search_by_scope(self, scopes: list[str]) -> list[OrdinaryMemory]:
        requested_scopes = set(scopes)
        return [
            memory
            for memory in self.list_all()
            if requested_scopes.intersection(memory.scope)
        ]


class ActiveConstraintStore(_JsonListStore):
    """JSON list store for active constraints."""

    def __init__(self, path: str | Path = "data/active_constraints.json") -> None:
        super().__init__(path)

    def add(self, constraint: ActiveConstraint) -> None:
        items = self._read_raw()
        items.append(_dump_model(constraint))
        self._write_raw(items)

    def list_all(self) -> list[ActiveConstraint]:
        return [ActiveConstraint(**item) for item in self._read_raw()]

    def get_active(self) -> list[ActiveConstraint]:
        return [
            constraint
            for constraint in self.list_all()
            if constraint.status == "active"
        ]

    def search_by_scope(self, scopes: list[str]) -> list[ActiveConstraint]:
        requested_scopes = set(scopes)
        return [
            constraint
            for constraint in self.list_all()
            if requested_scopes.intersection(constraint.scope)
        ]

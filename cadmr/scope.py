"""Scope normalization helpers for CADMR."""


def canonicalize_scope(scope: str) -> list[str]:
    """Normalize formatting only; semantic scope selection is owned by the LLM."""
    normalized = scope.strip().casefold().replace(" ", "_").replace("-", "_")
    return [normalized] if normalized else []


def canonicalize_scopes(scopes: list[str]) -> list[str]:
    """Normalize scope labels while preserving order and removing duplicates."""
    canonical: list[str] = []
    for scope in scopes:
        if not isinstance(scope, str):
            continue
        for normalized in canonicalize_scope(scope):
            if normalized not in canonical:
                canonical.append(normalized)
    return canonical or ["general"]

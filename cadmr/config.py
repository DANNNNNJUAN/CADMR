"""Small .env configuration helper for CADMR scripts."""

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: str | Path | None = None, override: bool = False) -> dict[str, str]:
    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _clean_value(value.strip())
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def get_env(name: str, default: str | None = None) -> str | None:
    load_dotenv()
    return os.getenv(name, default)


def get_int_env(name: str, default: int | None = None) -> int | None:
    value = get_env(name)
    if value is None or value == "":
        return default
    return int(value)


def get_bool_env(name: str, default: bool = False) -> bool:
    value = get_env(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _clean_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value

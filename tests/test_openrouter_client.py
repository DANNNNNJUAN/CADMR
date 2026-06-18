import pytest

import cadmr.openrouter_client as openrouter
from cadmr.openrouter_client import OpenRouterClient


def test_openrouter_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(openrouter, "load_dotenv", lambda: {})

    with pytest.raises(ValueError):
        OpenRouterClient()


def test_strip_json_fence():
    content = '```json\n{"signals": []}\n```'

    assert OpenRouterClient._strip_json_fence(content) == '{"signals": []}'


def test_parse_json_content():
    parsed = OpenRouterClient._parse_json_content('{"signals": []}')

    assert parsed == {"signals": []}


def test_openrouter_client_reads_model_config_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "test/model")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.test")
    monkeypatch.setenv("OPENROUTER_TIMEOUT", "5")

    client = OpenRouterClient()

    assert client.model == "test/model"
    assert client.base_url == "https://example.test"
    assert client.timeout == 5

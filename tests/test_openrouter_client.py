import pytest

from cadmr.openrouter_client import OpenRouterClient


def test_openrouter_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError):
        OpenRouterClient()


def test_strip_json_fence():
    content = '```json\n{"signals": []}\n```'

    assert OpenRouterClient._strip_json_fence(content) == '{"signals": []}'


def test_parse_json_content():
    parsed = OpenRouterClient._parse_json_content('{"signals": []}')

    assert parsed == {"signals": []}

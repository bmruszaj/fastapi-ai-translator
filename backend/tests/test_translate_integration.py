from fastapi.testclient import TestClient

from app.domain.language_rules import SUPPORTED
from tests.fakes import FakeTranslator


def test_languages_returns_supported_languages(
    client: TestClient,
) -> None:
    # When
    response = client.get("/api/languages")

    # Then
    assert response.status_code == 200
    assert response.json() == {"languages": SUPPORTED}


def test_frontend_config_returns_runtime_limits(
    client: TestClient,
) -> None:
    response = client.get("/api/frontend-config")

    assert response.status_code == 200
    assert response.json() == {
        "max_input_tokens": 400,
        "max_chars_per_token": 2,
        "max_input_chars": 800,
        "warning_input_chars": 600,
    }


def test_translate_returns_normalized_translation_for_valid_payload(
    client: TestClient,
    fake_translator: FakeTranslator,
    default_translate_payload: dict[str, str],
) -> None:
    # Given
    translation_request = default_translate_payload.copy()
    translation_request["text"] = "  Hello world  "
    translation_request["source_language"] = "EN"

    # When
    response = client.post(
        "/api/translate",
        json=translation_request,
    )

    # Then
    assert response.status_code == 200
    assert response.json() == {
        "translated_text": "Hello world|en->fr",
        "source_language": "en",
        "target_language": "fr",
        "model": "fake-model",
    }
    assert fake_translator.calls == [("Hello world", "en", "fr")]

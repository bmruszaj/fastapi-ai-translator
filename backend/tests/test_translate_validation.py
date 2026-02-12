import pytest
from fastapi.testclient import TestClient

from app.application.ports.errors import InputTooLongError, TranslationExecutionError
from app.domain.errors import DomainError
from tests.fakes import FakeTranslator


@pytest.mark.parametrize(
    ("payload", "expected_status", "expected_code", "expected_message"),
    [
        (
            {"text": "hello", "source_language": "xx", "target_language": "fr"},
            400,
            "unsupported_language",
            "Unsupported language: xx",
        ),
        (
            {"text": "hello", "source_language": "en", "target_language": "en"},
            400,
            "invalid_language_pair",
            "Invalid language pair: en -> en",
        ),
        (
            {"text": "   ", "source_language": "en", "target_language": "fr"},
            422,
            "empty_text",
            "Text must not be empty.",
        ),
    ],
    ids=[
        "unsupported_language",
        "same_language_pair",
        "whitespace_only_text",
    ],
)
def test_translate_returns_domain_error_for_invalid_payload(
    client: TestClient,
    payload: dict[str, str],
    expected_status: int,
    expected_code: str,
    expected_message: str,
) -> None:
    # Given
    request_payload = payload.copy()

    # When
    response = client.post("/api/translate", json=request_payload)
    response_body = response.json()

    # Then
    assert response.status_code == expected_status
    assert response_body["code"] == expected_code
    assert response_body["message"] == expected_message


@pytest.mark.parametrize(
    "payload",
    [
        {"source_language": "en", "target_language": "fr"},
        {"text": 123, "source_language": "en", "target_language": "fr"},
    ],
    ids=["missing_text_field", "wrong_text_type"],
)
def test_translate_returns_validation_error_for_schema_invalid_payload(
    client: TestClient,
    payload: dict[str, str | int],
) -> None:
    # Given

    # When
    response = client.post("/api/translate", json=payload)

    # Then
    assert response.status_code == 422
    assert response.json() == {
        "code": "validation_error",
        "message": "Request payload validation failed.",
    }


def test_translate_returns_internal_error_for_runtime_error(
    client: TestClient,
    fake_translator: FakeTranslator,
    default_translate_payload: dict[str, str],
) -> None:
    # Given
    fake_translator.translate_error = RuntimeError("boom")

    # When
    response = client.post("/api/translate", json=default_translate_payload)

    # Then
    assert response.status_code == 500
    assert response.json() == {
        "code": "internal_error",
        "message": "Internal server error",
    }


def test_translate_returns_unavailable_error_for_translator_port_failure(
    client: TestClient,
    fake_translator: FakeTranslator,
    default_translate_payload: dict[str, str],
) -> None:
    # Given
    sensitive_message = (
        "Tokenizer for model 'facebook/nllb-200-distilled-600M' does not provide "
        "a valid target language token for 'fra_Latn'."
    )

    fake_translator.translate_error = TranslationExecutionError(sensitive_message)

    # When
    response = client.post("/api/translate", json=default_translate_payload)

    # Then
    assert response.status_code == 503
    assert response.json() == {
        "code": "translation_execution_failed",
        "message": "Translation service temporarily unavailable.",
    }


def test_translate_returns_text_too_long_for_tokenizer_limit(
    client: TestClient,
    fake_translator: FakeTranslator,
    default_translate_payload: dict[str, str],
) -> None:
    max_input_tokens = client.app.state.container.settings.max_input_tokens
    actual_tokens = max_input_tokens + 1
    fake_translator.translate_error = InputTooLongError(
        f"Text exceeds max length of {max_input_tokens} tokens "
        f"(received {actual_tokens})."
    )

    response = client.post("/api/translate", json=default_translate_payload)

    assert response.status_code == 422
    assert response.json() == {
        "code": "text_too_long",
        "message": (
            f"Text exceeds max length of {max_input_tokens} tokens "
            f"(received {actual_tokens})."
        ),
    }


def test_translate_returns_domain_error_for_unknown_domain_error(
    client: TestClient,
    fake_translator: FakeTranslator,
    default_translate_payload: dict[str, str],
) -> None:
    # Given
    class UnknownDomainError(DomainError):
        code = "unknown_domain_error"

    fake_translator.translate_error = UnknownDomainError(
        "Unexpected domain validation error."
    )

    # When
    response = client.post("/api/translate", json=default_translate_payload)

    # Then
    assert response.status_code == 422
    assert response.json() == {
        "code": "unknown_domain_error",
        "message": "Unexpected domain validation error.",
    }

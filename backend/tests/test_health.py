from fastapi.testclient import TestClient

from tests.fakes import FakeTranslator


def test_health_returns_model_status(
    client: TestClient,
) -> None:
    # When
    response = client.get("/health")

    # Then
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "fake-model",
        "loaded": True,
    }


def test_health_returns_unloaded_status_when_translator_not_ready(
    client: TestClient,
    fake_translator: FakeTranslator,
) -> None:
    # Given
    fake_translator.ready = False

    # When
    response = client.get("/health")

    # Then
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "fake-model",
        "loaded": False,
    }


def test_health_returns_container_not_initialized_error_when_container_is_missing(
    client: TestClient,
) -> None:
    # Given
    del client.app.state.container

    # When
    response = client.get("/health")

    # Then
    assert response.status_code == 500
    assert response.json() == {
        "code": "container_not_initialized",
        "message": "Application container is not initialized.",
    }


def test_health_returns_invalid_container_type_error(
    client: TestClient,
) -> None:
    # Given
    client.app.state.container = object()

    # When
    response = client.get("/health")

    # Then
    assert response.status_code == 500
    assert response.json() == {
        "code": "invalid_container_type",
        "message": "Application container has invalid type: object.",
    }

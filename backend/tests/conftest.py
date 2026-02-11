from collections.abc import Iterator
from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.application.use_cases.translate_text import TranslateTextUseCase  # noqa: E402
from app.application.use_cases.get_health_status import GetHealthStatusUseCase  # noqa: E402
from app.bootstrap.container import AppContainer  # noqa: E402
from app.core.config import AppSettings, DEFAULT_MAX_INPUT_CHARS  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.fakes import FakeTranslator  # noqa: E402


@pytest.fixture
def fake_translator() -> FakeTranslator:
    """Provide a fake translator instance for tests."""
    return FakeTranslator()


@pytest.fixture
def app(fake_translator: FakeTranslator) -> FastAPI:
    """Create a FastAPI app wired with fake dependencies."""
    settings = AppSettings(
        model_id="fake-model",
        max_input_chars=DEFAULT_MAX_INPUT_CHARS,
        max_new_tokens=32,
        device="cpu",
    )

    def container_factory(runtime_settings: AppSettings) -> AppContainer:
        translate_use_case = TranslateTextUseCase(
            translator=fake_translator,
            model_id=runtime_settings.model_id,
            max_input_chars=runtime_settings.max_input_chars,
        )
        health_use_case = GetHealthStatusUseCase(
            model_id=runtime_settings.model_id,
            is_loaded=fake_translator.is_ready,
        )
        return AppContainer(
            settings=runtime_settings,
            get_health_status_use_case=health_use_case,
            translate_text_use_case=translate_use_case,
        )

    return create_app(settings=settings, container_factory=container_factory)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Create an HTTP client for backend API tests."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def default_translate_payload() -> dict[str, str]:
    """Provide a valid translation payload for HTTP translation tests."""
    return {
        "text": "hello",
        "source_language": "en",
        "target_language": "fr",
    }

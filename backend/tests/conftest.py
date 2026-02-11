from collections.abc import Callable, Iterator
from contextlib import nullcontext
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if TYPE_CHECKING:
    from app.adapters.outbound.nllb_translator import NllbRuntimeDependencies

from app.application.use_cases.translate_text import TranslateTextUseCase  # noqa: E402
from app.application.use_cases.get_health_status import GetHealthStatusUseCase  # noqa: E402
from app.bootstrap.container import AppContainer  # noqa: E402
from app.core.config import AppSettings, DEFAULT_MAX_INPUT_CHARS  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.fakes import FakeTranslator  # noqa: E402
from tests.fakes_nllb import FakeConfig, FakeModel, FakeTensor, FakeTokenizer  # noqa: E402


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


@pytest.fixture
def fake_tensor_cls() -> type[FakeTensor]:
    """Expose tensor fake class for adapter tests."""
    return FakeTensor


@pytest.fixture
def fake_config_cls() -> type[FakeConfig]:
    """Expose config fake class for adapter tests."""
    return FakeConfig


@pytest.fixture
def fake_tokenizer_cls() -> type[FakeTokenizer]:
    """Expose tokenizer fake class for adapter tests."""
    return FakeTokenizer


@pytest.fixture
def fake_model_cls() -> type[FakeModel]:
    """Expose model fake class for adapter tests."""
    return FakeModel


@pytest.fixture
def get_nllb_runtime_dependencies(
    fake_tokenizer_cls: type[FakeTokenizer],
    fake_model_cls: type[FakeModel],
    fake_config_cls: type[FakeConfig],
) -> Callable[..., "NllbRuntimeDependencies"]:
    """Build runtime dependency bundle for NLLB translator tests."""
    from app.adapters.outbound import nllb_translator

    def _make(
        *,
        tokenizer: FakeTokenizer | None = None,
        model: FakeModel | None = None,
        config: FakeConfig | None = None,
        tokenizer_error: Exception | None = None,
        tokenizer_call: dict[str, Any] | None = None,
        config_call: dict[str, Any] | None = None,
        model_call: dict[str, Any] | None = None,
    ) -> "NllbRuntimeDependencies":
        resolved_tokenizer = tokenizer or fake_tokenizer_cls()
        resolved_model = model or fake_model_cls()
        resolved_config = config or fake_config_cls()

        def load_tokenizer(model_id: str, **kwargs: Any) -> FakeTokenizer:
            if tokenizer_call is not None:
                tokenizer_call["model_id"] = model_id
                tokenizer_call["kwargs"] = kwargs
            if tokenizer_error is not None:
                raise tokenizer_error
            return resolved_tokenizer

        def load_config(model_id: str, **kwargs: Any) -> FakeConfig:
            if config_call is not None:
                config_call["model_id"] = model_id
                config_call["kwargs"] = kwargs
            return resolved_config

        def load_model(model_id: str, **kwargs: Any) -> FakeModel:
            if model_call is not None:
                model_call["model_id"] = model_id
                model_call["kwargs"] = kwargs
            return resolved_model

        return nllb_translator.NllbRuntimeDependencies(
            validate_device=lambda _device: None,
            load_tokenizer=load_tokenizer,
            load_config=load_config,
            load_model=load_model,
            inference_mode=lambda: nullcontext(),
        )

    return _make

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
import types
from typing import Any, TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _install_optional_ml_stubs() -> None:
    """Install lightweight torch/transformers stubs when optional ML deps are missing."""
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        torch_stub = types.ModuleType("torch")

        class _FakeDevice:
            def __init__(self, value: str) -> None:
                normalized = value.strip().lower()
                if normalized.startswith("cuda"):
                    self.type = "cuda"
                    if ":" in normalized:
                        _, raw_index = normalized.split(":", maxsplit=1)
                        self.index = int(raw_index)
                    else:
                        self.index = None
                elif normalized == "mps":
                    self.type = "mps"
                    self.index = None
                else:
                    self.type = "cpu"
                    self.index = None

            def __str__(self) -> str:
                if self.type == "cuda" and self.index is not None:
                    return f"cuda:{self.index}"
                return self.type

        class _FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def device_count() -> int:
                return 0

        class _FakeMps:
            @staticmethod
            def is_available() -> bool:
                return False

        @contextmanager
        def _fake_inference_mode() -> Iterator[None]:
            yield

        torch_stub.device = _FakeDevice
        torch_stub.cuda = _FakeCuda()
        torch_stub.backends = types.SimpleNamespace(mps=_FakeMps())
        torch_stub.inference_mode = _fake_inference_mode
        sys.modules["torch"] = torch_stub

    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        transformers_stub = types.ModuleType("transformers")

        class _UnavailableLoader:
            @staticmethod
            def from_pretrained(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError(
                    "transformers dependency is not installed in this environment."
                )

        class _PreTrainedTokenizerBase:
            pass

        transformers_stub.AutoConfig = _UnavailableLoader
        transformers_stub.AutoModelForSeq2SeqLM = _UnavailableLoader
        transformers_stub.AutoTokenizer = _UnavailableLoader
        transformers_stub.PreTrainedTokenizerBase = _PreTrainedTokenizerBase
        sys.modules["transformers"] = transformers_stub


_install_optional_ml_stubs()

if TYPE_CHECKING:
    from app.adapters.outbound.nllb_translator import NllbRuntimeDependencies

from app.application.use_cases.translate_text import TranslateTextUseCase  # noqa: E402
from app.application.use_cases.get_health_status import GetHealthStatusUseCase  # noqa: E402
from app.bootstrap.container import AppContainer  # noqa: E402
from app.core.config import AppSettings  # noqa: E402
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
        max_input_tokens=400,
        max_new_tokens=32,
        device="cpu",
    )

    def container_factory(runtime_settings: AppSettings) -> AppContainer:
        translate_use_case = TranslateTextUseCase(
            translator=fake_translator,
            model_id=runtime_settings.model_id,
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

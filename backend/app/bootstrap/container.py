from dataclasses import dataclass
import logging

from app.adapters.outbound.nllb_translator import NllbTranslator
from app.application.use_cases.get_health_status import GetHealthStatusUseCase
from app.application.use_cases.translate_text import TranslateTextUseCase
from app.core.config import AppSettings

logger = logging.getLogger("uvicorn.error")


@dataclass(slots=True)
class AppContainer:
    """Container with initialized runtime dependencies."""

    settings: AppSettings
    get_health_status_use_case: GetHealthStatusUseCase
    translate_text_use_case: TranslateTextUseCase


def build_container(settings: AppSettings) -> AppContainer:
    """Build and wire all application dependencies."""
    logger.info("Building application container.")
    cache_dir = settings.resolve_cache_dir()
    logger.info(
        "Initializing translator adapter "
        "(model_id=%s, device=%s, max_input_tokens=%s, max_new_tokens=%s, "
        "repetition_penalty=%s, cache_dir=%s).",
        settings.model_id,
        settings.device,
        settings.max_input_tokens,
        settings.max_new_tokens,
        settings.repetition_penalty,
        cache_dir,
    )
    translator = NllbTranslator(
        model_id=settings.model_id,
        max_input_tokens=settings.max_input_tokens,
        max_new_tokens=settings.max_new_tokens,
        repetition_penalty=settings.repetition_penalty,
        device=settings.device,
        cache_dir=cache_dir,
    )
    translate_text_use_case = TranslateTextUseCase(
        translator=translator,
        model_id=settings.model_id,
    )
    get_health_status_use_case = GetHealthStatusUseCase(
        model_id=settings.model_id,
        is_loaded=translator.is_ready,
    )
    return AppContainer(
        settings=settings,
        get_health_status_use_case=get_health_status_use_case,
        translate_text_use_case=translate_text_use_case,
    )


async def cleanup_container(container: AppContainer) -> None:
    """Clean up runtime dependencies on shutdown (implement adapter teardown for production)."""
    logger.info("Cleaning up application container resources.")
    # Prototype scope: keep shutdown as no-op. In production, release model/backend resources here.
    _ = container

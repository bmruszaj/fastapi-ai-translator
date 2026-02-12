import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.adapters.inbound.http.schemas import (
    ErrorResponse,
    HealthResponse,
    LanguagesResponse,
    TranslateRequest,
    TranslateResponse,
)
from app.application.dto import TranslateCommand
from app.application.ports.errors import InputTooLongError, TranslatorPortError
from app.application.use_cases.get_health_status import GetHealthStatusUseCase
from app.application.use_cases.translate_text import TranslateTextUseCase
from app.core.dependencies import (
    get_health_status_use_case,
    get_translate_use_case,
)
from app.domain.errors import (
    DomainError,
    EmptyTextError,
    InvalidLanguagePairError,
    TextTooLongError,
    UnsupportedLanguageError,
)
from app.domain.language_rules import SUPPORTED

logger = logging.getLogger(__name__)
TRANSLATOR_UNAVAILABLE_MESSAGE = "Translation service temporarily unavailable."

router = APIRouter()


def _to_error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """Build a standardized JSON error response."""
    error_response = ErrorResponse(code=code, message=message)
    return JSONResponse(status_code=status_code, content=error_response.model_dump())


def _map_domain_error(error: DomainError) -> JSONResponse:
    """Map domain exceptions to API error responses."""
    if isinstance(error, (UnsupportedLanguageError, InvalidLanguagePairError)):
        return _to_error_response(status_code=400, code=error.code, message=str(error))
    if isinstance(error, (EmptyTextError, TextTooLongError)):
        return _to_error_response(status_code=422, code=error.code, message=str(error))
    return _to_error_response(status_code=422, code=error.code, message=str(error))


def _map_translator_error(error: TranslatorPortError) -> JSONResponse:
    """Map translation backend exceptions to API error responses."""
    if isinstance(error, InputTooLongError):
        return _to_error_response(status_code=422, code=error.code, message=str(error))
    return _to_error_response(
        status_code=503,
        code=error.code,
        message=TRANSLATOR_UNAVAILABLE_MESSAGE,
    )


@router.get("/health", response_model=HealthResponse)
def health(
    use_case: GetHealthStatusUseCase = Depends(get_health_status_use_case),
) -> HealthResponse:
    """Report service health and model metadata."""
    result = use_case.execute()
    return HealthResponse(
        status=result.status,
        model=result.model,
        loaded=result.loaded,
    )


@router.get("/languages", response_model=LanguagesResponse)
def languages() -> LanguagesResponse:
    """Return supported translation languages."""
    return LanguagesResponse(languages=SUPPORTED)


@router.post(
    "/translate",
    response_model=TranslateResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def translate(
    payload: TranslateRequest,
    use_case: TranslateTextUseCase = Depends(get_translate_use_case),
) -> TranslateResponse | JSONResponse:
    """Translate text from source language into target language."""
    command = TranslateCommand(
        text=payload.text,
        source_language=payload.source_language,
        target_language=payload.target_language,
    )
    try:
        result = use_case.execute(command=command)
    except DomainError as error:
        return _map_domain_error(error)
    except TranslatorPortError as error:
        logger.exception("Translation backend error during translation request.")
        return _map_translator_error(error)
    except Exception:
        logger.exception("Unexpected error during translation request.")
        return _to_error_response(
            status_code=500,
            code="internal_error",
            message="Internal server error",
        )

    return TranslateResponse(
        translated_text=result.translated_text,
        source_language=result.source_language,
        target_language=result.target_language,
        model=result.model,
    )

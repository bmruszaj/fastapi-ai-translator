class TranslatorPortError(Exception):
    """Base class for translation adapter runtime errors."""

    code = "translator_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)


class TranslatorUnavailableError(TranslatorPortError):
    """Raised when translation backend is unavailable."""

    code = "translator_unavailable"


class TranslationExecutionError(TranslatorPortError):
    """Raised when translation execution fails."""

    code = "translation_execution_failed"


class InputTooLongError(TranslatorPortError):
    """Raised when tokenized input exceeds configured limit."""

    code = "text_too_long"

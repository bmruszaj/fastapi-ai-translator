class DomainError(Exception):
    """Base class for domain-level translation errors."""

    code = "domain_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnsupportedLanguageError(DomainError):
    """Raised when a requested language is not supported."""

    code = "unsupported_language"

    def __init__(self, language_code: str) -> None:
        super().__init__(f"Unsupported language: {language_code}")


class InvalidLanguagePairError(DomainError):
    """Raised when source and target languages are invalid as a pair."""

    code = "invalid_language_pair"

    def __init__(self, source_language: str, target_language: str) -> None:
        super().__init__(
            f"Invalid language pair: {source_language} -> {target_language}"
        )


class EmptyTextError(DomainError):
    """Raised when the text to translate is empty."""

    code = "empty_text"

    def __init__(self) -> None:
        super().__init__("Text must not be empty.")


class TextTooLongError(DomainError):
    """Raised when input text exceeds allowed size."""

    code = "text_too_long"

    def __init__(self, max_chars: int, actual_chars: int) -> None:
        super().__init__(
            f"Text exceeds max length of {max_chars} characters (received {actual_chars})."
        )

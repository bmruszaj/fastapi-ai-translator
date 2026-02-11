from app.domain.errors import InvalidLanguagePairError, UnsupportedLanguageError


SUPPORTED: list[str] = ["de", "en", "el", "es", "fr", "it", "pl", "pt", "ro", "nl"]
SUPPORTED_SET: set[str] = set(SUPPORTED)


def validate_language(language_code: str) -> str:
    """Validate and normalize a language code."""
    normalized_code = language_code.strip().lower()
    if normalized_code not in SUPPORTED_SET:
        raise UnsupportedLanguageError(normalized_code)
    return normalized_code


def validate_pair(source_language: str, target_language: str) -> tuple[str, str]:
    """Validate source and target language pair."""
    normalized_source = validate_language(source_language)
    normalized_target = validate_language(target_language)
    if normalized_source == normalized_target:
        raise InvalidLanguagePairError(normalized_source, normalized_target)
    return normalized_source, normalized_target

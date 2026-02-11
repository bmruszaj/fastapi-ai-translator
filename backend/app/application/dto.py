from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranslateCommand:
    """Input data required to execute translation."""

    text: str
    source_language: str
    target_language: str


@dataclass(frozen=True, slots=True)
class TranslateResult:
    """Output data returned by translation use case."""

    translated_text: str
    source_language: str
    target_language: str
    model: str

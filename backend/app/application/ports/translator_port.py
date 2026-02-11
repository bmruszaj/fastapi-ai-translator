from abc import ABC, abstractmethod


class TranslatorPort(ABC):
    """Port for translation adapters."""

    @abstractmethod
    def translate(self, text: str, source: str, target: str) -> str:
        """Translate text from source language to target language.

        Args:
            text: Input text to translate.
            source: Source language code.
            target: Target language code.

        Returns:
            Translated text.

        Raises:
            TranslatorPortError: If translation backend is unavailable or execution fails.
        """

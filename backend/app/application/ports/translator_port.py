from abc import ABC, abstractmethod


class TranslatorPort(ABC):
    """Port for translation adapters."""

    @abstractmethod
    def translate(self, text: str, source: str, target: str) -> str:
        """Translate text from source language to target language."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Report whether the translator backend is ready to serve requests."""

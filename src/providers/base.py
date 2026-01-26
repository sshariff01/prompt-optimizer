"""Base interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    This interface allows swapping between different LLM providers
    (Anthropic, OpenAI, Google, etc.) without changing core logic.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt/message to send to the model
            system: Optional system message/instructions
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific additional parameters

        Returns:
            The generated text response from the model

        Raises:
            Exception: If the API call fails or returns an error
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model being used."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider (e.g., 'anthropic', 'openai')."""
        pass

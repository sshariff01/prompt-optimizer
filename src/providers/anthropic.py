"""Anthropic provider implementation."""

import os
from typing import Any

import anthropic

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Provider implementation for Anthropic's Claude models."""

    def __init__(
        self,
        model: str = "claude-opus-4.5",
        api_key: str | None = None,
    ):
        """Initialize the Anthropic provider.

        Args:
            model: The Claude model to use (e.g., 'claude-opus-4.5')
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var
        """
        self._model = model
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """Generate a response using Claude.

        Args:
            prompt: The user message to send
            system: Optional system message
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters passed to the API

        Returns:
            The generated text response

        Raises:
            anthropic.APIError: If the API call fails
        """
        messages = [{"role": "user", "content": prompt}]

        params = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            **kwargs,
        }

        if system:
            params["system"] = system

        response = self._client.messages.create(**params)

        # Extract text from response
        return response.content[0].text

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using Anthropic's tokenizer.

        Args:
            text: The text to count tokens for

        Returns:
            Approximate token count

        Note:
            Anthropic's API doesn't provide a direct tokenizer.
            This is a rough approximation (4 chars ≈ 1 token).
            For accurate counts, use the response usage data.
        """
        # Rough approximation: 4 characters per token
        # This is conservative and matches Anthropic's general guidance
        return len(text) // 4

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "anthropic"

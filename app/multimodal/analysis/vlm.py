from __future__ import annotations

from abc import ABC, abstractmethod


class VLMProvider(ABC):
    """Interface for vision-language model providers."""

    @abstractmethod
    def analyze_image(
        self,
        image_path: str,
        prompt: str,
    ) -> str:
        """Analyze an image and return textual information."""
        raise NotImplementedError
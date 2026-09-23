from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class OCRProvider(ABC):
    """
    Abstract OCR provider.

    OCR implementations convert an image into extracted text.
    Concrete providers such as PaddleOCR can be added without
    changing the PDF parser or ingestion pipeline.
    """

    @abstractmethod
    def extract_text(self, image_path: str | Path) -> str:
        """
        Extract text from an image.

        Args:
            image_path: Path to the image.

        Returns:
            Extracted text.
        """
        raise NotImplementedError
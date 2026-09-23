from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.ingestion.document import Document


class DocumentParser(ABC):
    """Base interface for all document parsers."""

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Return file extensions supported by this parser."""
        raise NotImplementedError

    @abstractmethod
    def parse(self, path: Path) -> Document:
        """Parse a document into the internal document representation."""
        raise NotImplementedError

    def supports(self, path: Path) -> bool:
        """Return True when this parser supports the given file."""
        return path.suffix.lower() in self.supported_extensions
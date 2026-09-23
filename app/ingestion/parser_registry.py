from __future__ import annotations

from pathlib import Path

from app.ingestion.document import Document
from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.mineru import MinerUParser


class ParserRegistry:
    """Resolves the correct parser for a document."""

    def __init__(self, parsers: list[DocumentParser] | None = None) -> None:
        self._parsers = parsers or [
            MinerUParser(),
        ]

    def register(self, parser: DocumentParser) -> None:
        self._parsers.append(parser)

    def get_parser(self, path: Path) -> DocumentParser:
        for parser in self._parsers:
            if parser.supports(path):
                return parser

        raise ValueError(
            f"No parser registered for file extension: {path.suffix}"
        )

    def parse(self, path: Path) -> Document:
        parser = self.get_parser(path)
        return parser.parse(path)
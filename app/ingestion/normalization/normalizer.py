from __future__ import annotations

from typing import Sequence
from uuid import uuid4

from app.ingestion.document import Document
from app.ingestion.content import ContentItem
from app.ingestion.normalization.models import (
    NormalizedContent,
    NormalizedDocument,
)


class ContentNormalizer:
    """Convert parser output into a stable downstream representation."""

    def normalize(
        self,
        document: Document | None = None,
        *,
        document_id: str | None = None,
        filename: str | None = None,
        content: Sequence[ContentItem] | None = None,
        metadata: dict | None = None,
    ) -> NormalizedDocument:
        if document is None:
            if (
                document_id is None
                or filename is None
                or content is None
            ):
                raise ValueError(
                    "Provide either a Document or document_id, "
                    "filename, and content."
                )

            document = Document(
                document_id=document_id,
                source_path=filename,
                filename=filename,
                content=list(content),
                metadata=metadata or {},
            )
        elif any(
            value is not None
            for value in (
                document_id,
                filename,
                content,
                metadata,
            )
        ):
            raise ValueError(
                "Document cannot be combined with document fields."
            )

        elements: list[NormalizedContent] = []

        for position, item in enumerate(document.content):

            element_id = str(uuid4())

            normalized = NormalizedContent(
                element_id=element_id,
                document_id=document.document_id,
                type=item.type,
                page_idx=item.page_idx,
                position=position,
                text=item.text,
                payload=self._build_payload(item),
            )

            elements.append(normalized)

        self._link_neighbors(elements)

        return NormalizedDocument(
            document_id=document.document_id,
            filename=document.filename,
            elements=elements,
            metadata=document.metadata.copy(),
        )

    def _build_payload(
        self,
        item: ContentItem,
    ) -> dict:
        """Extract type-specific information into a payload."""

        payload = {
            "caption": item.caption,
            "footnote": item.footnote,
            **item.metadata,
        }

        if item.img_path:
            payload["img_path"] = item.img_path

        if item.table_body:
            payload["table_body"] = item.table_body

        if item.latex:
            payload["latex"] = item.latex

        return payload

    def _link_neighbors(
        self,
        elements: list[NormalizedContent],
    ) -> None:
        """Create previous/next relationships between elements."""

        for index, element in enumerate(elements):

            if index > 0:
                element.previous_id = elements[index - 1].element_id

            if index < len(elements) - 1:  
                element.next_id = elements[index + 1].element_id
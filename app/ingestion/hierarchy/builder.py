from __future__ import annotations

from collections import defaultdict

from app.ingestion.hierarchy.models import (
    DocumentHierarchy,
    PageNode,
)
from app.ingestion.normalization.models import NormalizedDocument


class HierarchyBuilder:
    """Build document/page hierarchy from normalized content."""

    def build(
        self,
        document: NormalizedDocument,
    ) -> DocumentHierarchy:

        pages: dict[int, PageNode] = {}

        for element in document.elements:

            if element.page_idx is None:
                continue

            if element.page_idx not in pages:
                pages[element.page_idx] = PageNode(
                    page_id=f"{document.document_id}:page:{element.page_idx}",
                    document_id=document.document_id,
                    page_idx=element.page_idx,
                )

            pages[element.page_idx].element_ids.append(
                element.element_id
            )

        ordered_pages = [
            pages[index]
            for index in sorted(pages)
        ]

        return DocumentHierarchy(
            document_id=document.document_id,
            page_nodes=ordered_pages,
            metadata=document.metadata.copy(),
        )
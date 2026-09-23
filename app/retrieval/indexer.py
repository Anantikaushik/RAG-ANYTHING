from __future__ import annotations

from app.ingestion.normalization.models import NormalizedDocument
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.vector_store import InMemoryVectorStore


class DocumentVectorIndexer:
    """Indexes normalized document content for vector retrieval."""

    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def index(self, document: NormalizedDocument) -> int:
        indexed_count = 0

        for element in document.elements:
            text = self._extract_text(element)

            if not text:
                continue

            vector = self.embedding_provider.embed(text)

            metadata = {
                "document_id": document.document_id,
                "filename": document.filename,
                "element_id": element.element_id,
                "content_type": element.type,
                "page_idx": element.page_idx,
                **element.payload,
            }

            self.vector_store.add(
                item_id=element.element_id,
                vector=vector,
                text=text,
                metadata=metadata,
            )

            indexed_count += 1

        return indexed_count

    @staticmethod
    def _extract_text(element) -> str | None:
        if element.text and element.text.strip():
            return element.text.strip()

        payload = element.payload or {}

        for key in ("table_body", "latex", "extracted_text"):
            value = payload.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        return None
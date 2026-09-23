from app.ingestion.normalization.models import (
    NormalizedContent,
    NormalizedDocument,
)
from app.retrieval.embeddings import (
    DeterministicEmbeddingProvider,
)
from app.retrieval.indexer import DocumentVectorIndexer
from app.retrieval.vector_store import InMemoryVectorStore


def create_document():
    return NormalizedDocument(
        document_id="doc-1",
        filename="sample.pdf",
        elements=[
            NormalizedContent(
                element_id="element-1",
                document_id="doc-1",
                type="text",
                page_idx=0,
                position=0,
                text="Python programming language",
                payload={},
            ),
            NormalizedContent(
                element_id="element-2",
                document_id="doc-1",
                type="table",
                page_idx=1,
                position=1,
                text=None,
                payload={
                    "table_body": "| Model | Score |\n|---|---|\n| A | 95 |"
                },
            ),
            NormalizedContent(
                element_id="element-3",
                document_id="doc-1",
                type="image",
                page_idx=2,
                position=2,
                text=None,
                payload={
                    "img_path": "image.png",
                },
            ),
        ],
    )


def create_indexer():
    store = InMemoryVectorStore()
    provider = DeterministicEmbeddingProvider(dimensions=32)

    return DocumentVectorIndexer(
        vector_store=store,
        embedding_provider=provider,
    ), store


def test_indexes_text_content():
    indexer, store = create_indexer()

    count = indexer.index(create_document())

    assert count == 2
    assert store.get("element-1") is not None


def test_indexes_table_content():
    indexer, store = create_indexer()

    indexer.index(create_document())

    record = store.get("element-2")

    assert record is not None
    assert "Model" in record.text
    assert record.metadata["content_type"] == "table"


def test_skips_content_without_text():
    indexer, store = create_indexer()

    indexer.index(create_document())

    assert store.get("element-3") is None


def test_index_is_idempotent():
    indexer, store = create_indexer()
    document = create_document()

    first_count = indexer.index(document)
    second_count = indexer.index(document)

    assert first_count == 2
    assert second_count == 2
    assert len(store) == 2
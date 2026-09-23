from pathlib import Path

from dotenv import load_dotenv

from app.ingestion.normalization.models import (
    NormalizedContent,
    NormalizedDocument,
)
from app.multimodal.analysis.vlm_processor import (
    create_vlm_analysis_processor,
)
from app.retrieval.embeddings import DeterministicEmbeddingProvider
from app.retrieval.indexer import DocumentVectorIndexer
from app.retrieval.vector_store import InMemoryVectorStore


def main() -> None:
    load_dotenv()

    image_path = Path("data/uploads/test_image.png")

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    document = NormalizedDocument(
        document_id="vlm-vector-test",
        filename=image_path.name,
        elements=[
            NormalizedContent(
                element_id="vlm-image-001",
                document_id="vlm-vector-test",
                type="image",
                page_idx=0,
                position=0,
                text=None,
                payload={
                    "img_path": str(image_path),
                },
                parent_id=None,
                previous_id=None,
                next_id=None,
            )
        ],
        metadata={},
    )

    # 1. Run VLM analysis.
    processor = create_vlm_analysis_processor()

    print("\n--- VLM ANALYSIS ---")

    processor.materialize(document)

    element = document.elements[0]

    print("Generated text:")
    print(element.text)

    # 2. Index the generated text.
    vector_store = InMemoryVectorStore()
    embedding_provider = DeterministicEmbeddingProvider()

    indexer = DocumentVectorIndexer(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    indexed_count = indexer.index(document)

    print("\n--- VECTOR INDEX ---")
    print(f"Indexed records: {indexed_count}")

    record = vector_store.get(element.element_id)

    if record is None:
        raise RuntimeError(
            f"Indexed record not found: {element.element_id}"
        )

    for record in [record]:
        print(f"\nItem ID: {record.item_id}")
        print(f"Text: {record.text}")
        print(f"Metadata: {record.metadata}")

    print("\n--- SUCCESS ---")
    print(
        "Image -> VLM -> normalized text -> embedding -> vector store"
    )


if __name__ == "__main__":
    main()
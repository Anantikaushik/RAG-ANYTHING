from pathlib import Path

from dotenv import load_dotenv

from app.ingestion.pipeline import IngestionPipeline
from app.multimodal.analysis.vlm_processor import (
    create_vlm_analysis_processor,
)


def main() -> None:
    load_dotenv()

    image_path = Path("data/uploads/test_image.png")

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Use the VLM-enabled analysis processor.
    vlm_processor = create_vlm_analysis_processor()

    pipeline = IngestionPipeline(
        multimodal_analysis_processor=vlm_processor,
    )

    print("\n--- MULTIMODAL INGESTION ---")
    print(f"Image: {image_path}")

    # NOTE:
    # The current pipeline's parser registry does not yet treat
    # standalone PNG files as documents. Therefore this script
    # verifies the VLM-enabled analysis layer through a normalized
    # document rather than pretending PNG ingestion is already
    # implemented.
    from app.ingestion.normalization.models import NormalizedDocument, NormalizedContent

    document = NormalizedDocument(
        document_id="multimodal-test-document",
        filename=image_path.name,
        elements=[
            NormalizedContent(
                element_id="multimodal-test-image",
                document_id="multimodal-test-document",
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

    print("\nRunning VLM analysis...")

    analysis = vlm_processor.materialize(document)

    print("\n--- VLM RESULT ---")

    for result in analysis:
        print(f"Content ID: {result.content_id}")
        print(f"Content Type: {result.content_type}")
        print(f"\nSummary:\n{result.summary}")

        print("\nExtracted text:")
        print(result.extracted_text)

    print("\n--- MATERIALIZED CONTENT ---")

    element = document.elements[0]

    print("Element text:")
    print(element.text)

    print("\nPayload:")
    print(element.payload)

    print("\n--- SUCCESS ---")
    print("Image -> VLM -> MultimodalAnalysis -> NormalizedDocument")


if __name__ == "__main__":
    main()
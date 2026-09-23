from pathlib import Path

from dotenv import load_dotenv

from app.ingestion.normalization.models import NormalizedContent
from app.multimodal.analysis.vlm_processor import (
    create_vlm_analysis_processor,
)


def main() -> None:
    load_dotenv()

    image_path = Path("data/uploads/test_image.png")

    if not image_path.exists():
        raise FileNotFoundError(
            f"Test image not found: {image_path}"
        )

    # Create the normalized image content expected
    # by the multimodal analysis layer.
    content = NormalizedContent(
        element_id="test-image-001",
        document_id="test-document",
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

    processor = create_vlm_analysis_processor()

    print("\n--- RUNNING VLM PROCESSOR ---")

    analysis = processor.analyze(content)

    print("\n--- ANALYSIS RESULT ---")
    print(f"Content ID: {analysis.content_id}")
    print(f"Content Type: {analysis.content_type}")
    print(f"\nSummary:\n{analysis.summary}")

    print("\n--- EXTRACTED TEXT ---")
    print(analysis.extracted_text)

    print("\n--- METADATA ---")
    print(analysis.metadata)


if __name__ == "__main__":
    main()
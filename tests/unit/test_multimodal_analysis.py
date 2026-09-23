from app.ingestion.normalization.models import (
    NormalizedContent,
    NormalizedDocument,
)
from app.multimodal.analysis.processor import (
    MultimodalAnalysisProcessor,
)


def test_image_analysis():
    content = NormalizedContent(
        element_id="image-001",
        document_id="doc-001",
        type="image",
        page_idx=0,
        position=0,
        payload={
            "img_path": "image.png",
        },
    )

    processor = MultimodalAnalysisProcessor()

    result = processor.analyze(content)

    assert result.content_id == "image-001"
    assert result.content_type == "image"
    assert result.summary == "Image content"


def test_table_analysis():
    content = NormalizedContent(
        element_id="table-001",
        document_id="doc-001",
        type="table",
        page_idx=1,
        position=1,
        payload={
            "table_body": "| Name | Value |",
        },
    )

    processor = MultimodalAnalysisProcessor()

    result = processor.analyze(content)

    assert result.content_type == "table"
    assert result.extracted_text == "| Name | Value |"


def test_equation_analysis():
    content = NormalizedContent(
        element_id="equation-001",
        document_id="doc-001",
        type="equation",
        page_idx=2,
        position=2,
        payload={
            "latex": "E = mc^2",
        },
    )

    processor = MultimodalAnalysisProcessor()

    result = processor.analyze(content)

    assert result.content_type == "equation"
    assert result.extracted_text == "E = mc^2"


def test_document_analysis_skips_text():
    document = NormalizedDocument(
        document_id="doc-001",
        filename="test.pdf",
        elements=[
            NormalizedContent(
                element_id="text-001",
                document_id="doc-001",
                type="text",
                page_idx=0,
                position=0,
                text="Hello",
            ),
            NormalizedContent(
                element_id="table-001",
                document_id="doc-001",
                type="table",
                page_idx=1,
                position=1,
                payload={
                    "table_body": "| A | B |",
                },
            ),
        ],
    )

    processor = MultimodalAnalysisProcessor()

    results = processor.analyze_document(document)

    assert len(results) == 1
    assert results[0].content_type == "table"
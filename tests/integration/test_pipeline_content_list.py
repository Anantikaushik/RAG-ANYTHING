from app.ingestion.content import ContentItem
from app.ingestion.content_list import ContentList
from app.ingestion.pipeline import IngestionPipeline


def test_ingest_content_list_end_to_end():
    content = ContentList(
        [
            ContentItem(
                type="text",
                text="RAG-Anything is a multimodal retrieval system.",
                page_idx=0,
            ),
            ContentItem(
                type="table",
                table_body="| Metric | Value |\n|---|---|\n| Accuracy | 95% |",
                page_idx=1,
            ),
            ContentItem(
                type="equation",
                latex="E = mc^2",
                page_idx=2,
            ),
        ]
    )

    pipeline = IngestionPipeline()

    result = pipeline.ingest_content_list(
        content,
        filename="demo-content-list",
        document_id="demo-document",
    )

    assert result.document.document_id == "demo-document"
    assert result.document.filename == "demo-content-list"

    assert len(result.document.content) == 3
    assert len(result.normalized_document.elements) == 3

    assert result.normalized_document.elements[0].type == "text"
    assert result.normalized_document.elements[1].type == "table"
    assert result.normalized_document.elements[2].type == "equation"

    assert result.graph is not None
    assert result.vector_store is not None


def test_ingest_content_list_with_image():
    content = ContentList(
        [
            ContentItem(
                type="text",
                text="The following image shows the system architecture.",
                page_idx=0,
            ),
            ContentItem(
                type="image",
                img_path="data/uploads/test_image.png",
                image_caption=["System architecture diagram"],
                page_idx=1,
            ),
        ]
    )

    pipeline = IngestionPipeline()

    result = pipeline.ingest_content_list(
        content,
        filename="image-demo",
        document_id="image-demo-document",
    )

    elements = result.normalized_document.elements

    assert len(elements) == 2
    assert elements[0].type == "text"
    assert elements[1].type == "image"

    image = elements[1]

    assert image.payload["img_path"] == "data/uploads/test_image.png"
    assert "analysis" in image.payload
    assert image.payload["analysis"]["content_type"] == "image"    
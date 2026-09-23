from app.ingestion.content import ContentItem
from app.ingestion.content_list import ContentList
from app.ingestion.normalization.normalizer import ContentNormalizer


def test_content_list_can_be_normalized():
    content_list = ContentList(
        [
            ContentItem(
                type="text",
                text="RAG-Anything processes multimodal documents.",
                page_idx=0,
            ),
            ContentItem(
                type="image",
                img_path="data/image.png",
                image_caption=["System architecture"],
                page_idx=1,
            ),
            ContentItem(
                type="table",
                table_body="| Metric | Value |\n|---|---|\n| Accuracy | 95% |",
                page_idx=2,
            ),
            ContentItem(
                type="equation",
                latex="E = mc^2",
                page_idx=3,
            ),
        ]
    )

    document_id = "content-list-document"

    normalized = ContentNormalizer().normalize(
        document_id=document_id,
        filename="multimodal.md",
        content=content_list.items,
    )

    assert normalized.document_id == document_id
    assert len(normalized.elements) == 4

    assert normalized.elements[0].type == "text"
    assert normalized.elements[1].type == "image"
    assert normalized.elements[2].type == "table"
    assert normalized.elements[3].type == "equation"

    assert normalized.elements[0].next_id == normalized.elements[1].element_id
    assert normalized.elements[1].previous_id == normalized.elements[0].element_id
from app.ingestion.content import ContentItem
from app.ingestion.content_list import ContentList


def test_content_list_round_trip():
    items = ContentList(
        [
            ContentItem(
                type="text",
                text="This is a document.",
                page_idx=0,
            ),
            ContentItem(
                type="image",
                img_path="data/image.png",
                image_caption=["Architecture diagram"],
                page_idx=1,
            ),
            ContentItem(
                type="table",
                table_body="| Name | Value |\n|---|---|\n| A | 10 |",
                page_idx=2,
            ),
            ContentItem(
                type="equation",
                latex="E = mc^2",
                page_idx=3,
            ),
        ]
    )

    serialized = items.to_list()

    restored = ContentList.from_list(serialized)

    assert len(restored) == 4
    assert restored.text_items[0].text == "This is a document."
    assert restored.multimodal_items[0].type == "image"
    assert restored.multimodal_items[1].type == "table"
    assert restored.multimodal_items[2].type == "equation"


def test_content_list_filters_by_type():
    content = ContentList(
        [
            ContentItem(type="text", text="text"),
            ContentItem(type="image", img_path="image.png"),
            ContentItem(type="image", img_path="image2.png"),
        ]
    )

    images = content.by_type("image")

    assert len(images) == 2
    assert all(item.type == "image" for item in images)
import pytest

from app.query.context import ContextAssembler, QueryContext
from app.retrieval.models import RetrievalResult


def make_result(
    item_id="item-1",
    text="Python is a programming language.",
    score=0.9,
    content_type="text",
    source="hybrid",
):
    return RetrievalResult(
        item_id=item_id,
        score=score,
        source=source,
        content_type=content_type,
        text=text,
        metadata={"page_idx": 1},
    )


def test_assemble_context():
    assembler = ContextAssembler()

    context = assembler.assemble(
        query="What is Python?",
        results=[make_result()],
    )

    assert isinstance(context, QueryContext)
    assert context.query == "What is Python?"
    assert context.count == 1
    assert context.items[0].text.startswith("Python")


def test_context_preserves_metadata():
    assembler = ContextAssembler()

    context = assembler.assemble(
        query="Python",
        results=[make_result()],
    )

    assert context.items[0].metadata["page_idx"] == 1


def test_empty_text_results_are_skipped():
    assembler = ContextAssembler()

    results = [
        make_result(item_id="valid", text="Python"),
        make_result(item_id="empty", text=None),
    ]

    context = assembler.assemble(
        query="Python",
        results=results,
    )

    assert context.count == 1
    assert context.items[0].item_id == "valid"


def test_as_text_contains_context():
    assembler = ContextAssembler()

    context = assembler.assemble(
        query="Python",
        results=[make_result()],
    )

    output = context.as_text()

    assert "[Context 1]" in output
    assert "Python is a programming language." in output
    assert "hybrid" in output


def test_as_text_can_bound_prompt_size():
    assembler = ContextAssembler()
    context = assembler.assemble(
        query="Python",
        results=[
            make_result(item_id="first", text="A" * 40),
            make_result(item_id="second", text="B" * 40),
        ],
    )

    output = context.as_text(max_chars=150)

    assert "[Context 1]" in output
    assert "[Context 2]" not in output
    assert len(output) <= 150


def test_empty_query_is_rejected():
    assembler = ContextAssembler()

    with pytest.raises(ValueError):
        assembler.assemble(
            query="",
            results=[],
        )
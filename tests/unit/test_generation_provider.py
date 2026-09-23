import pytest

from app.query.context import ContextAssembler
from app.query.providers import (
    DeterministicGenerationProvider,
)


def make_context():
    assembler = ContextAssembler()

    return assembler.assemble(
        query="What is Python?",
        results=[
            {
                "item_id": "item-1",
                "score": 0.95,
                "source": "hybrid",
                "content_type": "text",
                "text": "Python is a programming language.",
                "metadata": {},
            }
        ],
    )


def test_generation_returns_answer():
    provider = DeterministicGenerationProvider()

    context = make_context()

    answer = provider.generate(
        query="What is Python?",
        context=context,
    )

    assert isinstance(answer, str)
    assert "Python" in answer


def test_generation_handles_empty_context():
    provider = DeterministicGenerationProvider()

    assembler = ContextAssembler()
    context = assembler.assemble(
        query="Unknown",
        results=[],
    )

    answer = provider.generate(
        query="Unknown",
        context=context,
    )

    assert answer == "No relevant context was found."


def test_empty_query_is_rejected():
    provider = DeterministicGenerationProvider()

    context = make_context()

    with pytest.raises(ValueError):
        provider.generate(
            query="",
            context=context,
        )
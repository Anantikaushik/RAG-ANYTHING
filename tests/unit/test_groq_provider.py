from unittest.mock import MagicMock, patch

import pytest

from app.query.context import ContextAssembler
from app.query.providers.groq import GroqGenerationProvider


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


@patch("app.query.providers.groq.Groq")
def test_groq_generation(mock_groq):
    mock_response = MagicMock()

    mock_response.choices[0].message.content = (
        "Python is a programming language."
    )

    mock_groq.return_value.chat.completions.create.return_value = (
        mock_response
    )

    provider = GroqGenerationProvider(
        api_key="test-key",
    )

    answer = provider.generate(
        query="What is Python?",
        context=make_context(),
    )

    assert answer == "Python is a programming language."

    mock_groq.return_value.chat.completions.create.assert_called_once()


def test_missing_api_key_is_rejected():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            GroqGenerationProvider(
                api_key=None,
            )


@patch("app.query.providers.groq.Groq")
def test_empty_response_is_rejected(mock_groq):
    mock_response = MagicMock()

    mock_response.choices[0].message.content = None

    mock_groq.return_value.chat.completions.create.return_value = (
        mock_response
    )

    provider = GroqGenerationProvider(
        api_key="test-key",
    )

    with pytest.raises(RuntimeError):
        provider.generate(
            query="What is Python?",
            context=make_context(),
        )
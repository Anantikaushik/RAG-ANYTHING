from __future__ import annotations

import os

from groq import Groq

from app.query.context import QueryContext
from app.query.providers.base import GenerationProvider


class GroqGenerationProvider(GenerationProvider):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required.")

        self.model = model or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        )
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

        self.client = Groq(api_key=self.api_key)

        # Usage from the most recent request.
        self.last_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def generate(
        self,
        query: str,
        context: QueryContext,
    ) -> str:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        prompt = self._build_prompt(query, context)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document question-answering assistant. "
                        "Answer using only the provided context. "
                        "If the context does not contain enough information, "
                        "say so clearly. Cite supporting context inline using "
                        "the matching [Context N] number, for example [1]. "
                        "Do not invent citation numbers. Keep answers brief: "
                        "use at most 3 short sentences or 4 bullet points."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
        )

        # Capture token usage returned by Groq.
        usage = getattr(response, "usage", None)

        if usage is not None:
            self.last_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(
                    usage, "completion_tokens", 0
                ) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("Groq returned an empty response.")

        return content.strip()

    @staticmethod
    def _build_prompt(
        query: str,
        context: QueryContext,
    ) -> str:
        return (
            f"Question:\n{query}\n\n"
            f"Retrieved Context:\n{context.as_text(max_chars=24000)}\n\n"
            "Answer the question directly in at most 3 short sentences or "
            "4 concise bullet points. Use only the relevant retrieved "
            "context, not every context item. "
            "Use inline citations such as [1] when a context item supports "
            "a statement."
        )
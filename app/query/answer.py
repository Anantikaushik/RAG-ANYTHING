from __future__ import annotations

from pydantic import BaseModel, Field

from app.query.context import ContextAssembler, QueryContext
from app.query.providers.base import GenerationProvider


class SourceCitation(BaseModel):
    citation_id: int
    item_id: str
    document_id: str | None = None
    filename: str | None = None
    page_idx: int | None = None
    content_type: str | None = None
    label: str


class AnswerResponse(BaseModel):
    query: str
    answer: str
    context: QueryContext
    sources: list[SourceCitation] = Field(default_factory=list)


class AnswerService:
    def __init__(
        self,
        retriever,
        generation_provider: GenerationProvider,
        context_assembler: ContextAssembler | None = None,
    ) -> None:
        self.retriever = retriever
        self.generation_provider = generation_provider
        self.context_assembler = context_assembler or ContextAssembler()

    def answer(
        self,
        query: str,
        top_k: int = 5,
    ) -> AnswerResponse:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        retrieval_response = self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        context = self.context_assembler.assemble(
            query=query,
            results=retrieval_response.results,
        )

        answer = self.generation_provider.generate(
            query=query,
            context=context,
        )

        sources = self._build_citations(
            retrieval_response.results
        )

        return AnswerResponse(
            query=query,
            answer=answer,
            context=context,
            sources=sources,
        )

    @staticmethod
    def _build_citations(results) -> list[SourceCitation]:
        citations: list[SourceCitation] = []

        for index, result in enumerate(results, start=1):
            metadata = result.metadata or {}

            filename = metadata.get("filename")
            document_id = metadata.get("document_id")
            page_idx = metadata.get("page_idx")
            content_type = result.content_type

            label_parts: list[str] = []

            if filename:
                label_parts.append(str(filename))

            if page_idx is not None:
                label_parts.append(f"Page {int(page_idx) + 1}")

            if content_type and content_type != "text":
                label_parts.append(f"Type: {content_type}")

            if not label_parts:
                label_parts.append(result.item_id)

            citations.append(
                SourceCitation(
                    citation_id=index,
                    item_id=result.item_id,
                    document_id=document_id,
                    filename=filename,
                    page_idx=page_idx,
                    content_type=content_type,
                    label=" — ".join(label_parts),
                )
            )

        return citations
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


ContentType = Literal[
    "text",
    "image",
    "table",
    "equation",
    "generic",
]


class ContentItem(BaseModel):
    """
    Canonical representation of one piece of document content.

    This is the internal contract shared by:
        parser -> multimodal processing -> indexing -> retrieval
    """

    type: ContentType

    page_idx: int | None = Field(
        default=None,
        ge=0,
    )

    text: str | None = None

    img_path: str | None = None

    table_body: str | None = None

    latex: str | None = None

    caption: list[str] = Field(
        default_factory=list,
    )

    footnote: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_content(self) -> "ContentItem":
        if self.type == "text":
            if not self.text or not self.text.strip():
                raise ValueError(
                    "Text content requires non-empty 'text'."
                )

        elif self.type == "image":
            if not self.img_path:
                raise ValueError(
                    "Image content requires 'img_path'."
                )

        elif self.type == "table":
            if not self.table_body:
                raise ValueError(
                    "Table content requires 'table_body'."
                )

        elif self.type == "equation":
            if not self.latex:
                raise ValueError(
                    "Equation content requires 'latex'."
                )
        return self

    @property
    def path(self) -> str | None:
     """Compatibility alias for image path."""
     return self.img_path

    @property
    def body(self) -> str | None:
      """Compatibility alias for table body."""
      return self.table_body
    
    
    def __getitem__(self, key: str):
        """Provide dictionary-style access for compatibility."""
        return getattr(self, key)

    @property
    def is_text(self) -> bool:
        return self.type == "text"

    @property
    def is_multimodal(self) -> bool:
        return self.type in {
            "image",
            "table",
            "equation",
            "generic",
        }
from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from groq import Groq

from app.multimodal.analysis.vlm import VLMProvider


class GroqVLMProvider(VLMProvider):
    """
    Groq-based Vision Language Model provider.

    Sends an image together with a text prompt and returns only the
    final model response, excluding reasoning traces such as <think>.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.model = model or os.getenv(
            "GROQ_VLM_MODEL",
            "qwen/qwen3.6-27b",
        )
        self.client = Groq(api_key=self.api_key)

    def analyze_image(
        self,
        image_path: str,
        prompt: str,
    ) -> str:
        """
        Analyze an image using the configured Groq VLM.
        """

        if not image_path:
            raise ValueError(
                "image_path is required."
            )

        if not prompt or not prompt.strip():
            raise ValueError(
                "prompt is required."
            )

        path = Path(image_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Image file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Image path is not a file: {path}"
            )

        # Detect MIME type.
        mime_type, _ = mimetypes.guess_type(path.name)

        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"

        # Encode image as base64.
        image_bytes = path.read_bytes()
        encoded_image = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        image_url = (
            f"data:{mime_type};base64,{encoded_image}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            temperature=0.0,
            max_completion_tokens=512,
        )

        content = (
            response.choices[0].message.content
            or ""
        )

        # Remove reasoning traces returned by some models.
        if "</think>" in content:
            content = content.split(
                "</think>",
                1,
            )[1]

        return content.strip()
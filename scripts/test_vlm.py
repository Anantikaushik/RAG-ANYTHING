from pathlib import Path
import struct
import zlib

from dotenv import load_dotenv

from app.multimodal.analysis import GroqVLMProvider


def _create_test_png() -> bytes:
    width = 64
    height = 64
    pixels = b"".join(
        b"\x00" + b"\xff\xff\xff\xff" * width
        for _ in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0),
        )
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    load_dotenv()

    image_path = Path("data/uploads/test_image.png")

    if not image_path.exists():
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(_create_test_png())
        print(f"Created test image at: {image_path}")

    provider = GroqVLMProvider()

    result = provider.analyze_image(
        image_path=str(image_path),
        prompt=(
            "Analyze this image carefully. "
            "Describe the important visual information, "
            "including text, objects, tables, diagrams, "
            "and relationships that could be useful for "
            "document question answering."
        ),
    )

    print("\n--- VLM ANALYSIS ---")
    print(result)


if __name__ == "__main__":
    main()

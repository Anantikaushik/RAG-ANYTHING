from dotenv import load_dotenv

from app.query.context import ContextAssembler
from app.query.providers.groq import GroqGenerationProvider


def main():
    load_dotenv()

    assembler = ContextAssembler()

    context = assembler.assemble(
        query="What is Python?",
        results=[
            {
                "item_id": "demo-1",
                "score": 1.0,
                "source": "demo",
                "content_type": "text",
                "text": (
                    "Python is a high-level programming language "
                    "used for general-purpose software development."
                ),
                "metadata": {"page_idx": 0},
            }
        ],
    )

    provider = GroqGenerationProvider()

    answer = provider.generate(
        query="What is Python?",
        context=context,
    )

    print("\n--- GROQ ANSWER ---")
    print(answer)

    print("\n--- TOKEN USAGE ---")
    print(f"Model: {provider.model}")
    print(f"Input tokens: {provider.last_usage['prompt_tokens']}")
    print(f"Output tokens: {provider.last_usage['completion_tokens']}")
    print(f"Total tokens: {provider.last_usage['total_tokens']}")


if __name__ == "__main__":
    main()
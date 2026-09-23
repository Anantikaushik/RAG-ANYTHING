from __future__ import annotations

import hashlib


class EntityIdentity:
    """Generate deterministic IDs for canonical graph entities."""

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize an entity name for identity comparison."""
        return " ".join(name.strip().lower().split())

    @classmethod
    def generate_id(
        cls,
        name: str,
        entity_type: str,
    ) -> str:
        """Generate a stable ID for an entity."""
        normalized_name = cls.normalize_name(name)
        normalized_type = entity_type.strip().lower()

        canonical_key = (
            f"{normalized_type}:{normalized_name}"
        )

        digest = hashlib.sha256(
            canonical_key.encode("utf-8")
        ).hexdigest()[:16]

        return f"entity:{digest}"
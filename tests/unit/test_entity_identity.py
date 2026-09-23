from app.knowledge_graph.extraction.identity import EntityIdentity


def test_same_entity_generates_same_id():
    first = EntityIdentity.generate_id(
        "Microsoft",
        "ORGANIZATION",
    )

    second = EntityIdentity.generate_id(
        "Microsoft",
        "ORGANIZATION",
    )

    assert first == second


def test_entity_identity_is_case_insensitive():
    first = EntityIdentity.generate_id(
        "Microsoft",
        "ORGANIZATION",
    )

    second = EntityIdentity.generate_id(
        " microsoft ",
        "organization",
    )

    assert first == second


def test_different_entity_names_generate_different_ids():
    microsoft = EntityIdentity.generate_id(
        "Microsoft",
        "ORGANIZATION",
    )

    azure = EntityIdentity.generate_id(
        "Azure",
        "ORGANIZATION",
    )

    assert microsoft != azure


def test_different_entity_types_generate_different_ids():
    organization = EntityIdentity.generate_id(
        "Apple",
        "ORGANIZATION",
    )

    product = EntityIdentity.generate_id(
        "Apple",
        "PRODUCT",
    )

    assert organization != product
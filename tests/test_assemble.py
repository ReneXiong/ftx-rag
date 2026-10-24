from ftx_rag.assemble import MAX_NAME_CHARS, merge, to_custom_kg


def extraction(**overrides):
    base = {
        "content_hash": "abc123",
        "doc_id": "Docket_342#0",
        "text": "Sullivan & Cromwell LLP represents the Debtors.",
        "entities": [
            {"entity_type": "LAW_FIRM", "entity_name": "Sullivan & Cromwell LLP",
             "description": "counsel"},
            {"entity_type": "LEGAL_PARTY", "entity_name": "Debtors", "description": None},
        ],
        "relationships": [
            {"src_id": "Sullivan & Cromwell LLP", "tgt_id": "Debtors",
             "description": "represents"},
        ],
    }
    return {**base, **overrides}


def test_entities_and_relationships_link_back_to_chunk():
    kg = to_custom_kg(extraction())
    assert kg["chunks"] == [{"content": extraction()["text"], "source_id": "abc123",
                             "file_path": "Docket_342#0"}]
    assert {e["source_id"] for e in kg["entities"]} == {"abc123"}
    assert kg["relationships"][0]["file_path"] == "Docket_342#0"
    assert kg["entities"][1]["description"] == "No description provided"


def test_malformed_model_output_is_dropped_not_fatal():
    kg = to_custom_kg(extraction(entities="none found", relationships=None))
    assert kg["entities"] == [] and kg["relationships"] == []

    kg = to_custom_kg(extraction(
        entities=[None, "x", {"entity_name": ""}, {"entity_name": 42}],
        relationships=[{"src_id": "Debtors", "tgt_id": ""}, ["a", "b"]],
    ))
    assert kg["entities"] == [] and kg["relationships"] == []


def test_overlong_names_are_dropped_with_their_edges():
    long_name = "Creditor " * (MAX_NAME_CHARS // 4)
    kg = to_custom_kg(extraction(
        entities=[{"entity_name": long_name, "entity_type": "CREDITOR"}],
        relationships=[{"src_id": long_name, "tgt_id": "Debtors"}],
    ))
    assert kg["entities"] == [] and kg["relationships"] == []


def test_merge_concatenates_batches():
    merged = merge([to_custom_kg(extraction()),
                    to_custom_kg(extraction(content_hash="def456"))])
    assert len(merged["chunks"]) == 2
    assert len(merged["entities"]) == 4

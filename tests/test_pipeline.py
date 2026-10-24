import numpy as np
import pytest

from ftx_rag.extract import chunk_text, value_score
from ftx_rag.kb import mrl_normalize
from ftx_rag.llm import parse_json_reply
from ftx_rag.process import doc_slug
from ftx_rag.rules import EMPTY_TITLE_RULE, is_bulk, match_bulk_rule


# ── rules ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("title, expected", [
    ("Transfer/Assignment of Claim. Transferors: (Redacted)", ("Claim Transfers", "Transfer")),
    ("Certificate of Service re: Docket 123", ("Administrative", "Certificate")),
    ("  Receipt of filing fee  for Motion", ("Administrative", "Fee Receipt")),
    ("", EMPTY_TITLE_RULE),
    ("   ", EMPTY_TITLE_RULE),
    ("Objection to Motion of the Debtors", None),
    ("Order Sustaining Debtors' Omnibus Objection", None),
])
def test_bulk_rules(title, expected):
    assert match_bulk_rule(title) == expected
    assert is_bulk(title) == (expected is not None)


# ── extraction triage ──────────────────────────────────────────────────
PROSE = (
    "WHEREAS the Debtors filed a MOTION pursuant to section 363 of the Bankruptcy "
    "Code, and the Court having found that notice of the proceeding was adequate, "
    "it is hereby ORDERED that the objection of the creditor is overruled and the "
    "obligation of the Trustee under the foregoing agreement is approved."
)
TABLE = " ".join(f"Creditor {i} 1,234.56 $ 7.89 10.00 22.11 33.44" for i in range(20))


def test_value_score_prefers_prose_over_tables():
    assert value_score(PROSE) >= 0.5
    assert value_score(TABLE) < 0
    assert value_score("too short") == 0.0


def test_chunk_text_caps_document_head():
    chunks = chunk_text("x" * 2500, size=1000, cap=0)
    assert [len(c) for c in chunks] == [1000, 1000, 500]
    assert sum(map(len, chunk_text("x" * 5000, size=1000, cap=1500))) == 1500


# ── LLM reply parsing ──────────────────────────────────────────────────
def test_parse_json_reply_strips_code_fence():
    assert parse_json_reply('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_reply('{"a": 1}') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_reply("[1, 2]")


# ── embeddings ─────────────────────────────────────────────────────────
def test_mrl_truncates_before_normalizing():
    raw = np.array([[1.0, 2.0, 3.0, 0.0, 5.0, 6.0],
                    [0.0, 1.0, 0.0, 2.0, 8.0, 0.0]], dtype=np.float32)
    out = mrl_normalize(raw, 4)
    assert out.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    # The wrong order (normalize, then cut) does not give unit vectors.
    wrong = (raw / np.linalg.norm(raw, axis=1, keepdims=True))[:, :4]
    assert np.all(np.linalg.norm(wrong, axis=1) < 1.0)


def test_doc_slug_is_filesystem_safe():
    assert doc_slug("Docket # 342") == "Docket_342"
    assert doc_slug("Docket # 342", 2) == "Docket_342_att2"
    assert doc_slug("") == "unknown"

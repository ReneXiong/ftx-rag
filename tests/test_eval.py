from ftx_rag.evaluation.build_questions import build_idf, leak_score, stratify
from ftx_rag.evaluation.run_eval import docket_of, evaluate


def test_docket_of_ignores_attachment_and_chunk():
    assert docket_of("Docket_20467_att2#61") == "20467"
    assert docket_of("Docket # 25_att_1#1") == "25"
    assert docket_of("") == ""


def test_evaluate_scores_rank_and_skips_unindexed():
    questions = [
        {"question": "q1", "gold_doc_id": "Docket_1#0", "gold_content_hash": "h1"},
        {"question": "q2", "gold_doc_id": "Docket_2#3", "gold_content_hash": "h2"},
        {"question": "q3", "gold_doc_id": "Docket_3#0", "gold_content_hash": "h3"},
        {"question": "q4", "gold_doc_id": "Docket_4#0", "gold_content_hash": "missing"},
    ]
    ranked = {"q1": ["1", "9"], "q2": ["9", "8", "2"], "q3": ["9"]}
    r = evaluate(questions, lambda q: ranked[q], indexed_hashes={"h1", "h2", "h3"})
    assert r["answerable"] == 3
    assert r["recall"][1] == 1 / 3
    assert r["recall"][5] == 2 / 3
    assert abs(r["mrr"] - (1 + 1 / 3) / 3) < 1e-9
    assert r["misses"] == [("3", "q3")]


def test_leak_score_separates_copied_wording():
    passage = "Emergent Fidelity Technologies objected to the Robinhood share forfeiture"
    corpus = [passage, "the court approved the motion", "notice of hearing on the motion"]
    idf = build_idf(corpus)
    copied = leak_score("Did Emergent Fidelity object to the Robinhood forfeiture?", passage, idf)
    paraphrased = leak_score("Did a lender fight over seized stock?", passage, idf)
    assert copied > paraphrased


def test_stratify_takes_one_passage_per_docket_and_skips_templates():
    pool = [{"doc_id": f"Docket_{d}#{i}", "text": "", "content_hash": f"{d}{i}"}
            for d in (1, 2, 3) for i in range(5)]
    titles = {"3": "Monthly Operating Report for March"}
    picked = stratify(pool, n=10, titles=titles)
    assert sorted(docket_of(p["doc_id"]) for p in picked) == ["1", "2"]

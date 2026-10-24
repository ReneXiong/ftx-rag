from ftx_rag.evaluation.run_eval import overlap
from ftx_rag.search import Hit, Searcher, rrf


def test_rrf_rewards_agreement_and_keeps_everything():
    fused = rrf([["a", "b", "c"], ["c", "d", "a"]])
    assert fused[:2] == ["a", "c"]
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_uses_rank_not_list_length():
    # Rank 1 in a single list beats rank 3 there, whatever the other list holds.
    assert rrf([["x", "y", "z"], []]) == ["x", "y", "z"]


class FakeSearcher(Searcher):
    def __init__(self, dense, lexical):
        super().__init__(conn=None, table="t")
        self._d, self._l = dense, lexical

    def dense(self, query, n=50):
        return [Hit(i, f"Docket_{i}", "") for i in self._d][:n]

    def lexical(self, query, n=50):
        return [Hit(i, f"Docket_{i}", "") for i in self._l][:n]


def test_hybrid_returns_union_of_both_candidate_lists():
    s = FakeSearcher(dense=["1", "2"], lexical=["3", "1"])
    hits = s.search("q", "hybrid")
    assert [h.chunk_id for h in hits][0] == "1"
    assert {h.chunk_id for h in hits} == {"1", "2", "3"}


def test_overlap_counts_complementary_finds():
    o = overlap([1, None, 3, None], [None, 2, 1, None])
    assert o == {"both": 0.25, "only_first": 0.25, "only_second": 0.25, "neither": 0.25}

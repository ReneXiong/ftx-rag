"""Score retrieval against a question set: recall@k and MRR, no LLM calls.

A question counts as answered when any chunk of its source filing comes back in
the top k chunks. Set membership is deterministic and free, so a change in the score
is always a change in the system, never in a judge model's mood.

Questions whose source chunk was never embedded are excluded rather than scored
as misses. During a partial rebuild that is most of them, and counting them
would report index coverage as if it were retrieval quality.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from ..search import DEPTH, MODES, open_searcher

KS = (1, 5, 10, 50)


def docket_of(doc_id: str) -> str:
    """'Docket_20467_att2#61' -> '20467': the filing, ignoring part and chunk."""
    m = re.search(r"\d+", doc_id or "")
    return m.group(0) if m else (doc_id or "")


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(questions: list[dict], retrieve: Callable[[str], list[str]],
             indexed_hashes: set[str] | None = None) -> dict:
    """`retrieve` maps a question to docket numbers, best first."""
    hits = dict.fromkeys(KS, 0)
    rr = 0.0
    answerable = 0
    ranks: list[int | None] = []
    misses = []
    for q in questions:
        h = q.get("gold_content_hash")
        if indexed_hashes is not None and h and h not in indexed_hashes:
            continue
        answerable += 1
        gold = docket_of(q["gold_doc_id"])
        rank = next((i for i, d in enumerate(retrieve(q["question"]), 1) if d == gold), None)
        ranks.append(rank)
        if rank is None:
            misses.append((gold, q["question"]))
            continue
        rr += 1 / rank
        for k in KS:
            hits[k] += rank <= k
    n = max(answerable, 1)
    return {
        "total": len(questions),
        "answerable": answerable,
        "recall": {k: hits[k] / n for k in KS},
        # Gold anywhere in the returned list: the most a reranker could recover.
        "found": sum(r is not None for r in ranks) / n,
        "mrr": rr / n,
        "ranks": ranks,
        "misses": misses,
    }


def overlap(a: list[int | None], b: list[int | None]) -> dict:
    """Per-question complementarity of two retrievers' ranks (None = not found)."""
    pairs = list(zip(a, b, strict=True))
    n = max(len(pairs), 1)
    return {
        "both": sum(x is not None and y is not None for x, y in pairs) / n,
        "only_first": sum(x is not None and y is None for x, y in pairs) / n,
        "only_second": sum(x is None and y is not None for x, y in pairs) / n,
        "neither": sum(x is None and y is None for x, y in pairs) / n,
    }


def run(questions_path: Path, modes: tuple[str, ...] = MODES, table: str | None = None,
        depth: int = DEPTH) -> dict:
    """Score each retrieval mode on the same answerable questions."""
    searcher = open_searcher(table)
    with searcher.conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT full_doc_id FROM {searcher.table}")
        indexed = {r[0] for r in cur.fetchall()}
    questions = load_questions(questions_path)
    results = {}
    for mode in modes:
        def retrieve(q: str, mode=mode) -> list[str]:
            return [docket_of(h.file_path) for h in searcher.search(q, mode, depth)]

        results[mode] = evaluate(questions, retrieve, indexed)
    return {"table": searcher.table, "depth": depth, "results": results}

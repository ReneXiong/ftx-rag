"""Generate a retrieval eval set from the corpus. No hand-writing required.

Each question is written by an LLM that sees one passage and nothing else, in
the voice of a creditor rather than a lawyer. The hard part is making the
questions test retrieval rather than string matching:

- A model shown a passage echoes its rarest wording, and then keyword search
  scores 100% while measuring nothing. `leak_score` sums the corpus IDF of
  rare words a question shares with its passage; the leakiest are set aside.
  (Separation measured on this corpus: unrelated question 6.6, verbatim 91.9.)
- Court dockets are heavily templated, so many questions are answered equally
  well by hundreds of filings and the gold label means nothing. Templated
  filing types are never sampled, and a second pass asks the model whether
  the question singles out one filing.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx

from .. import config, llm
from ..process import read_jsonl
from .run_eval import docket_of

MIN_CHARS = 400   # long enough to carry a fact
MIN_SCORE = 1.0   # extraction value score; skips number tables

PROMPT = """You are helping build a test set for a document retrieval system \
over the FTX bankruptcy court docket — roughly 20,000 filings. The users are \
FTX creditors: ordinary people who lost money, not lawyers.

Below is one passage from a court filing.

Write ONE question that a creditor might genuinely ask, in plain everyday \
language, AND that identifies this filing uniquely.

Bankruptcy filings are heavily templated, so a question like "did the judge \
find that notice was adequate?" is answered by hundreds of filings and is \
useless. Your question MUST contain at least one anchor that narrows it to \
this filing: a date, a dollar amount, a specific dispute, a named role, a \
distinctive event or outcome. Dates and numbers are encouraged.

Avoid copying distinctive phrases or defined legal terms verbatim, and legal \
jargon generally. Write how a worried person would type into a search box. \
Never write "this document" or similar — the reader does not have it.

Reply exactly SKIP if the passage is boilerplate, a signature block, a service \
list, a bare table of numbers, or otherwise carries nothing distinctive.

Reply with the question only. No preamble, no quotes.

PASSAGE:
{text}"""

SCREEN_PROMPT = """A search system indexes ~20,000 filings from the FTX \
bankruptcy case. Many are near-identical templates: claim transfers, fee \
receipts, monthly operating reports, and orders repeating the same findings.

Question: "{question}"

Would MANY different filings in that docket answer this question equally well?
Answer YES if it targets template language, a routine form field, or a \
recurring procedural step. Answer NO only if it points at one specific filing.

Reply with exactly one word: YES or NO"""

# Digits are ignored: dates and amounts are what distinguish one filing from
# another, so penalising them would select for generic questions.
_WORD = re.compile(r"[a-z]{3,}")

TEMPLATED = re.compile(
    r"transfer/assignment of claim|receipt of filing fee|monthly operating report"
    r"|post-confirmation report|certificate of (service|no objection)"
    r"|schedules/statements|affidavit of service|notice of appearance",
    re.I,
)


def tokenize(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def build_idf(texts: list[str]) -> dict[str, float]:
    df: Counter[str] = Counter()
    for t in texts:
        df.update(set(tokenize(t)))
    return {w: math.log(len(texts) / c) for w, c in df.items()}


def leak_score(question: str, passage: str, idf: dict[str, float]) -> float:
    """Summed IDF of words the question shares with its source passage."""
    shared = set(tokenize(question)) & set(tokenize(passage))
    unseen = max(idf.values(), default=1.0)
    return sum(idf.get(w, unseen) for w in shared)


def load_pool() -> list[dict]:
    pool = []
    for f in config.EXTRACTS_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        text = d.get("text")
        if isinstance(text, str) and len(text) >= MIN_CHARS \
                and float(d.get("score") or 0) >= MIN_SCORE and d.get("doc_id"):
            pool.append({"content_hash": d["content_hash"], "doc_id": d["doc_id"], "text": text})
    return pool


def stratify(pool: list[dict], n: int, titles: dict[str, str], seed: int = 0) -> list[dict]:
    """One passage per docket, skipping templated filing types.

    Sampling chunks uniformly would keep drawing from the few huge filings.
    """
    rng = random.Random(seed)
    by_docket: dict[str, list[dict]] = {}
    for p in pool:
        d = docket_of(p["doc_id"])
        if not TEMPLATED.search(titles.get(d, "")):
            by_docket.setdefault(d, []).append(p)
    dockets = sorted(by_docket)
    rng.shuffle(dockets)
    return [rng.choice(by_docket[d]) for d in dockets[:n]]


def eval_llm_settings() -> config.LLMSettings:
    """EVAL_* env vars override the pipeline's LLM, e.g. to use a stronger model."""
    base = config.llm_settings()
    return config.LLMSettings(
        os.getenv("EVAL_BASE_URL", base.base_url),
        os.getenv("EVAL_API_KEY", base.api_key),
        os.getenv("EVAL_MODEL", base.model),
        base.extra_body if not os.getenv("EVAL_BASE_URL") else {},
    )


def ask_question(client, settings, text: str) -> str | None:
    try:
        q = llm.chat(client, settings,
                     [{"role": "user", "content": PROMPT.format(text=text[:4000])}],
                     temperature=0.7)
    except Exception:
        return None
    if not q or q.upper().startswith("SKIP") or len(q) < 15:
        return None
    return q.splitlines()[0].strip().strip('"').strip()


def is_unique(client, settings, question: str) -> bool:
    """Screening errors keep the question: a flaky call must not shrink the set."""
    try:
        a = llm.chat(client, settings,
                     [{"role": "user", "content": SCREEN_PROMPT.format(question=question)}],
                     temperature=0)
    except Exception:
        return True
    return not a.strip().upper().startswith("YES")


def build(n: int = 60, workers: int = 4, leak_quantile: float = 0.8,
          indexed_table: str | None = None) -> dict:
    pool = load_pool()
    if indexed_table:
        # Draw only from embedded passages, so a partial index cannot make
        # questions unanswerable by construction.
        from ..search import connect
        with connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT full_doc_id FROM {indexed_table}")
            have = {r[0] for r in cur.fetchall()}
        pool = [p for p in pool if p["content_hash"] in have]
    if not pool:
        raise SystemExit("no usable passages — run `ftx-rag extract` first")

    rng = random.Random(0)
    idf = build_idf([p["text"] for p in rng.sample(pool, min(20000, len(pool)))])
    titles = {docket_of(r["docket_number"]): r.get("document_name", "")
              for r in read_jsonl(config.METADATA_FILE)}
    picked = stratify(pool, n, titles)
    settings = eval_llm_settings()

    with httpx.Client(timeout=120) as client, ThreadPoolExecutor(workers) as ex:
        questions = list(ex.map(lambda p: ask_question(client, settings, p["text"]), picked))
        rows = [{
            "question": q,
            "gold_doc_id": p["doc_id"],
            "gold_content_hash": p["content_hash"],
            "leak_score": round(leak_score(q, p["text"], idf), 2),
            "passage_preview": p["text"][:200].replace("\n", " "),
        } for p, q in zip(picked, questions, strict=True) if q]
        unique = list(ex.map(lambda r: is_unique(client, settings, r["question"]), rows))
    generic = [r for r, u in zip(rows, unique, strict=True) if not u]
    rows = [r for r, u in zip(rows, unique, strict=True) if u]
    if not rows:
        raise SystemExit("no questions generated — check the LLM endpoint")

    scores = sorted(r["leak_score"] for r in rows)
    cutoff = scores[min(int(len(scores) * leak_quantile), len(scores) - 1)]
    clean = [r for r in rows if r["leak_score"] <= cutoff]
    leaky = [r for r in rows if r["leak_score"] > cutoff]

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in (("questions.jsonl", clean), ("questions_leaky.jsonl", leaky),
                       ("questions_generic.jsonl", generic)):
        with (config.EVAL_DIR / name).open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
    with (config.EVAL_DIR / "questions_review.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*rows[0], "kept"])
        w.writeheader()
        w.writerows({**r, "kept": r["leak_score"] <= cutoff} for r in rows)

    return {"sampled": len(picked), "generated": len(rows) + len(generic),
            "generic": len(generic), "leaky": len(leaky), "kept": len(clean),
            "leak_cutoff": cutoff, "out_dir": str(config.EVAL_DIR)}

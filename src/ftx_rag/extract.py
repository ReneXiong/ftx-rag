"""Phase 1: LLM entity/relationship extraction, decoupled from graph writes.

LightRAG's `ainsert()` does extraction, embedding and the graph write in one
call, and the graph write re-persists the whole accumulated graph for every
chunk. On an 18K-node graph that alone costs ~15 s per chunk, so throughput was
capped at 2-3 chunks/min no matter how fast the LLM was.

Here the LLM step runs on its own: a direct async HTTP loop over the parsed
corpus, writing one JSON file per chunk. Graph assembly (`assemble.py`) then
runs offline with no LLM at all. The output schema maps 1:1 onto LightRAG's
`ainsert_custom_kg`.

Files are keyed by the chunk's content hash, so a crashed or interrupted run
resumes where it stopped. Delete a file to re-extract that chunk.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass

import httpx

from . import config, llm

logger = logging.getLogger(__name__)

# ── Chunk triage ───────────────────────────────────────────────────────
# Legal prose (motions, rulings, objections) is where the graph-worthy
# entities are. Schedule tables repeat the same name+amount tuples thousands of
# times. Scoring lets a run skip the tables and do the prose first.
_LEGAL_MARKERS = [
    "Bankruptcy", "§", "WHEREAS", "MOTION", "ORDER", "pursuant", "proceeding",
    "objection", "Claim No", "Trustee", "Court", "debtor", "creditor",
    "executory", "obligation", "foregoing", "hereby",
]
_NUMBER_ROW = re.compile(r"(?:\d+[.$]\d*\s+){3,}")
_MONEY = re.compile(r"\d{1,3}(,\d{3})*\.\d{2}")


def value_score(text: str) -> float:
    """Higher = more likely to hold legal substance worth extracting.

    Rough bands on this corpus: < 0 is numeric tables and claim schedules,
    0-0.5 is mixed, >= 0.5 is mostly prose.
    """
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) < 20:
        return 0.0
    prose_ratio = sum(1 for w in words if len(w) > 3) / len(words)
    markers = sum(text.count(m) for m in _LEGAL_MARKERS)
    number_rows = len(_NUMBER_ROW.findall(text))
    amounts = text.count("$") + len(_MONEY.findall(text))
    return (prose_ratio * 2.0
            + min(markers, 10) * 0.6
            - min(number_rows, 10) * 0.5
            - min(amounts, 20) * 0.2)


def chunk_text(text: str, size: int = config.CHUNK_CHARS,
               cap: int = config.MAX_GRAPH_DOC_CHARS) -> list[str]:
    if cap:
        text = text[:cap]
    return [text[i:i + size] for i in range(0, len(text), size)]


# ── Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are part of a graph-RAG pipeline for the FTX bankruptcy docket. "
    "Extract entities and relationships from the given court document chunk. "
    f"ENTITY_TYPES = {config.ENTITY_TYPES}. "
    "Respond ONLY with valid JSON. No reasoning, no markdown, no commentary. "
    "Include implicit-but-supported entities even in procedural text: courts, "
    "judges, statutes, government agencies, procedural roles (debtor, trustee, "
    "examiner, committee, retained professional), and named entities. "
    "Only include entities actually supported by the text; do not invent."
)

PROMPT = """Extract entities and relationships from this court document chunk.
Text:
{text}

Return JSON in EXACTLY this format (entity_type must be one of ENTITY_TYPES):
{{"entities":[{{"entity_type":"LEGAL_PARTY","entity_name":"Exact Name From Text","description":"brief role in this chunk"}}],
"relationships":[{{"src_id":"Exact Name From Text","tgt_id":"Exact Name From Text","description":"how src relates to tgt in this chunk"}}]}}

Rules:
- Use exact names as they appear in the text (preserve capitalization).
- Skip boilerplate: docket-stamp headers, page numbers, "FILED" lines.
- If nothing substantive, return {{"entities":[],"relationships":[]}}."""


@dataclass
class Chunk:
    doc_id: str
    text: str
    content_hash: str
    score: float


def plan(min_score: float, max_score: float | None = None) -> tuple[list[Chunk], int]:
    """Chunks still to extract, best first, plus how many fell outside the band."""
    work, out_of_band = [], 0
    for md in sorted(config.PARSED_DIR.glob("*.md")):
        for idx, text in enumerate(chunk_text(md.read_text(encoding="utf-8"))):
            if len(text.strip()) < 100:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if (config.EXTRACTS_DIR / f"{h}.json").exists():
                continue
            score = value_score(text)
            if score < min_score or (max_score is not None and score >= max_score):
                out_of_band += 1
                continue
            work.append(Chunk(f"{md.stem}#{idx}", text, h, score))
    work.sort(key=lambda c: c.score, reverse=True)
    return work, out_of_band


async def extract_chunk(client: httpx.AsyncClient, settings: config.LLMSettings,
                        text: str) -> dict:
    reply = await llm.achat(
        client, settings,
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": PROMPT.format(text=text)}],
        max_tokens=2000, response_format={"type": "json_object"},
    )
    return llm.parse_json_reply(reply)


async def run(concurrency: int = 8, retries: int = 3, min_score: float = 0.5,
              max_score: float | None = None, limit: int | None = None) -> dict:
    settings = config.llm_settings()
    config.EXTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    work, out_of_band = plan(min_score, max_score)
    work = work[:limit]
    logger.info("%d chunks to extract (%d outside score band)", len(work), out_of_band)

    queue: asyncio.Queue[Chunk] = asyncio.Queue()
    for c in work:
        queue.put_nowait(c)
    stats = {"extracted": 0, "failed": 0, "out_of_band": out_of_band}
    t0 = time.monotonic()

    async def worker(client: httpx.AsyncClient) -> None:
        while not queue.empty():
            c = queue.get_nowait()
            for attempt in range(retries):
                try:
                    result = await extract_chunk(client, settings, c.text)
                    break
                except Exception as e:
                    logger.debug("[%s] attempt %d: %s", c.content_hash, attempt + 1, e)
                    await asyncio.sleep(2 ** attempt)
            else:
                stats["failed"] += 1
                logger.warning("[%s] dropped after %d attempts", c.content_hash, retries)
                continue
            (config.EXTRACTS_DIR / f"{c.content_hash}.json").write_text(json.dumps({
                "content_hash": c.content_hash,
                "doc_id": c.doc_id,
                "text": c.text,
                "score": c.score,
                "entities": result.get("entities", []),
                "relationships": result.get("relationships", []),
            }, ensure_ascii=False), encoding="utf-8")
            stats["extracted"] += 1
            if stats["extracted"] % 200 == 0:
                rate = stats["extracted"] / (time.monotonic() - t0) * 60
                logger.info("%d/%d extracted, %.1f chunks/min",
                            stats["extracted"], len(work), rate)

    async with httpx.AsyncClient(timeout=180) as client:
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
    return stats

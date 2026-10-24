"""Stage 0: classify every docket entry and parse the substantive ones.

Input is `metadata.jsonl`, one docket entry per line:

    {"docket_number": "Docket # 342", "document_name": "...", "date_filed": "01/03/2023",
     "pdf_path": "pdfs/342.pdf", "attachments": ["pdfs/342_1.pdf"]}

Bulk filings (see `rules`) are classified from their title alone and never
parsed. Everything else is classified by the LLM and parsed to Markdown under
`parsed/`, which is what Phase 1 extraction reads.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)


def doc_slug(docket_number: str, attachment: int | None = None) -> str:
    """Filesystem-safe document id: 'Docket # 342', 2 -> 'Docket_342_att2'."""
    slug = re.sub(r"[^\w-]+", "_", docket_number).strip("_") or "unknown"
    return f"{slug}_att{attachment}" if attachment else slug


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("skipping malformed line in %s", path)
    return rows


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else config.DATA_DIR / p


def process_entry(entry: dict, classifier, parser) -> dict:
    docket = entry.get("docket_number", "")
    classification = classifier.classify(entry)
    parsed: list[str] = []
    if not classification["bulk"]:
        files = [entry.get("pdf_path")] + list(entry.get("attachments") or [])
        for i, f in enumerate(files):
            if not f or not _resolve(f).is_file():
                continue
            slug = doc_slug(docket, i or None)
            if parser.parse(_resolve(f), config.PARSED_DIR / f"{slug}.md"):
                parsed.append(slug)
    return {
        "docket_number": docket,
        "document_name": entry.get("document_name", ""),
        "date_filed": entry.get("date_filed", ""),
        "classification": classification,
        "parsed": parsed,
    }


def process_all(limit: int | None = None, workers: int = 4) -> dict:
    """Resumable: entries already in classified.jsonl are skipped."""
    from .classify import DocketClassifier
    from .parse import DocketParser

    entries = read_jsonl(config.METADATA_FILE)
    done = {r["docket_number"] for r in read_jsonl(config.CLASSIFIED_FILE)}
    todo = [e for e in entries if e.get("docket_number") not in done][:limit]
    if not todo:
        return {"processed": 0, "bulk": 0, "parsed_docs": 0, "skipped": len(done)}

    classifier = DocketClassifier()
    parser = DocketParser()
    stats = {"processed": 0, "bulk": 0, "parsed_docs": 0, "skipped": len(done)}

    config.CLASSIFIED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with config.CLASSIFIED_FILE.open("a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_entry, e, classifier, parser): e for e in todo}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:
                logger.error("failed %s: %s", futures[fut].get("docket_number"), e)
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            stats["processed"] += 1
            stats["bulk"] += row["classification"]["bulk"]
            stats["parsed_docs"] += len(row["parsed"])
    return stats

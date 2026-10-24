"""ftx-rag command line.

    ftx-rag process      classify entries, parse substantive PDFs
    ftx-rag extract      Phase 1: LLM entity/relationship extraction
    ftx-rag assemble     Phase 2: embed + build the graph (no LLM)
    ftx-rag search "..." find filings: dense + full-text, fused (no LLM)
    ftx-rag query "..."  ask a question
    ftx-rag chat         interactive session
    ftx-rag questions    generate an eval set
    ftx-rag evaluate     score dense / lexical / hybrid retrieval (no LLM)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import config

app = typer.Typer(help="Graph RAG over the FTX bankruptcy docket.", add_completion=False,
                  no_args_is_help=True)
console = Console()


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _show(title: str, stats: dict) -> None:
    table = Table(title=title, show_header=False)
    for k, v in stats.items():
        table.add_row(k.replace("_", " "), f"{v:,}" if isinstance(v, int) else str(v))
    console.print(table)


@app.command()
def process(
    limit: int | None = typer.Option(None, "--limit", "-l"),
    workers: int = typer.Option(4, "--workers", "-w"),
):
    """Classify every entry in metadata.jsonl; parse the non-bulk PDFs."""
    from .process import process_all

    _show("process", process_all(limit=limit, workers=workers))


@app.command()
def extract(
    concurrency: int = typer.Option(8, "--concurrency", "-c"),
    min_score: float = typer.Option(0.5, help="skip chunks scoring below this"),
    max_score: float | None = typer.Option(
        None, help="only chunks below this; with --min-score 0 --max-score 0.5, "
                   "extracts the mixed band a first pass skipped"),
    limit: int | None = typer.Option(None, "--limit", "-l"),
):
    """Phase 1: extract entities and relationships from parsed chunks."""
    from .extract import run

    _show("extract", asyncio.run(run(concurrency=concurrency, min_score=min_score,
                                     max_score=max_score, limit=limit)))


@app.command()
def assemble(batch_size: int = typer.Option(1000, help="extractions per graph write")):
    """Phase 2: embed saved extractions and write the graph. No LLM calls."""
    from .assemble import run

    _show("assemble", asyncio.run(run(batch_size=batch_size)))


async def _ask(questions, mode: str, top_k: int) -> None:
    from .kb import KnowledgeBase

    kb = KnowledgeBase()
    await kb.initialize()
    try:
        async for q in questions:
            console.print(await kb.query(q, mode=mode, top_k=top_k))
    finally:
        await kb.finalize()


@app.command()
def query(
    question: str,
    mode: str = typer.Option("hybrid", help="naive | local | global | hybrid | mix"),
    top_k: int = typer.Option(config.RAG_TOP_K, "--top-k", "-k"),
):
    """Answer one question from the knowledge base."""
    async def one():
        yield question

    asyncio.run(_ask(one(), mode, top_k))


@app.command()
def chat(
    mode: str = typer.Option("hybrid"),
    top_k: int = typer.Option(config.RAG_TOP_K, "--top-k", "-k"),
):
    """Interactive question answering. Empty line or Ctrl-D to exit."""
    async def lines():
        while True:
            try:
                q = await asyncio.to_thread(console.input, "\n[bold cyan]>[/] ")
            except (EOFError, KeyboardInterrupt):
                return
            if not q.strip():
                return
            yield q

    asyncio.run(_ask(lines(), mode, top_k))


@app.command()
def questions(
    n: int = typer.Option(60, help="passages to sample"),
    workers: int = typer.Option(4),
    leak_quantile: float = typer.Option(0.8, help="set aside questions above this quantile"),
    indexed_only: str | None = typer.Option(
        None, metavar="TABLE", help="sample only passages already embedded in TABLE"),
):
    """Generate a retrieval eval set from the extracted corpus."""
    from .evaluation.build_questions import build

    _show("questions", build(n=n, workers=workers, leak_quantile=leak_quantile,
                             indexed_table=indexed_only))


@app.command()
def evaluate(
    questions_file: Path = typer.Option(config.EVAL_DIR / "questions.jsonl", "--questions"),
    mode: list[str] = typer.Option(["dense", "lexical", "hybrid"], "--mode", "-m"),
    table: str | None = typer.Option(None, help="LightRAG chunk table (default: first)"),
    depth: int = typer.Option(50, help="candidates per retriever"),
):
    """Score retrieval modes side by side: recall@k, candidate recall, MRR. No LLM calls."""
    from .evaluation.run_eval import KS, overlap, run

    r = run(questions_file, modes=tuple(mode), table=table, depth=depth)
    results = r["results"]
    first = next(iter(results.values()))
    console.print(f"table [bold]{r['table']}[/], {first['answerable']}/{first['total']} "
                  f"questions answerable from the index, {depth} candidates per retriever")
    t = Table()
    t.add_column("mode")
    for k in KS:
        t.add_column(f"R@{k}", justify="right")
    t.add_column("found", justify="right")
    t.add_column("MRR", justify="right")
    for name, res in results.items():
        t.add_row(name, *(f"{res['recall'][k]:.3f}" for k in KS),
                  f"{res['found']:.3f}", f"{res['mrr']:.3f}")
    console.print(t)
    if "dense" in results and "lexical" in results:
        o = overlap(results["dense"]["ranks"], results["lexical"]["ranks"])
        console.print(f"found by both {o['both']:.0%}, dense only {o['only_first']:.0%}, "
                      f"lexical only {o['only_second']:.0%}, neither {o['neither']:.0%}")
    for gold, q in results[mode[-1]]["misses"][:5]:
        console.print(f"  [dim]{mode[-1]} miss [docket {gold}] {q[:90]}[/]")


@app.command()
def search(
    text: str,
    mode: str = typer.Option("hybrid", help="dense | lexical | hybrid"),
    n: int = typer.Option(10, "-n", help="results to show"),
    table: str | None = typer.Option(None, help="LightRAG chunk table (default: first)"),
):
    """Find the filings that best match TEXT. Retrieval only, no LLM."""
    from .search import open_searcher

    hits = open_searcher(table).search(text, mode)[:n]
    for i, h in enumerate(hits, 1):
        snippet = " ".join(h.content.split())[:160]
        console.print(f"{i:>2}. [bold]{h.file_path}[/]\n    [dim]{snippet}[/]")


if __name__ == "__main__":
    app()

"""PDF -> Markdown via Docling, with a pypdf fallback."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

logger = logging.getLogger(__name__)

DOCLING_TIMEOUT_S = 1200
DOCLING_MAX_MB = 10  # larger PDFs are almost always scanned exhibits; pypdf is enough


class DocketParser:
    def __init__(self, table_structure: bool | None = None):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorDevice,
            AcceleratorOptions,
            PdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        from .config import torch_device

        opts = PdfPipelineOptions()
        opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)
        # TableFormer deadlocks inside multi-head attention on PyTorch MPS, so
        # table structure recognition is off on Apple Silicon unless forced on.
        if table_structure is None:
            table_structure = torch_device() != "mps"
        opts.do_table_structure = table_structure
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

    def to_markdown(self, pdf: Path) -> str:
        """Docling first; pypdf when the PDF is too large or Docling fails."""
        size_mb = pdf.stat().st_size / 2**20
        if size_mb > DOCLING_MAX_MB:
            logger.info("%s is %.1f MB, using pypdf", pdf.name, size_mb)
            return pypdf_text(pdf)
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(self.converter.convert, str(pdf))
            try:
                return future.result(timeout=DOCLING_TIMEOUT_S).document.export_to_markdown()
            except FuturesTimeout:
                logger.warning("Docling timed out on %s, using pypdf", pdf.name)
            except Exception as e:
                logger.warning("Docling failed on %s (%s), using pypdf", pdf.name, e)
        return pypdf_text(pdf)

    def parse(self, pdf: Path, out: Path) -> bool:
        """Write `out` (Markdown) unless it already exists. True if it has text.

        Parsing is the slowest local step, so an existing output is always
        reused; delete the file to force a re-parse.
        """
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(self.to_markdown(pdf), encoding="utf-8")
        return out.stat().st_size > 0


def pypdf_text(pdf: Path) -> str:
    try:
        from pypdf import PdfReader

        return "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf)).pages).strip()
    except Exception as e:
        logger.warning("pypdf failed on %s: %s", pdf.name, e)
        return ""

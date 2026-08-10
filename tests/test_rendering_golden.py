"""Golden-image characterization tests for the rasterizer.

``RenderedPage.rasterize`` is large, branch-heavy, and until now largely
unreachable from tests.  These tests pin its *output* rather than its
structure, so that the decomposition work described in
``docs/architecture.md`` can proceed without a rendering regression slipping
through: any change to a painted pixel changes the digest.

Two layers:

* The always-on layer renders :data:`COVERING_SUBSET` — 24 corpus documents
  chosen by greedy set cover so that together they execute *every* line of
  ``rendering.py`` that the full 224-document corpus reaches.  It runs in a
  few seconds.
* A slow layer over the whole corpus, gated behind
  ``CORE_PDF_RASTER_GOLDEN_FULL``, catches the rest.

Regenerate the digests after an *intentional* rendering change::

    CORE_PDF_UPDATE_RASTER_GOLDEN=1 uv run pytest tests/test_rendering_golden.py -q

Review the resulting diff to ``first_page_scale1.json`` before committing it.
A refactor that is supposed to preserve behavior must produce no diff at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any, cast

import pytest

from core_pdf import PdfDocument

CORPUS = pathlib.Path(__file__).resolve().parent / "fixtures" / "SCORE-Bench" / "src"
SNAPSHOT = (
    pathlib.Path(__file__).resolve().parent / "snapshots" / "raster" / "first_page_scale1.json"
)

# Fixed so the digest is reproducible; do not vary these without regenerating.
RASTER_SCALE = 1.0
RASTER_BACKGROUND = (255, 255, 255, 255)

# Greedy line-cover of rendering.py over the corpus: these 24 documents reach
# 100% of the lines that all 224 reach. Recompute with scripts/raster_cover.py
# if the corpus or the rasterizer's structure changes substantially.
COVERING_SUBSET = (
    "PDFTriage-p7-p002.pdf",
    "i-9-p001.pdf",
    "national_ai_rd_strategic_plan-p28.pdf",
    "BurningCharacteristicsFir-p016.pdf",
    "BarrowArchAnalysis_Alaska1984-p076.pdf",
    "Index_to_Positions_table_vertical_text-p063.pdf",
    "TAMU-Annual-Report-2023p18-23-p004.pdf",
    "circuit schematic.pdf",
    "NASA-SNA-8-D-027III-Rev2-CsmLmSpacecraftOperationalDataBook-Volume3-MassProperties-Pg54.pdf",
    "This_Is_Caltech_2018_p20-20.pdf",
    "French-p076.pdf",
    "AlienPlantThreatAssess-p24-p27-p004.pdf",
    "wipo-2022-financial-report-p24-p30-p001.pdf",
    "csia_federal_plan-p47-p52-p001.pdf",
    "FOOD_ELEMENTS_FALL_2017p20-20.pdf",
    "sydd0278.pdf",
    "Employee_Health_Benefits_Assess-p006.pdf",
    "s12940-025-01154-x-p001.pdf",
    "160106big-data-rpt_p44-44.pdf",
    "GOOGLE-10Q-2023-03-31-7pages-p001.pdf",
    "GlobalTrends_2040p10-17-p007.pdf",
    "Mission-costs_p27-35-p006.pdf",
    "VCAs_REV2_SCHEMATIC-p001.pdf",
    "ijerph-19-00825-p020.pdf",
)


def raster_digest(pdf: pathlib.Path) -> str:
    """Render page 1 and hash the RGBA buffer."""
    with PdfDocument.open(pdf) as document:
        rendered = document.pages[0].render()
        raster = rendered.rasterize(
            scale=RASTER_SCALE,
            background=RASTER_BACKGROUND,
            cache=False,
        )
        return hashlib.sha256(raster.pixels.tobytes()).hexdigest()


def load_snapshot() -> dict[str, str]:
    if not SNAPSHOT.is_file():
        cast(Any, pytest.skip)(f"raster snapshot not present at {SNAPSHOT}")
    return json.loads(SNAPSHOT.read_text())


def corpus_pdfs() -> list[pathlib.Path]:
    if not CORPUS.is_dir():
        return []
    return sorted(CORPUS.glob("*.pdf"))


@pytest.mark.skipif(
    not os.environ.get("CORE_PDF_UPDATE_RASTER_GOLDEN"),
    reason="set CORE_PDF_UPDATE_RASTER_GOLDEN=1 to rewrite the raster snapshot",
)
def test_regenerate_raster_snapshot() -> None:
    """Not a test — the documented way to rewrite the snapshot after a change."""
    pdfs = corpus_pdfs()
    if not pdfs:
        cast(Any, pytest.skip)(f"SCORE-Bench corpus not present at {CORPUS}")
    digests = {pdf.name: raster_digest(pdf) for pdf in pdfs}
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(digests, indent=1, sort_keys=True) + "\n")


@pytest.mark.parametrize("name", COVERING_SUBSET)
def test_first_page_raster_matches_snapshot(name: str) -> None:
    pdf = CORPUS / name
    if not pdf.is_file():
        cast(Any, pytest.skip)(f"corpus document not present: {name}")
    expected = load_snapshot().get(name)
    if expected is None:
        cast(Any, pytest.fail)(f"no snapshot digest recorded for {name}")
    assert raster_digest(pdf) == expected, (
        f"raster output changed for {name}. If the change is intentional, "
        f"regenerate with CORE_PDF_UPDATE_RASTER_GOLDEN=1 and review the diff."
    )


def test_covering_subset_is_present_in_the_snapshot() -> None:
    """Guards against a rename silently turning the fast layer into skips."""
    snapshot = load_snapshot()
    missing = [name for name in COVERING_SUBSET if name not in snapshot]
    assert not missing, f"covering subset entries absent from the snapshot: {missing}"


@pytest.mark.skipif(
    not os.environ.get("CORE_PDF_RASTER_GOLDEN_FULL"),
    reason="set CORE_PDF_RASTER_GOLDEN_FULL=1 to raster the whole corpus",
)
def test_whole_corpus_raster_matches_snapshot() -> None:
    snapshot = load_snapshot()
    pdfs = corpus_pdfs()
    if not pdfs:
        cast(Any, pytest.skip)(f"SCORE-Bench corpus not present at {CORPUS}")
    changed = [pdf.name for pdf in pdfs if raster_digest(pdf) != snapshot.get(pdf.name)]
    assert not changed, f"raster output changed for {len(changed)} documents: {changed[:10]}"

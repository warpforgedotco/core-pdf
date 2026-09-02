"""Cross-platform golden-image characterization tests for the rasterizer.

``RenderedPage.rasterize`` is large and branch-heavy. These tests pin its
*output* rather than its structure so rendering regressions cannot slip through.
Ordinary pages use exact RGBA hashes. Irreversible JPEG 2000 decoding can differ
by one sample between OpenJPEG's x86 SIMD and ARM scalar paths, so those pages
use lossless canonical references with measured RGB bounds and exact alpha.

Two layers:

* The always-on exact layer renders :data:`COVERING_SUBSET` — 22 corpus
  documents chosen by greedy set cover after seeding coverage with the five
  tolerant pages. Together those cases execute *every* line of the ``render/``
  package that the full 224-document corpus reaches. They run in a few seconds,
  which is what makes them the right layer to run while iterating.
* A distributed layer over the whole corpus, behind ``CORE_PDF_RASTER_GOLDEN_FULL``.
  **CI sets that variable**, so the full corpus is a merge gate; the variable
  exists so local runs can opt out of the few minutes it costs, not so the
  sweep can be skipped indefinitely. Each document is a separate pytest case,
  so ``-n 2`` distributes the expensive raster work. Line coverage is not
  output coverage — the always-on cases reach every line of ``render/`` and
  still cannot see a changed pixel in the remaining documents.

Regenerate the digests after an *intentional* rendering change, without
running the corpus through pytest a second time::

    uv run python scripts/update_raster_golden.py

Reference regeneration is restricted to the pinned Ubuntu x86_64 codec
environment unless the explicitly destructive override is used. Review the
manifest and reference-image diff before committing it. A refactor that
preserves behavior must produce no diff at all.
"""

from __future__ import annotations

import functools
import os
from typing import Any, cast

import pytest

from scripts.raster_golden import (
    CANONICAL_SOURCE,
    CORPUS,
    REFERENCE_DIRECTORY,
    RasterSnapshot,
    RasterSnapshotError,
    corpus_pdfs,
    load_reference_raster,
    load_snapshot,
    raster_snapshot_failure,
)
from tests.helpers.paths import require_fixture

# Greedy line-cover of render/ over the corpus: these 22 exact-digest documents,
# together with the five always-on tolerant documents below, reach 100% of the
# lines that all 224 reach. Recompute with scripts/raster_cover.py if the corpus
# or the rasterizer's structure changes substantially.
COVERING_SUBSET = (
    "PDFTriage-p7-p002.pdf",
    "national_ai_rd_strategic_plan-p28.pdf",
    "BurningCharacteristicsFir-p016.pdf",
    "FOOD_ELEMENTS_FALL_2017p20-20.pdf",
    "i-9-p001.pdf",
    "Index_to_Positions_table_vertical_text-p063.pdf",
    "circuit schematic.pdf",
    "NASA-SNA-8-D-027III-Rev2-CsmLmSpacecraftOperationalDataBook-Volume3-MassProperties-Pg54.pdf",
    "csia_federal_plan-p47-p52-p006.pdf",
    "Employee_Health_Benefits_Assess-p006.pdf",
    "French-p076.pdf",
    "AlienPlantThreatAssess-p24-p27-p004.pdf",
    "wipo-2022-financial-report-p24-p30-p001.pdf",
    "O27Hara_DeepSeaFloorBio-p001.pdf",
    "UNH-Q4-2023-Form-10-K-p028.pdf",
    "cb9722en_p35-36-p001.pdf",
    "sydd0278.pdf",
    "BarrowArchAnalysis_Alaska1984-p076.pdf",
    "s12940-025-01154-x-p001.pdf",
    "Dellafera_Warrant-Redacted-p028.pdf",
    "UN_ProductComparison-p001.pdf",
    "ijerph-19-00825-p020.pdf",
)

# Every corpus page that paints an irreversible JPEG 2000 image. These run on
# every platform because they exercise the portable bounded-comparison path;
# the exact-digest covering subset deliberately excludes them.
IRREVERSIBLE_JPX_SUBSET = (
    "153rd-Omaha-Pow-Wow-p001.pdf",
    "2023-OFC-Technical-Brochure-p004.pdf",
    "33715_water_p34-35-p001.pdf",
    "Roosevelt-Letter-Oppenheimer-p001.pdf",
    "SPUR_Future_Of_Transportation-p001.pdf",
)

internal_FULL_CORPUS = bool(os.environ.get("CORE_PDF_RASTER_GOLDEN_FULL"))
internal_FULL_CORPUS_NAMES = (
    tuple(pdf.name for pdf in corpus_pdfs()) if internal_FULL_CORPUS else ("",)
)


@functools.cache
def internal_snapshot() -> RasterSnapshot:
    try:
        return load_snapshot()
    except RasterSnapshotError as exc:
        pytest.fail(str(exc))


@pytest.mark.skipif(
    internal_FULL_CORPUS,
    reason="the full-corpus cases include the covering subset",
)
@pytest.mark.parametrize("name", COVERING_SUBSET)
def test_first_page_raster_matches_snapshot(name: str) -> None:
    pdf = require_fixture(CORPUS / name, f"corpus document not present: {name}")
    failure = raster_snapshot_failure(pdf, internal_snapshot())
    assert failure is None, f"raster output changed for {name}: {failure}"


@pytest.mark.skipif(
    internal_FULL_CORPUS,
    reason="the full-corpus cases include every irreversible-JPX page",
)
@pytest.mark.parametrize("name", IRREVERSIBLE_JPX_SUBSET)
def test_irreversible_jpx_raster_is_portable(name: str) -> None:
    pdf = require_fixture(CORPUS / name, f"corpus document not present: {name}")
    failure = raster_snapshot_failure(pdf, internal_snapshot())
    assert failure is None, f"portable JPX raster changed for {name}: {failure}"


def test_covering_subset_is_present_in_the_snapshot() -> None:
    """Guards against a rename silently turning the fast layer into skips."""
    snapshot = internal_snapshot()
    missing = [name for name in COVERING_SUBSET if name not in snapshot.names]
    assert not missing, f"covering subset entries absent from the snapshot: {missing}"


def test_covering_subset_is_disjoint_from_tolerant_snapshots() -> None:
    overlap = set(COVERING_SUBSET).intersection(internal_snapshot().tolerant)
    assert not overlap, f"covering subset duplicates tolerant snapshot pages: {sorted(overlap)}"


def test_every_irreversible_jpx_page_has_a_tolerant_snapshot() -> None:
    snapshot = internal_snapshot()
    assert set(snapshot.tolerant) == set(IRREVERSIBLE_JPX_SUBSET)


def test_raster_snapshot_matches_the_corpus_inventory() -> None:
    pdfs = corpus_pdfs()
    if not pdfs:
        cast(Any, pytest.skip)(f"SCORE-Bench corpus not present at {CORPUS}")
    snapshot = internal_snapshot()
    assert snapshot.names == {pdf.name for pdf in pdfs}


def test_tolerant_raster_references_are_valid() -> None:
    snapshot = internal_snapshot()
    for entry in snapshot.tolerant.values():
        load_reference_raster(entry)
    referenced = {entry.reference_path.resolve() for entry in snapshot.tolerant.values()}
    present = {path.resolve() for path in REFERENCE_DIRECTORY.glob("*.png")}
    assert present == referenced


def test_raster_snapshot_records_the_canonical_environment() -> None:
    assert internal_snapshot().canonical_source == CANONICAL_SOURCE


@pytest.mark.skipif(
    not internal_FULL_CORPUS,
    reason="set CORE_PDF_RASTER_GOLDEN_FULL=1 to raster the whole corpus",
)
@pytest.mark.parametrize("name", internal_FULL_CORPUS_NAMES)
def test_whole_corpus_raster_matches_snapshot(name: str) -> None:
    pdf = require_fixture(CORPUS / name, f"SCORE-Bench corpus not present at {CORPUS}")
    failure = raster_snapshot_failure(pdf, internal_snapshot())
    assert failure is None, f"raster output changed for {name}: {failure}"

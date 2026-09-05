# SPDX-License-Identifier: AGPL-3.0-only
"""Structure ActualText replaces an element's complete marked-content sequence."""

import pytest

from core_pdf.impl._impl.extract.capture import (
    capture_page,
    internal_apply_structure_actual_text,
)
from core_pdf.impl._impl.model.geometry import bbox_union
from core_pdf.impl._impl.model.runs import TextRun
from tests.helpers.pdf_bytes import HELVETICA, assemble_pdf, open_pdf, stream_obj


def internal_tagged_pdf(
    content: bytes,
    replacements: tuple[bytes, ...],
    owners: tuple[int, ...],
) -> bytes:
    element_refs = [f"{7 + index} 0 R".encode() for index in range(len(replacements))]
    parents = b" ".join(element_refs[owner] for owner in owners)
    elements = []
    for owner, replacement in enumerate(replacements):
        kids = b" ".join(str(mcid).encode() for mcid, index in enumerate(owners) if index == owner)
        elements.append(
            b"<< /Type /StructElem /S /Span /P 6 0 R /Pg 3 0 R /K ["
            + kids
            + b"] /ActualText ("
            + replacement
            + b") >>"
        )
    return assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R /StructTreeRoot 6 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R /StructParents 0 >>",
            stream_obj(b"BT /F1 12 Tf 10 40 Td " + content + b" ET"),
            HELVETICA,
            b"<< /Type /StructTreeRoot /K ["
            + b" ".join(element_refs)
            + b"] /ParentTree << /Nums [0 ["
            + parents
            + b"]] >> >>",
            *elements,
        ]
    )


@pytest.mark.parametrize("replacement", [b"AlphaBeta", b""])
def test_structure_actual_text_replaces_shared_element_once(replacement: bytes) -> None:
    content = b"(L) Tj /Span << /MCID 0 >> BDC (A) Tj EMC /Span << /MCID 1 >> BDC (B) Tj EMC (R) Tj"
    with open_pdf(internal_tagged_pdf(content, (replacement,), (0, 0))) as document:
        page = document.pages[0]
        structure = page.structure
        assert structure is not None
        assert structure[0] is structure[1]
        captured = capture_page(page)
        expected = ("L", replacement.decode(), "R") if replacement else ("L", "R")
        assert captured.observations.text == expected
        assert captured.evidence.glyphs.actual_text_characters == len(replacement)
        assert "".join(document.extract().text.split()) == "".join(expected)


def test_structure_actual_text_keeps_identical_replacements_from_distinct_elements() -> None:
    content = b"/Span << /MCID 0 >> BDC (A) Tj EMC 100 0 Td /Span << /MCID 1 >> BDC (B) Tj EMC"
    with open_pdf(internal_tagged_pdf(content, (b"Same", b"Same"), (0, 1))) as document:
        captured = capture_page(document.pages[0])
        assert captured.observations.text == ("Same", "Same")
        assert captured.evidence.glyphs.actual_text_characters == 8
        assert document.extract().text.split() == ["Same", "Same"]


@pytest.mark.parametrize("second_mcid", [0, 1])
def test_structure_actual_text_keeps_all_covered_geometry_and_source_clusters(
    second_mcid: int,
) -> None:
    content = (
        b"/Span << /MCID 0 >> BDC (A) Tj EMC /F1 24 Tf /Span << /MCID "
        + str(second_mcid).encode()
        + b" >> BDC (B) Tj EMC"
    )
    with open_pdf(internal_tagged_pdf(content, (b"AlphaBeta",), (0, 0))) as document:
        page = document.pages[0]
        source = page.get_page_program().runs
        assert tuple(run.text for run in source) == ("A", "B")
        source_boxes = tuple((run.x0, run.y0, run.x1, run.y1) for run in source)
        expected_bbox = bbox_union(source_boxes)
        expected_advance = bbox_union(run.advance_bbox for run in source)
        expected_ink = bbox_union(run.ink_bbox for run in source)
        expected_clusters = tuple(cluster for run in source for cluster in run.glyph_clusters)
        assert len(expected_clusters) == 2
        first_baseline, last_baseline = source[0].baseline, source[-1].baseline
        assert first_baseline is not None
        assert last_baseline is not None

        (replacement,) = internal_apply_structure_actual_text(page, source)

        assert (replacement.x0, replacement.y0, replacement.x1, replacement.y1) == expected_bbox
        assert replacement.advance_bbox == expected_advance
        assert replacement.ink_bbox == expected_ink
        assert replacement.baseline == (*first_baseline[:2], *last_baseline[2:])
        assert replacement.glyph_clusters == expected_clusters
        assert tuple(run.text for run in source) == ("A", "B")
        assert tuple((run.x0, run.y0, run.x1, run.y1) for run in source) == source_boxes

        captured = capture_page(page)
        assert captured.observations.bbox[0] == pytest.approx(expected_bbox)
        assert captured.evidence.glyphs.glyph_count == 2
        assert captured.evidence.glyphs.actual_text_characters == len("AlphaBeta")
        assert document.extract().text == "AlphaBeta\f"


def test_structure_actual_text_preserves_neighboring_inline_replacements() -> None:
    content = (
        b"/Span << /ActualText (Inline) >> BDC (X) Tj EMC "
        b"/Span << /MCID 0 >> BDC (A) Tj EMC "
        b"/Span << /MCID 1 >> BDC (B) Tj EMC "
        b"/Span << /ActualText () >> BDC (Y) Tj EMC"
    )
    with open_pdf(internal_tagged_pdf(content, (b"Structured",), (0, 0))) as document:
        captured = capture_page(document.pages[0])
        assert captured.observations.text == ("Inline", "Structured")
        first, second = captured.observations.references
        assert isinstance(first, TextRun)
        assert isinstance(second, TextRun)
        assert ("unicode_source", "actual_text") in first.provenance
        assert ("unicode_source", "structure_actual_text") in second.provenance
        assert captured.evidence.glyphs.actual_text_characters == len("InlineStructured")
        assert "".join(document.extract().text.split()) == "InlineStructured"

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


def internal_nested_tagged_pdf(
    replacement: bytes | None,
    child_replacement: bytes | None = None,
    *,
    parent_cycle: bool = False,
) -> bytes:
    def actual_text(value: bytes | None) -> bytes:
        return b"" if value is None else b"/ActualText (" + value + b") "

    content = (
        b"BT /F1 12 Tf 40 200 Td (L) Tj /Span << /MCID 0 >> BDC (A) Tj EMC "
        b"/F1 24 Tf /Span << /MCID 1 >> BDC (B) Tj EMC (R) Tj ET"
    )
    return assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R /StructTreeRoot 6 0 R /MarkInfo << /Marked true >> >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R /StructParents 0 >>",
            stream_obj(content),
            HELVETICA,
            b"<< /Type /StructTreeRoot /K 7 0 R /ParentTree << /Nums [0 [9 0 R 10 0 R]] >> >>",
            b"<< /Type /StructElem /S /Document /P 6 0 R /K 8 0 R >>",
            b"<< /Type /StructElem /S /Span /P "
            + (b"8" if parent_cycle else b"7")
            + b" 0 R /Pg 3 0 R /K [9 0 R 10 0 R] "
            + actual_text(replacement)
            + b">>",
            b"<< /Type /StructElem /S /Span /P 8 0 R /Pg 3 0 R /K 0 "
            + actual_text(child_replacement)
            + b">>",
            b"<< /Type /StructElem /S /Span /P 8 0 R /Pg 3 0 R /K 1 "
            + actual_text(child_replacement)
            + b">>",
        ],
        version="1.7",
    )


# External inspection: qpdf 12.3.2 validates this hierarchy; pdfinfo 26.07.0
# shows the two child spans. Poppler 26.07.0 ignores structure ActualText;
# MuPDF 1.28.2 honors direct values but ignores ancestor values. Expectations
# follow ISO 32000-2 Table 355: replacement includes the element's children.
# PDF Association summary, page 2:
# https://pdfa.org/download-area/cheat-sheets/LogicalStructureObjects.pdf
@pytest.mark.parametrize("replacement", [b"Joined", b""])
@pytest.mark.parametrize("child_replacement", [None, b"Child", b""])
def test_structure_actual_text_replaces_descendants_once(
    replacement: bytes, child_replacement: bytes | None
) -> None:
    with open_pdf(internal_nested_tagged_pdf(replacement, child_replacement)) as document:
        captured = capture_page(document.pages[0])
        expected = ("L", replacement.decode(), "R") if replacement else ("L", "R")

        assert captured.observations.text == expected
        assert captured.evidence.glyphs.actual_text_characters == len(replacement)
        assert "".join(document.extract().text.split()) == "".join(expected)


def test_structure_actual_text_keeps_geometry_and_clusters_across_child_elements() -> None:
    with open_pdf(internal_nested_tagged_pdf(b"Joined")) as document:
        page = document.pages[0]
        source = page.get_page_program().runs
        assert tuple(run.text for run in source) == ("L", "A", "B", "R")
        covered = source[1:3]
        expected_boxes = tuple((run.x0, run.y0, run.x1, run.y1) for run in covered)
        expected_clusters = tuple(cluster for run in covered for cluster in run.glyph_clusters)
        first_baseline, last_baseline = covered[0].baseline, covered[-1].baseline
        assert first_baseline is not None
        assert last_baseline is not None

        left, replacement, right = internal_apply_structure_actual_text(page, source)

        assert left is source[0]
        assert right is source[-1]
        assert replacement.text == "Joined"
        assert (replacement.x0, replacement.y0, replacement.x1, replacement.y1) == bbox_union(
            expected_boxes
        )
        assert replacement.advance_bbox == bbox_union(run.advance_bbox for run in covered)
        assert replacement.ink_bbox == bbox_union(run.ink_bbox for run in covered)
        assert replacement.baseline == (*first_baseline[:2], *last_baseline[2:])
        assert replacement.glyph_clusters == expected_clusters
        assert tuple(run.text for run in source) == ("L", "A", "B", "R")
        assert tuple((run.x0, run.y0, run.x1, run.y1) for run in covered) == expected_boxes


@pytest.mark.parametrize("replacement", [None, b"Joined"])
def test_structure_actual_text_stops_at_repeated_parent_dictionary(
    replacement: bytes | None,
) -> None:
    with open_pdf(internal_nested_tagged_pdf(replacement, parent_cycle=True)) as document:
        captured = capture_page(document.pages[0])

        assert captured.observations.text == (
            ("L", "Joined", "R") if replacement else ("L", "A", "B", "R")
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

# SPDX-License-Identifier: AGPL-3.0-only
"""Display rotation must move every piece of source geometry together."""

import pytest

from core_pdf.impl._impl.model.geometry import rotate_page_runs
from core_pdf.impl._impl.model.glyphs import GlyphCluster, GlyphObservation
from core_pdf.impl._impl.model.runs import TextRun
from tests.helpers.pdf_bytes import one_page_pdf, open_pdf


@pytest.mark.parametrize(
    ("rotation", "origin"),
    [(0, (10.0, 20.0)), (90, (20.0, 190.0)), (180, (190.0, 280.0)), (270, (280.0, 10.0))],
)
def test_verified_rotated_pdf_keeps_baselines_and_clusters_at_the_displayed_origin(
    rotation: int, origin: tuple[float, float]
) -> None:
    # These exact PDFs passed qpdf 12.3.2; MuPDF 1.28.2 independently reports
    # these glyph origins after page.rotation_matrix and a top-to-bottom flip.
    # Poppler 26.07.0 pdftotext -bbox and a 72dpi raster also confirm A at
    # displayed x=20, y=10 from the top on the 90-degree page.
    pdf = one_page_pdf(
        b"BT /F1 12 Tf 10 20 Td (A) Tj ET",
        media_box=(0, 0, 200, 300),
        page_extra=f"/Rotate {rotation}".encode(),
    )
    with open_pdf(pdf) as document:
        page = document.pages[0]
        source = page.chars[0]
        displayed = page.display_chars[0]

        assert displayed.text == "A"
        assert displayed.baseline is not None
        assert displayed.baseline[:2] == pytest.approx(origin)
        assert (displayed.tx, displayed.ty) == pytest.approx(origin)
        assert len(displayed.glyph_clusters) == len(source.glyph_clusters) == 1
        cluster = displayed.glyph_clusters[0]
        assert cluster.baseline == displayed.baseline
        assert cluster.glyphs[0].baseline == displayed.baseline
        assert cluster.glyphs[0].rotation_angle == (-rotation) % 360
        assert source.baseline is not None
        assert source.baseline[:2] == pytest.approx((10.0, 20.0))


def test_rotation_preserves_precise_boxes_and_transforms_paint_and_clip_evidence() -> None:
    baseline = (10.0, 20.0, 30.0, 20.0)
    advance = (10.0, 18.0, 30.0, 28.0)
    ink = (11.0, 20.0, 29.0, 27.0)
    provenance = (("clip_bbox", (0.0, 0.0, 100.0, 100.0)), ("layout_form_bbox", None))
    glyph = GlyphObservation(
        "A",
        ink,
        advance,
        7,
        baseline=baseline,
        glyph_transform=(2.0, 0.0, 0.0, 3.0, 10.0, 20.0),
        provenance=provenance,
    )
    cluster = GlyphCluster(0, "A", (glyph,), advance, ink, baseline, 1.0)
    run = TextRun(
        "A",
        10.0,
        15.0,
        30.0,
        30.0,
        10.0,
        20.0,
        12.0,
        3.0,
        0,
        0,
        0,
        advance_bbox=advance,
        ink_bbox=ink,
        baseline=baseline,
        glyph_clusters=(cluster,),
        provenance=provenance,
    )

    result = rotate_page_runs([run], rotate=90, page_width=200, page_height=300)[0]
    rotated_cluster = result.glyph_clusters[0]
    rotated_glyph = rotated_cluster.glyphs[0]

    assert (result.x0, result.y0, result.x1, result.y1) == (15.0, 170.0, 30.0, 190.0)
    assert (
        result.advance_bbox
        == rotated_cluster.advance_bbox
        == rotated_glyph.advance_bbox
        == (
            18.0,
            170.0,
            28.0,
            190.0,
        )
    )
    assert (
        result.ink_bbox
        == rotated_cluster.ink_bbox
        == rotated_glyph.ink_bbox
        == (
            20.0,
            171.0,
            27.0,
            189.0,
        )
    )
    assert (
        result.baseline
        == rotated_cluster.baseline
        == rotated_glyph.baseline
        == (
            20.0,
            190.0,
            20.0,
            170.0,
        )
    )
    assert rotated_glyph.glyph_transform == (0.0, -2.0, 3.0, 0.0, 20.0, 190.0)
    assert dict(result.provenance)["clip_bbox"] == (0.0, 100.0, 100.0, 200.0)
    assert result.provenance == rotated_glyph.provenance
    assert result is not run
    assert rotated_cluster is not cluster
    assert rotated_glyph is not glyph
    assert glyph.baseline == run.baseline == baseline
    assert glyph.ink_bbox == run.ink_bbox == ink
    assert glyph.provenance == run.provenance == provenance

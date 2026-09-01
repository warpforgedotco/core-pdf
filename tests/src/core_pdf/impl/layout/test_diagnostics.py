from core_pdf.impl.layout.diagnostics import (
    text_run_geometry_issues,
)
from core_pdf.impl.model.glyphs import GlyphCluster
from core_pdf.impl.model.runs import TextRun
from tests.helpers.extract_fakes import text_run as make_text_run


def text_run() -> TextRun:
    return make_text_run("A", 0.0, 0.0, 10.0, 10.0, tx=0.0, ty=0.0)


def recycle(existing: TextRun, text: str = "B", x1: float = 11.0) -> TextRun:
    """Push a run back through the freelist the way alloc_prepared_run does."""
    return TextRun.reinit(
        existing,
        text,
        1.0,
        1.0,
        x1,
        11.0,
        0.0,
        0.0,
        10.0,
        4.0,
        1,
        1,
        1,
        None,
        False,
        0,
        True,
        False,
        1,
        None,
    )


def test_text_run_geometry_issue_cache_tracks_attributes_and_direct_coordinates() -> None:
    run = text_run()
    run.confidence = 0.2

    first = text_run_geometry_issues(run)
    assert text_run_geometry_issues(run) is first
    assert {issue.code for issue in first} == {"low_confidence_text_run"}

    run.confidence = 1.0
    after_confidence_change = text_run_geometry_issues(run)
    assert after_confidence_change is not first
    assert not after_confidence_change

    run.coords[TextRun.X1] = 0.0
    after_direct_coordinate_change = text_run_geometry_issues(run)
    assert after_direct_coordinate_change is not after_confidence_change
    assert "run_nonpositive_bbox" in {issue.code for issue in after_direct_coordinate_change}


def test_glyph_clusters_are_validated_against_canonical_advance_geometry() -> None:
    run = text_run()
    run.advance_bbox = (-5.0, -4.0, 5.0, 6.0)
    run.glyph_clusters = (GlyphCluster(0, "A", (), run.advance_bbox, run.advance_bbox, None, 1.0),)

    issues = text_run_geometry_issues(run)

    assert issues == ()


def test_recycled_text_run_does_not_reuse_diagnostics_from_its_previous_life() -> None:
    run = text_run()
    run.confidence = 0.2
    assert {issue.code for issue in text_run_geometry_issues(run)} == {"low_confidence_text_run"}

    recycled = recycle(run, text="B", x1=10.0)

    assert recycled is run
    assert text_run_geometry_issues(recycled) == ()

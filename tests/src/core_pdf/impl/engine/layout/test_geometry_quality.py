from core_pdf.impl.engine.layout.geometry_quality import (
    text_run_geometry_issues,
)
from core_pdf.impl.engine.layout.glyphs import GlyphCluster
from core_pdf.impl.engine.layout.models import TextRun, TrackedTextRun


def text_run() -> TextRun:
    return TextRun("A", 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 4.0, 0, 0, 0)


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

    assert "glyph_clusters_outside_run_bbox" not in {issue.code for issue in issues}


def test_untracked_text_run_does_not_pay_revision_bookkeeping() -> None:
    run = text_run()

    assert type(run) is TextRun
    run.confidence = 0.5
    run.visible = False
    assert run.internal_revision == 0


def test_storing_a_memo_promotes_the_run_and_later_writes_bump_the_revision() -> None:
    run = text_run()
    assert type(run) is TextRun

    text_run_geometry_issues(run)

    assert type(run) is TrackedTextRun
    assert isinstance(run, TextRun)

    before = run.internal_revision
    run.confidence = 0.25
    assert run.internal_revision == before + 1


def test_recycling_a_promoted_run_demotes_it_back_to_plain_textrun() -> None:
    run = text_run()
    text_run_geometry_issues(run)
    assert type(run) is TrackedTextRun

    recycled = recycle(run)

    assert recycled is run
    assert type(recycled) is TextRun
    assert recycled.internal_geometry_issues_cache is None

    revision = recycled.internal_revision
    recycled.confidence = 0.75
    assert recycled.internal_revision == revision


def test_recycling_invalidates_memo_keys_captured_in_a_previous_life() -> None:
    run = text_run()
    stale_key = (run.internal_revision, tuple(run.coords))

    recycled = recycle(run, text="A", x1=10.0)
    recycled.coords[TextRun.X0] = 0.0
    recycled.coords[TextRun.Y0] = 0.0
    recycled.coords[TextRun.Y1] = 10.0
    recycled.coords[TextRun.TX] = 0.0
    recycled.coords[TextRun.TY] = 0.0
    recycled.coords[TextRun.FONT_SIZE] = 10.0
    recycled.coords[TextRun.SPACE_WIDTH] = 4.0

    assert tuple(recycled.coords) == stale_key[1]
    assert (recycled.internal_revision, tuple(recycled.coords)) != stale_key

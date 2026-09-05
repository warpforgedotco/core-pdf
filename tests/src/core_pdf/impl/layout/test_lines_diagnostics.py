from core_pdf.impl.layout.lines import (
    LayoutLine,
    page_layout_geometry_issues,
    text_run_geometry_issues,
)
from core_pdf.impl.model.glyphs import GlyphCluster
from core_pdf.impl.model.runs import TextRun
from tests.helpers.extract_fakes import text_run as make_text_run


def text_run() -> TextRun:
    return make_text_run("A", 0.0, 0.0, 10.0, 10.0, tx=0.0, ty=0.0)


def test_text_run_geometry_issues_reflect_attribute_changes() -> None:
    run = text_run()
    run.confidence = 0.2

    first = text_run_geometry_issues(run)
    assert {issue.code for issue in first} == {"low_confidence_text_run"}

    run.confidence = 1.0
    after_confidence_change = text_run_geometry_issues(run)
    assert not after_confidence_change

    run.x1 = 0.0
    after_coordinate_change = text_run_geometry_issues(run)
    assert "run_nonpositive_bbox" in {issue.code for issue in after_coordinate_change}


def test_glyph_clusters_are_validated_against_canonical_advance_geometry() -> None:
    run = text_run()
    run.advance_bbox = (-5.0, -4.0, 5.0, 6.0)
    run.glyph_clusters = (GlyphCluster(0, "A", (), run.advance_bbox, run.advance_bbox, None, 1.0),)

    issues = text_run_geometry_issues(run)

    assert issues == ()


def test_page_diagnostics_preserve_run_details_and_add_source_indexes() -> None:
    run = text_run()
    run.confidence = 0.2

    issues = page_layout_geometry_issues([LayoutLine(), LayoutLine([text_run(), run])])

    issue = next(issue for issue in issues if issue.code == "low_confidence_text_run")
    assert issue.details == (("confidence", 0.2), ("run_index", 1), ("line_index", 1))
    assert issue.bbox == (0.0, 0.0, 10.0, 10.0)
    assert issue.subject == "text_run"
    assert issue.severity == "warning"
    assert issue.repairable

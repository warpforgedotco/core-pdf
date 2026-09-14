"""Layout diagnostics distinguish malformed geometry from valid text controls."""

import pytest

from core_pdf.impl._impl.layout.lines import (
    LayoutLine,
    layout_line_geometry_issues,
    page_layout_geometry_issues,
    page_layout_geometry_summary,
    text_run_geometry_issues,
)
from core_pdf.impl._impl.model.glyphs import GlyphCluster
from core_pdf.impl._impl.model.runs import TextRun


def internal_run(text="A", bbox: tuple[float, float, float, float] = (0, 0, 10, 10), **kwargs):
    return TextRun(text, *bbox, 0, 0, 10, 3, 0, 0, 0, **kwargs)


@pytest.mark.parametrize("bbox", [(0, 0, 0, 10), (0, 0, 10, 0), (10, 0, 0, 10)])
@pytest.mark.parametrize("visible", [False, True])
@pytest.mark.parametrize("text", ["A", " "])
def test_visible_text_requires_positive_run_advance_and_ink_geometry(bbox, visible, text):
    issues = text_run_geometry_issues(internal_run(text, bbox, visible=visible))
    expected = (
        {"run_nonpositive_bbox", "run_nonpositive_advance_bbox", "run_nonpositive_ink_bbox"}
        if visible and text.strip()
        else set()
    )
    assert {issue.code for issue in issues} == expected
    assert all(not issue.repairable for issue in issues)


@pytest.mark.parametrize(
    ("text", "confidence", "expected"),
    [
        ("A", None, set()),
        ("A", 0.35, {"low_confidence_text_run"}),
        ("A", 0.351, set()),
        ("\ue000", 1, {"unsupported_text_run"}),
        ("\ue000", 0, {"unsupported_text_run", "low_confidence_text_run"}),
    ],
)
def test_run_diagnostics_identify_repairable_text_and_confidence_boundary(
    text, confidence, expected
):
    issues = text_run_geometry_issues(internal_run(text, confidence=confidence))
    assert {issue.code for issue in issues} == expected
    assert all(issue.repairable and issue.severity == "warning" for issue in issues)


@pytest.mark.parametrize(
    ("text", "confidence", "expected"),
    [
        ("A", None, None),
        ("A", 0.35, "low_confidence_repairable_glyph"),
        ("A", 0.36, None),
        ("!", 0.62, "low_confidence_repairable_glyph"),
        ("!", 0.621, None),
        ("\ue000", 1, "unsupported_glyph_cluster_text"),
        (" ", 0, None),
    ],
)
def test_cluster_diagnostics_apply_text_specific_confidence_thresholds(text, confidence, expected):
    cluster = GlyphCluster(17, text, (), (0, 0, 10, 10), (0, 0, 10, 10), None, confidence)
    issues = text_run_geometry_issues(internal_run(text, glyph_clusters=(cluster,)))
    assert [issue.code for issue in issues if issue.subject.startswith("glyph_cluster[")] == (
        [expected] if expected else []
    )
    for issue in issues:
        if issue.subject.startswith("glyph_cluster["):
            assert dict(issue.details)["cluster_id"] == 17
            assert issue.repairable


@pytest.mark.parametrize("bbox", [(0, 0, 0, 10), (float("nan"), 0, 10, 10), (30, 0, 40, 10)])
def test_cluster_geometry_errors_report_owner_and_text_mismatch(bbox):
    cluster = GlyphCluster(17, "B", (), bbox, bbox, None, 1)
    issues = text_run_geometry_issues(internal_run(glyph_clusters=(cluster,)))
    codes = {issue.code for issue in issues}
    assert "glyph_cluster_text_mismatch" in codes
    assert (
        "glyph_clusters_outside_advance_bbox" if bbox[0] == 30 else "glyph_cluster_nonpositive_bbox"
    ) in codes
    mismatch = next(issue for issue in issues if issue.code == "glyph_cluster_text_mismatch")
    assert dict(mismatch.details) == {"run_text": "A", "cluster_text": "B", "cluster_count": 1}


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_oversized_run_diagnostic_uses_writing_axis_and_advance_geometry(rotation):
    box = (0, 0, 10, 100) if rotation in (90, 270) else (0, 0, 100, 10)
    cluster = GlyphCluster(1, "A", (), (0, 0, 10, 10), (0, 0, 10, 10), None, 1)
    run = internal_run(
        bbox=box,
        advance_bbox=cluster.advance_bbox,
        glyph_clusters=(cluster,),
        rotation_angle=rotation,
    )
    issues = text_run_geometry_issues(run)
    assert [issue.code for issue in issues] == ["run_bbox_oversized_for_glyph_clusters"]
    assert dict(issues[0].details)["axis"] == ("y" if rotation in (90, 270) else "x")


def test_page_summary_retains_issue_locations_and_counts_severities():
    good = LayoutLine([internal_run()])
    bad = LayoutLine([internal_run("\ue000", (0, 0, 0, 10), confidence=0)])
    issues = page_layout_geometry_issues([good, bad])
    assert len(issues) == 6
    assert all(dict(issue.details)["line_index"] == 1 for issue in issues)
    assert all(
        dict(issue.details)["run_index"] == 0 for issue in issues if issue.subject != "layout_line"
    )
    summary = page_layout_geometry_summary([good, bad])
    assert (
        summary.issue_count,
        summary.error_count,
        summary.warning_count,
        summary.repairable_count,
    ) == (6, 3, 3, 2)
    assert (summary.line_count, summary.text_run_count) == (2, 2)
    assert layout_line_geometry_issues(good) == ()


def test_empty_line_has_neutral_bounds_and_no_diagnostics():
    line = LayoutLine()
    assert (line.x0, line.y0, line.x1, line.y1, line.height) == (0, 0, 0, 0, 0)
    assert line.text_and_words() == ("", ())
    assert layout_line_geometry_issues(line) == ()
    assert page_layout_geometry_summary([]).issue_count == 0

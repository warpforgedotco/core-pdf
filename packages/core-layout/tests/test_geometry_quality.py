from core_layout.impl.layout.geometry_quality import text_run_geometry_issues
from core_layout.impl.layout.models import TextRun


def text_run() -> TextRun:
    return TextRun("A", 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 4.0, 0, 0, 0)


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

from core_pdf.impl.capture_program import PageProgram
from core_pdf.impl.extract_capture import observations_from_runs, select_runs
from core_pdf.impl.extract_contracts import PageEvidence


def test_selection_without_text_keeps_the_runs_and_measures_an_empty_page() -> None:
    selection = select_runs((), PageProgram())
    assert selection.runs == ()
    assert not selection.trusted_hidden_text
    evidence = PageEvidence.measure(
        200.0, 100.0, observations_from_runs(selection.runs), PageProgram(), selection
    )
    assert evidence.page_area == 20_000.0
    assert evidence.native_characters == evidence.visible_native_characters == 0
    assert (evidence.image_count, evidence.text_coverage, evidence.full_page_image) == (
        0,
        0.0,
        False,
    )


def test_extending_evidence_copies_every_field_and_applies_changes() -> None:
    evidence = PageEvidence(100.0, 10, 8, 1, 2, 0.5, image_boxes=((0.0, 0.0, 1.0, 1.0),))
    assert evidence.extended(PageEvidence) == evidence
    changed = evidence.extended(PageEvidence, image_count=3)
    assert changed.image_count == 3
    assert changed.image_boxes == evidence.image_boxes
    assert changed.native_characters == evidence.native_characters

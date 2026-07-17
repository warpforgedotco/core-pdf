from core_pdf.impl.engine.extraction.common.render import layout_segment_observation
from core_pdf.impl.engine.layout.text_lines import LayoutLineTextSegment


def segment(*, bbox: tuple[float, float, float, float]) -> LayoutLineTextSegment:
    return LayoutLineTextSegment(
        text="Hello",
        separator_before=" ",
        spacing_decision="word_gap",
        bbox=bbox,
        advance_bbox=(1.0, 2.0, 5.0, 6.0),
        ink_bbox=(1.5, 2.5, 4.5, 5.5),
        baseline=(1.0, 2.0, 5.0, 2.0),
        writing_mode="horizontal",
        rotation_angle=0,
        provenance=(("glyph_cluster_id", 3),),
        confidence=0.9,
        visible=True,
    )


def test_layout_segment_observation_preserves_geometry_and_diagnostic_provenance() -> None:
    observation = layout_segment_observation(segment(bbox=(5.0, 6.0, 1.0, 2.0)), segment_index=4)

    assert observation is not None
    assert observation.bbox == (1.0, 2.0, 5.0, 6.0)
    assert observation.advance_bbox == (1.0, 2.0, 5.0, 6.0)
    assert observation.ink_bbox == (1.5, 2.5, 4.5, 5.5)
    assert observation.baseline == (1.0, 2.0, 5.0, 2.0)
    assert observation.text == "Hello"
    assert observation.confidence == 0.9
    assert observation.provenance == (
        ("glyph_cluster_id", 3),
        ("object_type", "LayoutLineTextSegment"),
        ("segment_index", 4),
        ("spacing_decision", "word_gap"),
        ("separator_before", " "),
        ("writing_mode", "horizontal"),
        ("rotation_angle", 0),
        ("visible", True),
    )


def test_layout_segment_observation_rejects_degenerate_geometry() -> None:
    assert layout_segment_observation(segment(bbox=(1.0, 2.0, 1.0, 6.0)), segment_index=0) is None

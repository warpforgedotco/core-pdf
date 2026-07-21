from core_ocr.impl import policy
from core_ocr.impl.policy import PageTextGeometryProfile


def test_portrait_raster_formula_noise_prefers_dense_table_route(
    monkeypatch,
) -> None:
    geometry = PageTextGeometryProfile(
        page_width=600,
        page_height=900,
        native_run_count=0,
        native_line_count=0,
        wide_line_ratio=0.0,
        short_line_ratio=0.0,
        centered_line_ratio=0.0,
        numeric_line_ratio=0.0,
        left_anchor_count=0,
        right_anchor_count=0,
        estimated_column_count=0,
        native_aligned_column_count=0,
        candidate_aligned_column_count=2,
        candidate_table_signals=10,
        candidate_schematic_signals=220,
        drawing_line_count=0,
        horizontal_rule_count=0,
        vertical_rule_count=0,
        dominant_image=True,
        occupied_area_ratio=0.0,
    )
    monkeypatch.setattr(policy, "page_text_geometry_profile", lambda *args, **kwargs: geometry)
    monkeypatch.setattr(policy, "formula_heavy_ocr_text", lambda text: True)

    classification = policy.classify_page_region("noisy formula text")

    assert classification.kind == "dense_table"


def test_landscape_raster_formula_noise_keeps_technical_route(
    monkeypatch,
) -> None:
    geometry = PageTextGeometryProfile(
        page_width=900,
        page_height=600,
        native_run_count=0,
        native_line_count=0,
        wide_line_ratio=0.0,
        short_line_ratio=0.0,
        centered_line_ratio=0.0,
        numeric_line_ratio=0.0,
        left_anchor_count=0,
        right_anchor_count=0,
        estimated_column_count=0,
        native_aligned_column_count=0,
        candidate_aligned_column_count=0,
        candidate_table_signals=0,
        candidate_schematic_signals=0,
        drawing_line_count=0,
        horizontal_rule_count=0,
        vertical_rule_count=0,
        dominant_image=True,
        occupied_area_ratio=0.0,
    )
    monkeypatch.setattr(policy, "page_text_geometry_profile", lambda *args, **kwargs: geometry)
    monkeypatch.setattr(policy, "formula_heavy_ocr_text", lambda text: True)

    classification = policy.classify_page_region("noisy formula text")

    assert classification.kind == "patent_formula"

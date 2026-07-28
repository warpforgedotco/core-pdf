from core_ocr.impl.full_page import should_try_vector_diagram_thresholding_ocr


def test_vector_text_table_uses_thresholding_only_at_highest_dpi() -> None:
    assert should_try_vector_diagram_thresholding_ocr(
        strategy="text_table",
        dpi=475,
        max_render_dpi=475,
        vector_diagram_sparse=True,
    )
    assert not should_try_vector_diagram_thresholding_ocr(
        strategy="text_table",
        dpi=300,
        max_render_dpi=475,
        vector_diagram_sparse=True,
    )


def test_general_vector_diagram_does_not_use_thresholding() -> None:
    assert not should_try_vector_diagram_thresholding_ocr(
        strategy="vector_or_table",
        dpi=475,
        max_render_dpi=475,
        vector_diagram_sparse=True,
    )

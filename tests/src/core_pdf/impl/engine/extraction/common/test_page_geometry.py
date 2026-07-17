from core_pdf.impl.engine.extraction.common.page_geometry import normalize_rect


def test_normalize_rect_reuses_ordered_internal_float_tuple() -> None:
    rect = (1.0, 2.0, 3.0, 4.0)

    assert normalize_rect(rect) is rect


def test_normalize_rect_orders_inverted_internal_float_tuple() -> None:
    assert normalize_rect((3.0, 4.0, 1.0, 2.0)) == (1.0, 2.0, 3.0, 4.0)


def test_normalize_rect_preserves_defensive_external_coercion() -> None:
    assert normalize_rect(["3", 4, "1", 2]) == (1.0, 2.0, 3.0, 4.0)
    assert normalize_rect((1.0, 2.0, object(), 4.0)) is None

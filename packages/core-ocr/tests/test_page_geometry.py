from dataclasses import replace

from core_ocr.impl.page_geometry import PageObservation, normalize_rect


def test_normalize_rect_reuses_ordered_internal_float_tuple() -> None:
    rect = (1.0, 2.0, 3.0, 4.0)

    assert normalize_rect(rect) is rect


def test_normalize_rect_orders_inverted_internal_float_tuple() -> None:
    assert normalize_rect((3.0, 4.0, 1.0, 2.0)) == (1.0, 2.0, 3.0, 4.0)


def test_normalize_rect_preserves_defensive_external_coercion() -> None:
    assert normalize_rect(["3", 4, "1", 2]) == (1.0, 2.0, 3.0, 4.0)
    assert normalize_rect((1.0, 2.0, object(), 4.0)) is None


def test_page_observation_is_compact_and_supports_immutable_replacement() -> None:
    observation = PageObservation(
        kind="native_line",
        source="native_text",
        bbox=(1.0, 2.0, 3.0, 4.0),
        text="original",
    )

    updated = replace(observation, text="updated")

    assert not hasattr(observation, "__dict__")
    assert updated.text == "updated"
    assert updated.bbox is observation.bbox

"""Unit tests for the pure helpers lifted out of ``RenderedPage.rasterize``.

These were closures until the rasterizer decomposition began, which made them
unreachable from tests and left the raster kernels at roughly half coverage.
Now that they are module-level, pin their edge cases directly.
"""

from __future__ import annotations

import numpy
import pytest

from core_pdf.impl.engine.layout.geometry import RectBox
from core_pdf.impl.engine.render.kernels import (
    RASTER_COORDINATE_CACHE_MAX_ENTRIES,
    internal_cached_raster_coordinates,
    internal_color_component,
    internal_fill_path_crossing_spans,
    internal_fill_path_sample_crossings,
    internal_fill_path_sample_crossings_numpy,
    internal_image_mask_decode_inverts,
    internal_image_quad,
    internal_image_raw_bytes,
    internal_intersect_box,
    internal_shading_color_rgba,
    internal_soft_mask_alpha_at,
    internal_translate_rect,
)


class TestIntersectBox:
    def test_overlapping_boxes_intersect(self) -> None:
        assert internal_intersect_box((0, 0, 10, 10), (5, 5, 20, 20)) == (5, 5, 10, 10)

    def test_disjoint_boxes_return_none(self) -> None:
        assert internal_intersect_box((0, 0, 4, 4), (5, 5, 9, 9)) is None

    def test_edge_contact_is_not_an_intersection(self) -> None:
        # Touching along x=4 has zero area, so it must not count as overlap.
        assert internal_intersect_box((0, 0, 4, 4), (4, 0, 8, 4)) is None

    def test_containment_returns_the_inner_box(self) -> None:
        assert internal_intersect_box((0, 0, 10, 10), (2, 3, 4, 5)) == (2, 3, 4, 5)


class TestTranslateRect:
    def test_rectbox_translates_and_keeps_its_metadata(self) -> None:
        source = RectBox(1.0, 2.0, 3.0, 4.0, seqno=7, fill=(1.0, 0.0, 0.0), fill_opacity=0.5)
        moved = internal_translate_rect(source, 10.0, 20.0)
        assert (moved.x0, moved.y0, moved.x1, moved.y1) == (11.0, 22.0, 13.0, 24.0)
        assert moved.seqno == 7
        assert moved.fill == (1.0, 0.0, 0.0)
        assert moved.fill_opacity == 0.5

    @pytest.mark.parametrize("factory", [tuple, list])
    def test_four_element_sequences_translate_to_tuples(self, factory: type) -> None:
        assert internal_translate_rect(factory([1, 2, 3, 4]), 1.0, -1.0) == (2.0, 1.0, 4.0, 3.0)

    def test_unrecognized_shapes_pass_through_untouched(self) -> None:
        sentinel = object()
        assert internal_translate_rect(sentinel, 5.0, 5.0) is sentinel
        assert internal_translate_rect((1, 2, 3), 5.0, 5.0) == (1, 2, 3)


class TestColorComponent:
    def test_unit_floats_scale_to_bytes(self) -> None:
        assert internal_color_component(0.0) == 0
        assert internal_color_component(1.0) == 255
        assert internal_color_component(0.5) == 128

    def test_out_of_range_values_clamp(self) -> None:
        assert internal_color_component(-3.0) == 0
        assert internal_color_component(9.0) == 255

    def test_bools_are_rejected_rather_than_treated_as_numbers(self) -> None:
        # bool is a subclass of int, so this guard is load-bearing.
        assert internal_color_component(True, default=42) == 42

    def test_non_numeric_values_fall_back_to_the_default(self) -> None:
        assert internal_color_component("nope", default=7) == 7
        assert internal_color_component(None, default=7) == 7


def test_shading_cmyk_clamps_components_before_conversion() -> None:
    assert internal_shading_color_rgba("DeviceCMYK", [-1.0, 0.5, 2.0, 0.0], 0.25) == (
        255,
        128,
        0,
        64,
    )


class TestImageRawBytes:
    def test_bytes_and_memoryview_are_returned_without_copying(self) -> None:
        raw = b"abc"
        assert internal_image_raw_bytes(raw) is raw
        view = memoryview(b"abc")
        assert internal_image_raw_bytes(view) is view

    def test_bytearray_is_wrapped_as_an_unsigned_byte_view(self) -> None:
        result = internal_image_raw_bytes(bytearray(b"abc"))
        assert isinstance(result, memoryview)
        assert bytes(result) == b"abc"


class TestImageMaskDecodeInverts:
    def test_descending_decode_array_inverts(self) -> None:
        assert internal_image_mask_decode_inverts([1, 0]) is True

    def test_ascending_decode_array_does_not_invert(self) -> None:
        assert internal_image_mask_decode_inverts([0, 1]) is False

    @pytest.mark.parametrize("value", [None, [], [1], "10", {"a": 1}, ["x", "y"]])
    def test_malformed_decode_arrays_do_not_invert(self, value: object) -> None:
        assert internal_image_mask_decode_inverts(value) is False


class TestSoftMaskAlphaAt:
    def test_absent_mask_is_fully_opaque(self) -> None:
        assert internal_soft_mask_alpha_at(None, 0.5, 0.5) == 255

    def test_samples_are_read_with_a_flipped_v_axis(self) -> None:
        # 2x2 mask laid out row-major from the top.
        mask = (bytes([10, 20, 30, 40]), 2, 2)
        assert internal_soft_mask_alpha_at(mask, 0.0, 1.0) == 10
        assert internal_soft_mask_alpha_at(mask, 0.9, 1.0) == 20
        assert internal_soft_mask_alpha_at(mask, 0.0, 0.0) == 30

    def test_coordinates_outside_the_mask_clamp_to_its_edge(self) -> None:
        mask = (bytes([10, 20, 30, 40]), 2, 2)
        assert internal_soft_mask_alpha_at(mask, 5.0, 0.0) == 40
        assert internal_soft_mask_alpha_at(mask, -5.0, 5.0) == 10

    def test_truncated_sample_buffer_reads_as_opaque(self) -> None:
        assert internal_soft_mask_alpha_at((b"", 2, 2), 0.5, 0.5) == 255


class TestImageQuad:
    def test_quad_key_is_read_directly(self) -> None:
        data = {"quad": [(0, 0), (1, 0), (1, 1), (0, 1)]}
        assert internal_image_quad(data) == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

    def test_quad_is_recovered_from_the_items_list(self) -> None:
        data = {"items": [("move", None), ("quad", [(2, 2), (3, 2), (3, 3)])]}
        assert internal_image_quad(data) == ((2.0, 2.0), (3.0, 2.0), (3.0, 3.0))

    def test_too_few_points_is_not_a_quad(self) -> None:
        assert internal_image_quad({"quad": [(0, 0), (1, 1)]}) is None
        assert internal_image_quad({"items": [("quad", [(0, 0)])]}) is None

    def test_missing_and_malformed_sources_return_none(self) -> None:
        assert internal_image_quad({}) is None
        assert internal_image_quad({"items": "not a list"}) is None
        assert internal_image_quad({"quad": [("x", "y"), (1, 1), (2, 2)]}) is None


class TestFillPathCrossings:
    #    A single edge from (0,0) to (0,10), scanned at y=5.
    EDGES = [(0.0, 0.0, 0.0, 10.0, 0.0, 10.0)]

    def test_a_scanline_inside_the_edge_range_crosses_once(self) -> None:
        assert internal_fill_path_sample_crossings(self.EDGES, 5.0) == [(0.0, 1)]

    def test_scanlines_outside_the_range_do_not_cross(self) -> None:
        assert internal_fill_path_sample_crossings(self.EDGES, 10.0) == []
        assert internal_fill_path_sample_crossings(self.EDGES, -1.0) == []

    def test_downward_edges_carry_the_opposite_winding_direction(self) -> None:
        downward = [(0.0, 10.0, 0.0, 0.0, 0.0, 10.0)]
        assert internal_fill_path_sample_crossings(downward, 5.0) == [(0.0, -1)]

    def test_numpy_scanlines_agree_with_the_scalar_path(self) -> None:
        edges = numpy.array(self.EDGES, dtype=numpy.float64)
        rows = internal_fill_path_sample_crossings_numpy(edges, numpy.array([5.0, 20.0]))
        assert rows[0] == [(0.0, 1)]
        assert rows[1] == []


class TestCrossingSpans:
    def test_no_crossings_produce_no_spans(self) -> None:
        assert internal_fill_path_crossing_spans([], "nonzero") == []

    def test_evenodd_pairs_sorted_crossings(self) -> None:
        crossings = [(3.0, 1), (1.0, 1), (8.0, 1), (5.0, 1)]
        assert internal_fill_path_crossing_spans(crossings, "evenodd") == [(1.0, 3.0), (5.0, 8.0)]

    def test_evenodd_drops_zero_width_spans(self) -> None:
        assert internal_fill_path_crossing_spans([(2.0, 1), (2.0, 1)], "evenodd") == []

    def test_nonzero_closes_a_span_where_winding_returns_to_zero(self) -> None:
        crossings = [(1.0, 1), (3.0, -1), (5.0, 1), (7.0, -1)]
        assert internal_fill_path_crossing_spans(crossings, "nonzero") == [(1.0, 3.0), (5.0, 7.0)]

    def test_coincident_crossings_are_folded_into_one_boundary(self) -> None:
        crossings = [(1.0, 1), (1.0, 1), (4.0, -1), (4.0, -1)]
        assert internal_fill_path_crossing_spans(crossings, "nonzero") == [(1.0, 4.0)]


class TestCachedRasterCoordinates:
    def test_coordinates_span_the_requested_range(self) -> None:
        cache: dict[tuple[int, int], numpy.ndarray] = {}
        assert list(internal_cached_raster_coordinates(cache, 2, 6)) == [2.0, 3.0, 4.0, 5.0]

    def test_repeat_requests_return_the_cached_array(self) -> None:
        cache: dict[tuple[int, int], numpy.ndarray] = {}
        first = internal_cached_raster_coordinates(cache, 0, 4)
        assert internal_cached_raster_coordinates(cache, 0, 4) is first

    def test_cache_cap_stops_growth_without_changing_results(self) -> None:
        cache: dict[tuple[int, int], numpy.ndarray] = {}
        for start in range(RASTER_COORDINATE_CACHE_MAX_ENTRIES + 10):
            internal_cached_raster_coordinates(cache, start, start + 1)

        assert len(cache) == RASTER_COORDINATE_CACHE_MAX_ENTRIES
        assert list(internal_cached_raster_coordinates(cache, 9_000, 9_003)) == [
            9000.0,
            9001.0,
            9002.0,
        ]
        assert len(cache) == RASTER_COORDINATE_CACHE_MAX_ENTRIES

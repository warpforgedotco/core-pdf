from __future__ import annotations

from collections.abc import Callable

import pytest

import core_pdf.impl.spec.s_09_fonts.font_program as font_program_module
from core_pdf._vendor.fontTools.cffLib import (
    cffExpertSubsetStrings,
    cffIExpertStrings,
    cffISOAdobeStrings,
    cffStandardStrings,
)
from core_pdf.impl.spec.s_09_fonts.font_program import (
    STANDARD_GLYPH_SIDS,
    CFFFont,
    internal_type2_glyph_geometry_impl,
)


def authoritative_standard_strings() -> list[str]:
    return list(cffStandardStrings)


def test_standard_glyph_sids_match_authoritative_cff_mapping() -> None:
    expected = {name: sid for sid, name in enumerate(authoritative_standard_strings())}

    assert expected == STANDARD_GLYPH_SIDS


@pytest.mark.parametrize(
    ("charset_id", "glyph_names"),
    [
        (0, cffISOAdobeStrings),
        (1, cffIExpertStrings),
        (2, cffExpertSubsetStrings),
    ],
)
def test_predefined_charsets_map_sids_to_gids_in_authoritative_order(
    charset_id: int, glyph_names: list[str]
) -> None:
    font = CFFFont(None)
    glyph_count = min(len(glyph_names), 12)

    assert font.internal_read_charset(charset_id, glyph_count) == {
        STANDARD_GLYPH_SIDS[name]: gid for gid, name in enumerate(glyph_names[:glyph_count])
    }


def test_predefined_expert_encoding_exposes_only_present_charset_names() -> None:
    font = CFFFont(None)
    font.charstrings = [b"\x0e"] * 5
    font.cid_to_gid = font.internal_read_charset(1, len(font.charstrings))
    font.top_dict = {16: [1.0]}

    assert font.builtin_encoding() == {
        32: "space",
        33: "exclamsmall",
        34: "Hungarumlautsmall",
        36: "dollaroldstyle",
    }


def test_sparse_nonstandard_builtin_encodings_remain_authoritative() -> None:
    font = CFFFont(None)

    assert not font.builtin_encoding_is_authoritative()

    font.top_dict = {16: [1.0]}
    assert font.builtin_encoding() == {}
    assert font.builtin_encoding_is_authoritative()

    font.top_dict = {16: [400.0]}
    assert font.builtin_encoding() == {}
    assert font.builtin_encoding_is_authoritative()

    font.is_cid_keyed = True
    assert not font.builtin_encoding_is_authoritative()


@pytest.mark.parametrize("charset_id", [0, 1, 2])
def test_cid_keyed_fonts_reject_predefined_charsets(charset_id: int) -> None:
    font = CFFFont(None)
    font.is_cid_keyed = True

    with pytest.raises(ValueError, match="predefined charset"):
        font.internal_read_charset(charset_id, 2)


def test_reserved_real_number_nibble_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid CFF real number"):
        CFFFont.internal_parse_real_number(bytes([0x1D, 0xFF]), 0)


@pytest.mark.parametrize(
    ("name", "sid"),
    [("sterling", 98), ("fi", 109), ("fl", 110), ("Semibold", 390)],
)
def test_standard_glyph_names_resolve_to_their_charset_glyph(name: str, sid: int) -> None:
    font = CFFFont(None)
    font.cid_to_gid = {sid: 7}

    assert font.glyph_id_for_name(name) == 7


def test_cff_geometry_is_derived_for_bbox_feature_and_bitmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    charstring = bytes([239, 239, 21, 239, 139, 5, 139, 239, 5, 39, 139, 5, 14])
    font = CFFFont(None)
    font.charstrings = [charstring]
    original = font_program_module.internal_type2_glyph_geometry_impl
    calls = 0

    def counting_geometry(
        value: bytes,
        *,
        local_subrs: tuple[bytes, ...],
        global_subrs: tuple[bytes, ...],
        seac_resolver: (
            Callable[
                [int, int, float, float],
                tuple[tuple[tuple[float, float], ...], ...],
            ]
            | None
        ) = None,
    ) -> tuple[list[list[tuple[float, float]]], tuple[float, float, float, float] | None]:
        nonlocal calls
        calls += 1
        return original(
            value,
            local_subrs=local_subrs,
            global_subrs=global_subrs,
            seac_resolver=seac_resolver,
        )

    monkeypatch.setattr(
        font_program_module, "internal_type2_glyph_geometry_impl", counting_geometry
    )

    assert font.glyph_bbox_for_gid(0) == (100.0, 100.0, 200.0, 200.0)
    assert font.glyph_feature(0).contours == 1
    assert font.glyph_bitmap_for_gid(0, width=4, height=4)
    assert font.normalized_glyph_contours(0) == font.glyph_contours_for_gid(0)
    assert calls == 5


def internal_type2_number(value: int) -> bytes:
    if -107 <= value <= 107:
        return bytes([value + 139])
    if 108 <= value <= 1131:
        encoded = value - 108
        return bytes([247 + encoded // 256, encoded % 256])
    if -1131 <= value <= -108:
        encoded = -value - 108
        return bytes([251 + encoded // 256, encoded % 256])
    assert -32768 <= value <= 32767
    return bytes([28]) + value.to_bytes(2, "big", signed=True)


def internal_type2_program(*tokens: int | tuple[int, ...]) -> bytes:
    return b"".join(
        bytes(token) if isinstance(token, tuple) else internal_type2_number(token)
        for token in tokens
    )


def internal_type2_geometry(
    *tokens: int | tuple[int, ...],
) -> tuple[list[list[tuple[float, float]]], tuple[float, float, float, float] | None]:
    return internal_type2_glyph_geometry_impl(
        internal_type2_program(*tokens),
        local_subrs=(),
        global_subrs=(),
    )


def internal_type2_calculated_line(
    *tokens: int | tuple[int, ...],
) -> list[tuple[float, float]]:
    contours, ignored_bbox = internal_type2_geometry(
        0,
        0,
        (21,),
        *tokens,
        0,
        (5,),
        (14,),
    )
    assert len(contours) == 1
    return contours[0]


def internal_type2_contour(
    operands: list[int], operator: int | tuple[int, int]
) -> list[tuple[float, float]]:
    encoded_operator = (operator,) if isinstance(operator, int) else operator
    contours, ignored_bbox = internal_type2_geometry(
        0, 0, (21,), *operands, encoded_operator, (14,)
    )
    assert len(contours) == 1
    return contours[0]


def test_rcurveline_draws_curves_before_its_final_line() -> None:
    contour = internal_type2_contour([10, 0, 10, 10, 10, 0, 5, -5], 24)

    assert contour[-2:] == [(30.0, 10.0), (35.0, 5.0)]


def test_rlinecurve_draws_all_lines_before_its_single_curve() -> None:
    contour = internal_type2_contour(
        [2, 0, 3, 0, 4, 0, 10, 0, 10, 10, 10, 0],
        25,
    )

    assert contour[1:4] == [(2.0, 0.0), (5.0, 0.0), (9.0, 0.0)]
    assert contour[-1] == (39.0, 10.0)


@pytest.mark.parametrize(
    ("operator", "operands", "expected_endpoint"),
    [
        (26, [5, 10, 10, 10, 10], (15.0, 30.0)),
        (27, [5, 10, 10, 10, 10], (30.0, 15.0)),
    ],
)
def test_optional_vv_hh_coordinate_belongs_to_the_first_curve(
    operator: int,
    operands: list[int],
    expected_endpoint: tuple[float, float],
) -> None:
    contour = internal_type2_contour(operands, operator)

    assert len(contour) >= 5
    assert contour[-1] == expected_endpoint


@pytest.mark.parametrize(
    ("escaped_operator", "operands", "expected_join", "expected_endpoint"),
    [
        (34, [10, 20, 5, 30, 40, 50, 60], (60.0, 5.0), (210.0, 0.0)),
        (
            35,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
            (9.0, 12.0),
            (36.0, 42.0),
        ),
        (36, [10, 1, 20, 2, 30, 40, 50, 3, 60], (60.0, 3.0), (210.0, 0.0)),
        (
            37,
            [1, 1, 3, 1, 5, 1, 7, 1, 9, 1, 20],
            (9.0, 3.0),
            (45.0, 0.0),
        ),
        (
            37,
            [1, 1, 1, 3, 1, 5, 1, 7, 1, 9, 20],
            (3.0, 9.0),
            (0.0, 45.0),
        ),
    ],
)
def test_type2_flex_operators_emit_both_curves(
    escaped_operator: int,
    operands: list[int],
    expected_join: tuple[float, float],
    expected_endpoint: tuple[float, float],
) -> None:
    contour = internal_type2_contour(operands, (12, escaped_operator))

    assert expected_join in contour
    assert contour[-1] == expected_endpoint


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        ((-12, (12, 9)), 12.0),
        ((7, 5, (12, 10)), 12.0),
        ((20, 8, (12, 11)), 12.0),
        ((24, 2, (12, 12)), 12.0),
        ((-12, (12, 14)), 12.0),
        ((3, 4, (12, 24)), 12.0),
        ((144, (12, 26)), 12.0),
        ((1, 2, (12, 3)), 1.0),
        ((0, 2, (12, 4)), 1.0),
        ((0, (12, 5)), 1.0),
        ((3, 3, (12, 15)), 1.0),
        ((12, 99, 1, 2, (12, 22)), 12.0),
    ],
)
def test_type2_arithmetic_and_conditional_operators_feed_path_construction(
    tokens: tuple[int | tuple[int, ...], ...], expected: float
) -> None:
    contour = internal_type2_calculated_line(*tokens)

    assert contour[-1] == (expected, 0.0)


def test_type2_transient_storage_round_trips_values() -> None:
    contour = internal_type2_calculated_line(42, 3, (12, 20), 3, (12, 21))

    assert contour[-1] == (42.0, 0.0)


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        ((6, (12, 27), (12, 10)), 12.0),
        ((20, 8, (12, 28), (12, 11)), -12.0),
        ((12, 99, (12, 18)), 12.0),
        ((4, 8, 1, (12, 29), (12, 10), (12, 28), (12, 18)), 12.0),
        ((1, 2, 3, 3, 1, (12, 30), (12, 11), (12, 11)), 4.0),
    ],
)
def test_type2_stack_operators_preserve_specified_order(
    tokens: tuple[int | tuple[int, ...], ...], expected: float
) -> None:
    contour = internal_type2_calculated_line(*tokens)

    assert contour[-1] == (expected, 0.0)


def test_type2_random_is_repeatable_and_in_the_specified_range() -> None:
    tokens = (
        0,
        0,
        (21,),
        (12, 23),
        100,
        (12, 24),
        0,
        (5,),
        (14,),
    )

    first = internal_type2_geometry(*tokens)
    second = internal_type2_geometry(*tokens)

    assert first == second
    endpoint = first[0][0][-1]
    assert 0.0 < endpoint[0] <= 100.0
    assert endpoint[1] == 0.0


@pytest.mark.parametrize(
    ("operands", "operator"),
    [
        ([10], (5,)),
        ([1, 2, 3, 4, 5, 6, 7], (8,)),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], (24,)),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], (25,)),
        ([1, 2, 3, 4, 5, 6], (26,)),
        ([1, 2, 3, 4, 5, 6], (27,)),
        ([1, 2, 3, 4, 5, 6], (30,)),
        ([1, 2, 3, 4, 5, 6, 7, 8], (12, 34)),
    ],
)
def test_type2_path_operators_reject_bad_arity_before_drawing(
    operands: list[int], operator: tuple[int, ...]
) -> None:
    contours, bbox = internal_type2_geometry(
        0,
        0,
        (21,),
        *operands,
        operator,
        (14,),
    )

    assert contours == []
    assert bbox is None


def test_type2_cubic_bbox_contains_exact_interior_extremum() -> None:
    contours, bbox = internal_type2_geometry(
        0,
        0,
        (21,),
        0,
        1000,
        0,
        -1000,
        100,
        0,
        (8,),
        (14,),
    )

    assert bbox == pytest.approx((0.0, 0.0, 100.0, 4000.0 / 9.0))
    assert max(y for contour in contours for ignored_x, y in contour) == pytest.approx(4000.0 / 9.0)


def test_top_dict_font_matrix_normalizes_shared_cff_geometry() -> None:
    charstring = bytes([239, 239, 21, 239, 139, 5, 139, 239, 5, 39, 139, 5, 14])
    default_font = CFFFont(None)
    default_font.charstrings = [charstring]
    transformed_font = CFFFont(None)
    transformed_font.charstrings = [charstring]
    transformed_font.top_dict = {(12, 7): [0.002, 0.001, 0.0, 0.003, 0.01, -0.02]}

    expected_contour = ((210.0, 380.0), (410.0, 480.0), (410.0, 780.0), (210.0, 680.0))
    for actual, expected in zip(
        transformed_font.glyph_contours_for_gid(0)[0], expected_contour, strict=True
    ):
        assert actual == pytest.approx(expected)
    assert transformed_font.glyph_bbox_for_gid(0) == pytest.approx((210.0, 380.0, 410.0, 780.0))
    assert transformed_font.glyph_bitmap_for_gid(0) != default_font.glyph_bitmap_for_gid(0)


def test_cid_font_dict_matrix_is_complete_when_top_dict_matrix_is_omitted() -> None:
    charstring = bytes([239, 239, 21, 239, 139, 5, 139, 239, 5, 39, 139, 5, 14])
    matrix = [0.002, 0.001, 0.0, 0.003, 0.01, -0.02]
    cid_font = CFFFont(None)
    cid_font.charstrings = [charstring]
    cid_font.fd_select = (0,)
    cid_font.font_dicts = ({(12, 7): matrix},)
    top_matrix_font = CFFFont(None)
    top_matrix_font.charstrings = [charstring]
    top_matrix_font.top_dict = {(12, 7): matrix}

    # PLRM 5.11 scales an FD matrix by 1000 when it inserts the omitted
    # 0.001 Top DICT matrix, so those operations cancel. The FD matrix is
    # therefore already the complete transform and must not be scaled again.
    assert cid_font.internal_font_matrix(0) == tuple(matrix)
    for actual, expected in zip(
        cid_font.glyph_contours_for_gid(0)[0],
        top_matrix_font.glyph_contours_for_gid(0)[0],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    assert cid_font.glyph_bbox_for_gid(0) == pytest.approx(top_matrix_font.glyph_bbox_for_gid(0))


def test_deprecated_endchar_seac_builds_standard_encoding_components() -> None:
    base = internal_type2_program(
        0,
        0,
        (21,),
        20,
        0,
        -10,
        20,
        -10,
        -20,
        (5,),
        (14,),
    )
    accent = internal_type2_program(
        0,
        0,
        (21,),
        4,
        0,
        -2,
        4,
        -2,
        -4,
        (5,),
        (14,),
    )
    composite = internal_type2_program(10, 30, 65, 194, (14,))
    font = CFFFont(None)
    font.charstrings = [b"\x0e", base, accent, composite]
    font.cid_to_gid = {
        STANDARD_GLYPH_SIDS["A"]: 1,
        STANDARD_GLYPH_SIDS["acute"]: 2,
    }

    contours = font.glyph_contours_for_gid(3)

    assert len(contours) == 2
    assert font.glyph_bbox_for_gid(3) == (0.0, 0.0, 20.0, 34.0)

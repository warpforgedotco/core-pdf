import pytest

from core_pdf.impl.fonts.font_program import (
    EMPTY_FEATURE,
    CFFFont,
    CFFGlyphFeature,
    CFFUnicodeRepairIndex,
    contours_bbox,
    cubic_sample_times,
    feature_from_contours,
    glyph_feature_distance,
    is_repairable_to_unicode_label,
    repair_candidate,
)


def make_font(payload: bytes, glyph_count: int = 4) -> CFFFont:
    font = CFFFont(None)
    font.data = b"\0\0\0" + payload
    font.charstrings = [b"\x0e"] * glyph_count
    return font


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"", {0: 0, 1: 1, 2: 2, 3: 3}),
        (b"\x09", {0: 0, 1: 1, 2: 2, 3: 3}),
        (b"\x00", {0: 0}),
        (b"\x00\x00\x2a\x00", {0: 0, 42: 1}),
        (b"\x00\x00\x2a\x00\x2a\x00\x2b", {0: 0, 42: 1, 43: 3}),
        (b"\x01", {0: 0}),
        (b"\x01\x00\x2a", {0: 0}),
        (b"\x01\x00\x2a\x09", {0: 0, 42: 1, 43: 2, 44: 3}),
        (b"\x02\x00\x2a\x00", {0: 0}),
        (b"\x02\x00\x2a\x00\x09", {0: 0, 42: 1, 43: 2, 44: 3}),
    ],
)
def test_charset_recovery_preserves_complete_entries(
    payload: bytes, expected: dict[int, int]
) -> None:
    assert make_font(payload).read_charset(3, 4) == expected


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_notdef_only_charsets(count: int) -> None:
    assert make_font(b"", count).read_charset(3, count) == ({0: 0} if count else {})


@pytest.mark.parametrize("offset", [0, 1, 2])
def test_cid_font_cannot_recover_a_predefined_charset(offset: int) -> None:
    font = make_font(b"")
    font.is_cid_keyed = True
    with pytest.raises(ValueError, match="predefined charset"):
        font.read_charset(offset, 4)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"", {}),
        (b"\x00", {}),
        (b"\x00\x03\x41", {65: 1}),
        (b"\x00\x04\x41\x41\x42\x43", {65: 1, 66: 3}),
        (b"\x01", {}),
        (b"\x01\x02\x41\x01\x50", {65: 1, 66: 2}),
        (b"\x01\x01\xfe\x03", {254: 1, 255: 2}),
        (b"\x01\x01\x41\x05", {65: 1, 66: 2, 67: 3}),
        (b"\x09", {}),
        (b"\x80\x01\x41", {65: 1}),
        (b"\x80\x01\x41\x01\x42\x00", {65: 1}),
        (b"\x80\x01\x41\x03\x42\x00\x2a\x43\x00\x2b\x44\x00\x2c", {65: 1, 66: 2}),
    ],
)
def test_encoding_recovery_bounds_glyphs_and_resolves_supplement_sids(
    payload: bytes, expected: dict[int, int]
) -> None:
    font = make_font(payload)
    font.cid_to_gid = {42: 2, 43: 8}
    assert font.read_encoding_codes(3) == expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"", (0, 0, 0, 0)),
        (b"\x00\x01\x02", (0, 0, 0, 0)),
        (b"\x00\x01\x02\x03\x04", (1, 2, 3, 4)),
        (b"\x09", (0, 0, 0, 0)),
        (b"\x03\x00", (0, 0, 0, 0)),
        (b"\x03\x00\x01\x00\x00", (0, 0, 0, 0)),
        (b"\x03\x00\x01\x00\x00\x02", (0, 0, 0, 0)),
        (b"\x03\x00\x02\x00\x00\x02\x00\x02\x03\x00\x09", (2, 2, 3, 3)),
    ],
)
def test_non_cid_fdselect_recovery_bounds_ranges(payload: bytes, expected: tuple[int, ...]) -> None:
    font = make_font(payload)
    font.top_dict = {(12, 37): [3]}
    assert font.read_fd_select() == expected


@pytest.mark.parametrize(
    ("label", "repairable"),
    [
        ("A", False),
        ("5", True),
        ("H", True),
        ("£", True),
        ("fi", False),
        ("ffi", False),
        ("ab", False),
        ("a ", False),
        ("a•", True),
        ("abcd", True),
        ("a!", True),
        ("", False),
    ],
)
def test_unicode_repair_preserves_letters_spaces_and_ligatures(
    label: str, repairable: bool
) -> None:
    assert is_repairable_to_unicode_label(label) is repairable


@pytest.mark.parametrize(
    ("label", "candidate", "distance", "expected"),
    [
        ("£", "L", 2.29, "L"),
        ("£", "L", 2.3, None),
        ("bad", "B", 2.29, "B"),
        ("5", "S", 1.89, "S"),
        ("5", "S", 1.9, None),
        ("H", "M", 1.79, "M"),
        ("H", "M", 1.8, None),
        ("5", "Z", 0.0, None),
        ("H", "N", 0.0, None),
    ],
)
def test_unicode_repair_requires_specific_confusion_and_strict_threshold(
    label: str, candidate: str, distance: float, expected: str | None
) -> None:
    feature = CFFGlyphFeature(((0, 0),), 1.0, 1)
    assert (
        repair_candidate(
            1, label, {1: feature, 2: feature}, {1: label, 2: candidate}, {2: distance}
        )
        == expected
    )


def test_unicode_repair_requires_a_better_shape_than_same_label() -> None:
    feature = CFFGlyphFeature(((0, 0),), 1.0, 1)
    features = dict.fromkeys(range(1, 7), feature)
    features[6] = EMPTY_FEATURE
    labels = {1: "5", 2: "5", 3: "Z", 4: "fi", 5: "!", 6: "A"}
    assert repair_candidate(1, "5", features, labels, {2: 1.0, 3: 0.64}) == "Z"
    assert repair_candidate(1, "5", features, labels, {2: 1.0, 3: 0.66}) is None
    assert repair_candidate(9, "£", features, labels) is None
    assert repair_candidate(1, "5", features, {1: "5", 2: "5"}) is None


def test_glyph_features_are_translation_invariant_and_detect_shape_changes() -> None:
    rectangle = [[(0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)]]
    translated = [[(x + 123, y - 456) for x, y in rectangle[0]]]
    first = feature_from_contours(rectangle)
    assert first == feature_from_contours(translated)
    assert first.aspect == 0.5
    assert first.contours == 1
    assert first.cells == ((0, 0), (0, 23), (17, 0), (17, 23))
    assert glyph_feature_distance(first, first) == 0.0
    triangle = feature_from_contours([[(0, 0), (10, 0), (5, 20)]])
    assert glyph_feature_distance(first, triangle) > 0.0
    assert feature_from_contours([]) == EMPTY_FEATURE
    assert feature_from_contours([[]]) == EMPTY_FEATURE
    assert contours_bbox(()) is None
    assert contours_bbox(tuple(tuple(c) for c in translated)) == (123, -456, 133, -436)


def test_cubic_flattening_keeps_extrema_and_refines_curves() -> None:
    assert cubic_sample_times((0, 0), (1, 1), (2, 2), (3, 3)) == (1.0,)
    assert cubic_sample_times((0, 0), (0, 0), (0, 0), (0, 0)) == (1.0,)
    times = cubic_sample_times((0, 0), (0, 100), (100, 100), (100, 0))
    assert 0.5 in times
    assert times == tuple(sorted(set(times)))
    assert len(times) > 4
    assert 0 < times[0] < times[-1] == 1


class FeatureFont(CFFFont):
    def glyph_feature(self, glyph_id: int) -> CFFGlyphFeature:
        return CFFGlyphFeature(((0, 0), (17, 23)), 1.0, 1)


@pytest.mark.parametrize("count", [2, 33])
def test_unicode_index_scalar_and_matrix_matching_preserve_code_identity(count: int) -> None:
    font = FeatureFont(None)
    font.charstrings = [b"\x0e"] * (count + 1)
    font.cid_to_gid = {gid: gid for gid in range(count + 1)}
    items = tuple((bytes([gid]), gid, "S" if gid == 1 else "5") for gid in range(1, count + 1))
    index = CFFUnicodeRepairIndex(font, (*items, (b"alias", 2, "5"), (b"invalid", 999, "£")))
    assert index.repairs_for_codes([]) == {}
    assert index.repairs_for_codes([b"unknown", b"\x01"]) == {}
    assert index.repairs_for_codes([b"\x02", b"alias", b"\x02", b"invalid"]) == {
        b"\x02": "S",
        b"alias": "S",
    }
    assert index.repairs_for_codes([code for code, _, _ in items]) == {
        code: "S" for code, _, _ in items if code != b"\x01"
    }


def test_unicode_index_with_no_glyphs_has_no_repairs() -> None:
    index = CFFUnicodeRepairIndex(CFFFont(None), ((b"a", 0, "£"),))
    assert index.repairs_for_codes([b"a"]) == {}


@pytest.mark.parametrize("ending", [b"\x0e", b""])
def test_glyph_outline_and_bounds_agree_for_terminated_and_unterminated_paths(
    ending: bytes,
) -> None:
    font = CFFFont(None)
    font.charstrings = [bytes([139, 139, 21, 149, 139, 139, 159, 129, 139, 5]) + ending]
    assert font.glyph_bbox_for_gid(0) == (0, 0, 10, 20)
    assert contours_bbox(font.normalized_glyph_contours(0)) == (0, 0, 10, 20)
    assert font.glyph_feature(0).aspect == 0.5
    assert any(font.glyph_bitmap_for_gid(0))
    assert font.glyph_bbox_for_gid(99) is None
    assert font.glyph_feature(99) == EMPTY_FEATURE
    assert font.glyph_bitmap_for_gid(99) == ()


def test_malformed_charstring_retains_only_completed_contours() -> None:
    font = CFFFont(None)
    font.charstrings = [bytes([139, 139, 21, 149, 159, 5, 149, 149, 21, 0])]
    assert font.glyph_bbox_for_gid(0) == (0, 0, 10, 20)
    assert font.normalized_glyph_contours(0) == (((0, 0), (10, 20)),)


@pytest.mark.parametrize("matrix", [[0.002, 0, 0, 0.003, 0, 0], [0, 0.001, -0.001, 0, 0, 0]])
def test_transformed_curve_bounds_match_flattened_outline(matrix: list[float]) -> None:
    font = CFFFont(None)
    font.charstrings = [bytes([139, 139, 21, 139, 239, 239, 139, 139, 39, 8, 14])]
    font.top_dict = {(12, 7): matrix}
    assert font.glyph_bbox_for_gid(0) == contours_bbox(font.normalized_glyph_contours(0))
    expected = (0, 0, 200, 225) if matrix[0] else (-75, 0, 0, 100)
    assert font.glyph_bbox_for_gid(0) == pytest.approx(expected)


@pytest.mark.parametrize("private", [[], [1], [-1, 3], [2, -1], [99, 3], [0, 3, 8]])
def test_invalid_private_dictionary_recovers_without_subroutines(private: list[float]) -> None:
    assert make_font(b"\x0e").read_private_subrs({18: private}) == []


def test_private_subroutines_use_private_relative_offset() -> None:
    font = make_font(b"\x8d\x13\x00\x01\x01\x01\x02\x0b")
    assert font.read_private_subrs({18: [2, 3]}) == [b"\x0b"]
    assert font.read_private_subrs({18: [2, 3, 99]}) == [b"\x0b"]
    font.data = font.data[:-1]
    assert font.read_private_subrs({18: [2, 3]}) == []


@pytest.mark.parametrize("offset", [-1, 3.5, float("nan"), float("inf"), -float("inf"), 99])
def test_invalid_font_dictionary_offsets_recover_without_entries(offset: float) -> None:
    font = make_font(b"")
    font.is_cid_keyed = True
    font.top_dict = {(12, 36): [offset]}
    assert font.read_font_dicts() == ()


def test_font_dictionary_recovery_preserves_indices_after_a_bad_entry() -> None:
    font = make_font(b"\x00\x03\x01\x01\x02\x02\x04\x1c\x8b\x11")
    font.is_cid_keyed = True
    font.top_dict = {(12, 36): [3]}
    assert font.read_font_dicts() == ({}, {}, {17: [0.0]})
    font.top_dict = {}
    assert font.read_font_dicts() == ()
    font.is_cid_keyed = False
    assert font.read_font_dicts() == ()


@pytest.mark.parametrize(("offset", "count"), [(0, 229), (1, 166), (2, 87)])
def test_predefined_charset_recovery_does_not_invent_extra_names(offset: int, count: int) -> None:
    font = make_font(b"")
    charset = font.read_charset(offset, count + 5)
    assert len(charset) == count
    assert charset[0] == 0
    assert set(charset.values()) == set(range(len(charset)))


def test_builtin_encoding_distinguishes_implicit_standard_and_explicit_expert() -> None:
    from core_pdf.impl.fonts.font_program import STANDARD_GLYPH_SIDS

    font = make_font(b"")
    assert font.builtin_encoding() == {}
    assert not font.builtin_encoding_is_authoritative()
    font.top_dict = {16: [1]}
    font.cid_to_gid = {STANDARD_GLYPH_SIDS["space"]: 1}
    assert font.builtin_encoding() == {32: "space"}
    assert font.builtin_encoding_is_authoritative()
    font.is_cid_keyed = True
    assert font.builtin_encoding() == {}
    assert not font.builtin_encoding_is_authoritative()


def test_custom_encoding_skips_notdef_and_unknown_names() -> None:
    font = make_font(b"\x00\x03ABC")
    font.top_dict = {16: [3]}
    font.cid_to_gid = {0: 1, 999: 2, 391: 3}
    font.custom_string_sids = {"custom": 391}
    assert font.builtin_encoding() == {67: "custom"}
    assert font.builtin_encoding_is_authoritative()


def test_negative_glyph_subroutine_lookup_uses_default_font_dictionary() -> None:
    font = make_font(b"")
    font.local_subrs = ((b"first",), (b"last",))
    font.fd_select = (0, 1)
    assert font.local_subrs_for_glyph(-1) == (b"first",)
    assert font.local_subrs_for_glyph(-9) == (b"first",)
    assert font.local_subrs_for_glyph(99) == (b"first",)


def test_invalid_top_matrix_recovers_using_child_matrix() -> None:
    font = make_font(b"")
    font.top_dict = {(12, 7): [1, 2]}
    font.font_dicts = ({(12, 7): [0.002, 0, 0, 0.003, 0, 0]},)
    assert tuple(font.font_matrix(0)) == (0.002, 0, 0, 0.003, 0, 0)
    font.top_dict = {(12, 7): [2, 0, 0, 3, 0, 0]}
    assert tuple(font.font_matrix(0)) == pytest.approx((0.004, 0, 0, 0.009, 0, 0))


def test_accent_components_are_translated_before_bounds_and_rasterization() -> None:
    from core_pdf.impl.fonts.font_program import STANDARD_GLYPH_SIDS

    font = make_font(b"")
    base = bytes([139, 139, 21, 149, 159, 5, 14])
    accent = bytes([139, 139, 21, 144, 144, 5, 14])
    composite = bytes([169, 179, 204, 205, 14])
    font.charstrings = [b"\x0e", base, accent, composite]
    font.cid_to_gid = {STANDARD_GLYPH_SIDS["A"]: 1, STANDARD_GLYPH_SIDS["B"]: 2}
    assert font.normalized_glyph_contours(3) == (((0, 0), (10, 20)), ((30, 40), (35, 45)))
    assert font.glyph_bbox_for_gid(3) == (0, 0, 35, 45)
    assert font.seac_contours(-1, 999, 0, 0) == ()
    assert font.seac_contours(67, 68, 0, 0) == ()
    font.is_cid_keyed = True
    assert font.normalized_glyph_contours(3) == ()


def test_random_charstring_geometry_is_repeatable() -> None:
    font = make_font(b"")
    font.charstrings = [bytes([12, 23, 12, 23, 21, 149, 159, 5, 14])]
    first = font.glyph_bbox_for_gid(0)
    assert first == font.glyph_bbox_for_gid(0)
    assert first is not None
    x0, y0, x1, y1 = first
    assert 0 < x0 <= 1
    assert 0 < y0 <= 1
    assert x1 - x0 == 10
    assert y1 - y0 == 20


@pytest.mark.parametrize("data", [b"", b"\x01\x00\x04", b"\x02\x00\x04\x01"])
def test_invalid_cff_headers_fail_clearly(data: bytes) -> None:
    with pytest.raises(ValueError, match="invalid CFF font program"):
        CFFFont(data)


def test_minimal_cff_font_initializes_from_bytes() -> None:
    font = CFFFont(
        b"\x01\x00\x04\x01"
        b"\x00\x01\x01\x01\x02F"
        b"\x00\x01\x01\x01\x03\xa0\x11"
        b"\x00\x00\x00\x00"
        b"\x00\x01\x01\x01\x02\x0e"
    )
    assert font.charstrings == [b"\x0e"]
    assert font.glyph_bbox_for_gid(0) is None
    assert font.dict_offset(16, default=0) == 0
    with pytest.raises(ValueError, match="CharStrings offset"):
        font.dict_offset(99)


@pytest.mark.parametrize("payload", [b"\x03\x00", b"\x09"])
def test_cid_fdselect_recovery_after_strict_reader_rejects_table(payload: bytes) -> None:
    font = make_font(payload)
    font.is_cid_keyed = True
    font.top_dict = {(12, 37): [3]}
    assert font.read_fd_select() == (0, 0, 0, 0)


def test_charset_multiple_ranges_preserve_separate_cid_intervals() -> None:
    font = make_font(b"\x01\x00\x2a\x00\x00\x3c\x09")
    assert font.read_charset(3, 4) == {0: 0, 42: 1, 60: 2, 61: 3}


def test_glyph_feature_grid_handles_subunit_dimensions() -> None:
    feature = feature_from_contours([[(0, 0), (0.5, 0.5)]])
    assert feature.cells == ((0, 0), (8, 12))
    assert feature.aspect == 1.0

import numpy
import pytest

from core_pdf.impl.fonts.decoder import FontDecoder, internal_outline_arrays


@pytest.mark.parametrize("code", [-1, -99])
def test_negative_glyph_codes_produce_no_paintable_geometry(code):
    decoder = FontDecoder({"Subtype": "Type1", "BaseFont": "Helvetica"})
    assert decoder.glyph_name(code) == ".notdef"
    assert decoder.glyph_bbox(code) is None
    assert decoder.glyph_bitmap(code) == ()
    assert decoder.glyph_outline(code) == ()
    assert decoder.glyph_outline_arrays(code) is None


@pytest.mark.parametrize("matrix", [None, [], [1, 2, 3], [1, 0, 0, "bad", 0, 0]])
def test_unusable_font_matrix_uses_conventional_glyph_space(matrix):
    decoder = FontDecoder({"Subtype": "Type1", "BaseFont": "Helvetica", "FontMatrix": matrix})
    assert tuple(decoder.font_matrix) == (0.001, 0, 0, 0.001, 0, 0)


def test_explicit_font_matrix_retains_anisotropy_shear_and_translation():
    matrix = [0.002, 0.001, -0.003, 0.004, 2, -3]
    decoder = FontDecoder({"Subtype": "Type1", "BaseFont": "Helvetica", "FontMatrix": matrix})
    assert tuple(decoder.font_matrix) == tuple(matrix)


@pytest.mark.parametrize("empty", [False, True])
def test_outline_cache_keys_include_explicit_gid_and_text_and_cache_empty_results(
    monkeypatch, empty
):
    calls = []
    contours = () if empty else (((0.0, 0.0), (2.0, 3.0), (4.0, 0.0)),)

    def resolve(self, code, gid, text):
        calls.append((code, gid, text))
        return contours

    monkeypatch.setattr(FontDecoder, "internal_glyph_outline_uncached", resolve)
    decoder = FontDecoder({"Subtype": "Type1", "BaseFont": "Helvetica"})
    keys = [(65, None, "A"), (65, 0, "A"), (65, 0, "B"), (66, 0, "B")]
    for key in keys:
        arrays = decoder.glyph_outline_arrays(*key)
        assert decoder.glyph_outline(*key) == contours
        assert decoder.glyph_outline_arrays(*key) is arrays
        if empty:
            assert arrays is None
        else:
            assert arrays is not None
            assert arrays.spans == ((0, 3),)
            numpy.testing.assert_array_equal(arrays.xs, [0, 2, 4])
            numpy.testing.assert_array_equal(arrays.ys, [0, 3, 0])
    assert calls == keys


@pytest.mark.parametrize("missing", [False, True])
def test_bbox_cache_keeps_missing_results_and_is_per_decoder(monkeypatch, missing):
    calls = []
    box = None if missing else (0.0, -2.0, 10.0, 8.0)

    def resolve(self, code):
        calls.append(code)
        return box

    monkeypatch.setattr(FontDecoder, "internal_glyph_bbox_uncached", resolve)
    for _ in range(2):
        decoder = FontDecoder({"Subtype": "Type1", "BaseFont": "Helvetica"})
        assert decoder.glyph_bbox(65) == box
        assert decoder.glyph_bbox(65) == box
    assert calls == [65, 65]


def test_linear_transform_cache_is_bounded_without_changing_geometry():
    arrays = internal_outline_arrays(((), ((9.0, 9.0),), ((1.0, 2.0), (3.0, 4.0))))
    assert arrays is not None
    assert arrays.spans == ((0, 2),)
    original = arrays.linear_columns(1, 2, 3, 4)
    assert arrays.linear_columns(1, 2, 3, 4) is original
    for scale in range(300):
        xs, ys = arrays.linear_columns(scale, -2, 3, 4)
        numpy.testing.assert_array_equal(xs, [scale + 6, 3 * scale + 12])
        numpy.testing.assert_array_equal(ys, [6, 10])
        assert len(arrays.linear) <= 256
    recomputed = arrays.linear_columns(1, 2, 3, 4)
    assert recomputed is not original
    numpy.testing.assert_array_equal(recomputed, original)
    numpy.testing.assert_array_equal(arrays.xs, [1, 3])
    numpy.testing.assert_array_equal(arrays.ys, [2, 4])


def test_failed_outline_lookup_is_retried_without_caching_partial_state(monkeypatch):
    calls = 0
    contours = (((0.0, 0.0), (1.0, 1.0)),)

    def resolve(self, code, gid, text):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("broken font")
        return contours

    monkeypatch.setattr(FontDecoder, "internal_glyph_outline_uncached", resolve)
    decoder = FontDecoder({"Subtype": "Type1", "BaseFont": "Helvetica"})
    with pytest.raises(ValueError, match="broken font"):
        decoder.glyph_outline_arrays(65)
    arrays = decoder.glyph_outline_arrays(65)
    assert arrays is not None
    assert decoder.glyph_outline_arrays(65) is arrays
    assert calls == 2


def test_missing_glyph_identity_does_not_request_fallback_outline(monkeypatch):
    monkeypatch.setattr(FontDecoder, "internal_resolve_glyph_id_for_code", lambda self, code: None)
    decoder = FontDecoder({"Subtype": "Type1", "BaseFont": "Helvetica"})
    assert decoder.glyph_id_for_code(65) is None
    assert decoder.glyph_outline(65, text="A") == ()
    assert decoder.glyph_bitmap(65) == ()

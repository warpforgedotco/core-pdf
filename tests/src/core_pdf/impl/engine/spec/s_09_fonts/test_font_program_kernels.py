from __future__ import annotations

from typing import Any, cast

import numpy
import pytest

from core_pdf.impl.engine.spec.s_09_fonts.cff import (
    CFFGlyphFeature,
    CFFUnicodeRepairIndex,
    glyph_feature_distance,
)
from core_pdf.impl.engine.spec.s_09_fonts.feature_distance_kernel import (
    feature_distance_matrix,
)


def test_feature_distance_matrix_matches_scalar_distance() -> None:
    features = (
        CFFGlyphFeature(((0, 0), (3, 4)), 1.0, 1, (1, 3, 7)),
        CFFGlyphFeature(((1, 1), (4, 5)), 1.2, 2, (1, 2, 8)),
        CFFGlyphFeature(((20, 30),), 1.5, 1, (1,)),
        CFFGlyphFeature((), 0.0, 0, ()),
    )
    matrix = feature_distance_matrix(
        [feature.cells for feature in features],
        [feature.bitmap for feature in features],
        [feature.aspect for feature in features],
        [feature.contours for feature in features],
        [feature.cells for feature in features],
        [feature.bitmap for feature in features],
        [feature.aspect for feature in features],
        [feature.contours for feature in features],
    )

    for left_index, left in enumerate(features):
        for right_index, right in enumerate(features):
            expected = glyph_feature_distance(left, right)
            if numpy.isinf(expected):
                assert numpy.isinf(matrix[left_index, right_index])
            else:
                assert matrix[left_index, right_index] == pytest.approx(expected)


def test_feature_distance_matrix_preserves_mismatched_bitmap_semantics() -> None:
    left = CFFGlyphFeature(((0, 0),), 1.0, 1, (15,))
    right = CFFGlyphFeature(((0, 0),), 1.0, 1, (15, 0))
    matrix = feature_distance_matrix(
        [left.cells],
        [left.bitmap],
        [left.aspect],
        [left.contours],
        [right.cells],
        [right.bitmap],
        [right.aspect],
        [right.contours],
    )

    assert matrix[0, 0] == pytest.approx(glyph_feature_distance(left, right))


def test_cff_unicode_repair_index_only_computes_requested_glyphs() -> None:
    feature = CFFGlyphFeature(((0, 0), (1, 1)), 1.0, 1, (3, 1))

    class FakeFont:
        charstrings = (b"", b"", b"")

        def __init__(self) -> None:
            self.feature_calls: list[int] = []

        def glyph_id_for_cid(self, cid: int) -> int:
            return cid

        def glyph_feature(self, gid: int) -> CFFGlyphFeature:
            self.feature_calls.append(gid)
            return feature

    font = FakeFont()
    index = CFFUnicodeRepairIndex(
        cast(Any, font),
        ((b"\x01", 1, "\ufffd"), (b"\x02", 2, "A")),
    )

    assert font.feature_calls == []
    assert index.repairs_for_codes((b"\x03",)) == {}
    assert font.feature_calls == []
    assert index.repairs_for_codes((b"\x01",)) == {b"\x01": "A"}
    assert sorted(font.feature_calls) == [1, 2]
    assert index.repairs_for_codes((b"\x01",)) == {b"\x01": "A"}
    assert sorted(font.feature_calls) == [1, 2]

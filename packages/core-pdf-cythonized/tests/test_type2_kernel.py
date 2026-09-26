# SPDX-License-Identifier: AGPL-3.0-only


import gzip
import pickle
from pathlib import Path

import pytest

from core_pdf_cythonized import type2_glyph_geometry

GOLDEN = pickle.loads(
    gzip.decompress((Path(__file__).parent / "type2_geometry_golden.pkl.gz").read_bytes())
)


def deep_repr(value: object) -> object:
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return [deep_repr(item) for item in value]
    return value


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 459
    modes = {(key[3], key[4]) for key, _ in GOLDEN}
    assert modes == {(False, False), (False, True), (True, True)}
    lengths = [len(key[0]) for key, _ in GOLDEN]
    assert min(lengths) <= 2
    assert max(lengths) >= 1500
    assert sum(len(contour) for _, (contours, _) in GOLDEN for contour in contours) > 25000


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_golden_vector(index: int) -> None:
    (charstring, local_subrs, global_subrs, flatten, retain), expected = GOLDEN[index]
    contours, bbox, seac, valid = type2_glyph_geometry(
        charstring, local_subrs, global_subrs, flatten, retain
    )
    if seac is not None:
        pytest.skip("composite glyph, completed by the caller")
    assert deep_repr(contours) == deep_repr(expected[0])
    assert deep_repr(bbox) == deep_repr(expected[1])
    assert valid is True


def test_bounds_only_matches_the_bbox_of_the_retained_contours():
    checked = 0
    for (charstring, local_subrs, global_subrs, _flatten, _retain), _ in GOLDEN:
        full = type2_glyph_geometry(charstring, local_subrs, global_subrs, False, True)
        bounds = type2_glyph_geometry(charstring, local_subrs, global_subrs, False, False)
        if full[2] is not None:
            continue
        assert deep_repr(bounds[1]) == deep_repr(full[1])
        assert bounds[0] == []
        checked += 1
    assert checked > 400


def test_an_invalid_charstring_reports_itself_rather_than_producing_geometry():
    contours, bbox, seac, valid = type2_glyph_geometry(bytes([12, 99]), (), ())
    assert valid is False
    assert contours == []
    assert bbox is None
    assert seac is None


def test_a_truncated_charstring_keeps_what_it_completed():
    program = bytes([139 + 10, 139 + 10, 21, 139 + 20, 139, 5, 14])
    contours, bbox, seac, valid = type2_glyph_geometry(program, (), ())
    assert valid is True
    assert contours
    assert bbox is not None


def test_subroutine_recursion_is_bounded():
    subr = bytes([139, 10, 11])
    contours, bbox, seac, valid = type2_glyph_geometry(bytes([139, 10, 11]), (subr,) * 108, ())
    assert valid is False

# SPDX-License-Identifier: AGPL-3.0-only

import gzip
import pickle
from pathlib import Path

import pytest

from core_pdf_cythonized import type1_glyph_bounds

GOLDEN_PATH = Path(__file__).parent / "type1_bounds_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))
CASES = [(font, glyph) for font in GOLDEN for glyph in font["glyphs"]]


def exact(bounds):
    return tuple(map(float.hex, bounds))


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 16
    assert len(CASES) == 496
    assert {glyph["origin"] for _, glyph in CASES} == {"corpus", "fuzz"}
    assert sum(glyph["compiled"] is None for _, glyph in CASES) == 120
    assert sum(glyph["expected"] is None for _, glyph in CASES) == 117
    assert sum(glyph["origin"] == "corpus" and glyph["compiled"] is None for _, glyph in CASES) == 2


@pytest.mark.parametrize("index", range(len(CASES)))
def test_kernel_reproduces_golden_vector_or_declines(index):
    font, glyph = CASES[index]
    bounds = type1_glyph_bounds(glyph["charstring"], font["subrs"], glyph["matrix"])
    if glyph["compiled"] is None:
        assert bounds is None
        return
    assert bounds is not None
    assert exact(bounds) == exact(glyph["compiled"])
    # Whatever the kernel accepts, fontTools drew to the same doubles.
    assert glyph["expected"] is not None
    assert exact(bounds) == exact(glyph["expected"])


IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def number(value):
    if -107 <= value <= 107:
        return bytes((value + 139,))
    if 108 <= value <= 1131:
        return bytes(((value - 108) // 256 + 247, (value - 108) % 256))
    return bytes(((-value - 108) // 256 + 251, (-value - 108) % 256))


def test_hsbw_moves_only_the_x_origin_and_lines_accumulate():
    program = number(10) + number(500) + b"\x0d" + number(0) + number(20) + b"\x15"
    program += number(100) + b"\x06" + number(50) + b"\x07" + b"\x09\x0e"
    assert type1_glyph_bounds(program, (), IDENTITY) == (10.0, 20.0, 110.0, 70.0)


def test_curve_extrema_widen_the_bounds_past_the_end_points():
    program = number(0) + number(0) + b"\x15"
    program += number(0) + number(100) + number(100) + number(0) + number(0) + number(-100)
    program += b"\x08\x09\x0e"
    assert type1_glyph_bounds(program, (), IDENTITY) == (0.0, 0.0, 100.0, 75.0)


def test_nothing_drawn_gives_empty_bounds_and_seac_is_declined():
    assert type1_glyph_bounds(number(0) + number(500) + b"\x0d\x0e", (), IDENTITY) == ()
    seac = b"".join(number(v) for v in (0, 10, 20, 65, 66)) + b"\x0c\x06"
    assert type1_glyph_bounds(seac, (), IDENTITY) is None


def test_unresolvable_subr_calls_are_declined_and_unknown_operators_stop_the_program():
    assert type1_glyph_bounds(number(5) + b"\x0a", (b"\x0b",), IDENTITY) is None
    assert type1_glyph_bounds(number(-1) + b"\x0a", (b"\x0b",), IDENTITY) is None
    stopped = number(0) + number(0) + b"\x15" + b"\x02" + number(100) + b"\x06"
    assert type1_glyph_bounds(stopped, (), IDENTITY) == (0.0, 0.0, 0.0, 0.0)

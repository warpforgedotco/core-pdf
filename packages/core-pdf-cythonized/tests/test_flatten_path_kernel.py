# SPDX-License-Identifier: AGPL-3.0-only

"""Path flattening: agreement with the capture code it replaced.

flatten_path_golden.pkl.gz was generated from the pre-change Python: the
recording module's flatten_path, CapturedPath.transformed, bbox,
has_segments and derived_lines, loaded from the last revision that had
them. The generator first proved that reference against 35,625 paths a
corpus sweep sent through the live capture code, point for point, and only
then used it. It holds:

- corpus: a spread of those paths from each of 94 documents, most heavily
  from the two -- PyMuPDF test_3806 and test_3362 -- that stroke hundreds of
  thousands of segments.
- synthetic: under five matrices including none and the identity, which
  differ on signed zeros, the rules a hand-written path can break -- a line
  after a close, a close with one point, a curve or line on an empty path, a
  rectangle beside open subpaths, a closing duplicate point, a move with
  nothing after it -- plus flatness and segment-count bounds, a scaled curve
  CTM, signed zeros in the box, an unknown operator, and the inputs the
  original raised on; and forty random paths.

Every value is compared by its bits. The commands are rebuilt from plain
tuples with stand-in classes, so this runs without core_pdf_spec installed.
"""

import gzip
import math
import pickle
import struct
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import flatten_path_commands

GOLDEN_PATH = Path(__file__).parent / "flatten_path_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


class Ctm:
    __slots__ = ("a", "b", "c", "d", "e", "f")

    def __init__(self, a, b, c, d, e, f):
        self.a, self.b, self.c, self.d, self.e, self.f = a, b, c, d, e, f


class Command:
    __slots__ = ("ctm", "flatness", "operands", "operator")

    def __init__(self, operator, operands, ctm, flatness):
        self.operator = operator
        self.operands = operands
        self.ctm = Ctm(*ctm)
        self.flatness = flatness


def commands(case):
    return [Command(*command) for command in case["commands"]]


def run(case):
    try:
        xs, ys, spans, bbox, has_segments, lines = flatten_path_commands(
            commands(case), case["matrix"], math.hypot
        )
    except Exception as error:
        return ("raises", type(error).__name__)
    return (
        xs.tolist(),
        ys.tolist(),
        spans,
        bbox,
        has_segments,
        [tuple(row) for row in lines.tolist()],
    )


def exact(value):
    if isinstance(value, float):
        return ("f", struct.pack("<d", value))
    if isinstance(value, (list, tuple)):
        return tuple(exact(item) for item in value)
    return value


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 675
    origins = [case["origin"] for case in GOLDEN]
    assert sum(origin.startswith("synthetic/") for origin in origins) == 130
    assert "corpus/test_3806.pdf" in origins
    assert "corpus/test_3362.pdf" in origins
    assert sum(case["expected"][0] == "raises" for case in GOLDEN) == 20
    assert {case["matrix"] is None for case in GOLDEN} == {True, False}


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_capture(index):
    case = GOLDEN[index]
    assert exact(run(case)) == exact(case["expected"])


def test_lines_come_from_consecutive_points_within_a_subpath():
    path = [
        Command("m", (0.0, 0.0), (1, 0, 0, 1, 0, 0), 0.0),
        Command("l", (10.0, 0.0), (1, 0, 0, 1, 0, 0), 0.0),
        Command("l", (10.0, 0.005), (1, 0, 0, 1, 0, 0), 0.0),
        Command("h", (), (1, 0, 0, 1, 0, 0), 0.0),
        Command("m", (50.0, 50.0), (1, 0, 0, 1, 0, 0), 0.0),
        Command("l", (50.0, 60.0), (1, 0, 0, 1, 0, 0), 0.0),
    ]
    _, _, spans, bbox, has_segments, lines = flatten_path_commands(path, None, math.hypot)
    # No line across the subpath break, none for the closing edge, none for
    # the 0.005 step.
    assert lines.tolist() == [[0.0, 0.0, 10.0, 0.0], [50.0, 50.0, 50.0, 60.0]]
    assert spans == [(0, 3, True), (3, 5, False)]
    assert bbox == (0.0, 0.0, 50.0, 60.0)
    assert has_segments


def test_matrix_applies_to_points_lines_and_box():
    path = [
        Command("m", (1.0, 2.0), (1, 0, 0, 1, 0, 0), 0.0),
        Command("l", (3.0, 2.0), (1, 0, 0, 1, 0, 0), 0.0),
    ]
    xs, ys, _, bbox, _, lines = flatten_path_commands(
        path, (2.0, 0.0, 0.0, -1.0, 10.0, 100.0), math.hypot
    )
    assert xs.tolist() == [12.0, 16.0]
    assert ys.tolist() == [98.0, 98.0]
    assert lines.tolist() == [[12.0, 98.0, 16.0, 98.0]]
    assert bbox == (12.0, 98.0, 16.0, 98.0)


def test_empty_and_pointless_paths():
    xs, ys, spans, bbox, has_segments, lines = flatten_path_commands([], None, math.hypot)
    assert (xs.size, ys.size, spans, bbox, has_segments, lines.shape) == (
        0,
        0,
        [],
        None,
        False,
        (0, 4),
    )
    move = [Command("m", (4.0, 5.0), (1, 0, 0, 1, 0, 0), 0.0)]
    _, _, spans, bbox, has_segments, lines = flatten_path_commands(move, None, math.hypot)
    assert spans == [(0, 1, False)]
    assert bbox == (4.0, 5.0, 4.0, 5.0)
    assert not has_segments
    assert lines.shape == (0, 4)


def test_segment_count_goes_through_the_hypot_it_is_given():
    calls = []

    def counting_hypot(x, y):
        calls.append((x, y))
        return math.hypot(x, y)

    curve = [Command("c", (0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0), (1, 0, 0, 1, 0, 0), 0.0)]
    flatten_path_commands(curve, None, counting_hypot)
    # Two for the CTM's scale, three for the control polygon.
    assert len(calls) == 5


def test_arrays_are_float64():
    path = [
        Command("m", (0.0, 0.0), (1, 0, 0, 1, 0, 0), 0.0),
        Command("l", (1.0, 1.0), (1, 0, 0, 1, 0, 0), 0.0),
    ]
    xs, ys, _, _, _, lines = flatten_path_commands(path, None, math.hypot)
    assert xs.dtype == ys.dtype == lines.dtype == numpy.float64

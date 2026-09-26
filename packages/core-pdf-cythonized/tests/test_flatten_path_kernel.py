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

Every value is compared by its bits. The commands are plain tuples, encoded
here the way core_pdf_spec's PdfPath stores them -- an operator byte each and
their numbers in one float array, a curve followed by its CTM's linear part
and its flatness -- so this runs without core_pdf_spec installed. PdfPath
cannot hold a command with the wrong number of operands, or operands that are
not numbers, so the golden inputs that raised for those are not encodable and
are checked to be exactly the ones that raised; an operator PdfPath does not
have was ignored, and is left out.
"""

import gzip
import math
import pickle
import struct
from array import array
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import flatten_path_commands

GOLDEN_PATH = Path(__file__).parent / "flatten_path_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


ARITY = {"m": 2, "l": 2, "re": 4, "c": 8}
CODES = {"m": b"m", "l": b"l", "h": b"h", "re": b"r", "c": b"c"}


def encode(commands):
    """(ops, coords) as PdfPath stores these commands, or None if it cannot."""
    ops = bytearray()
    coords = array("d")
    for operator, operands, ctm, flatness in commands:
        if operator == "h":
            ops += CODES["h"]
            continue
        if operator not in ARITY:
            continue
        if len(operands) != ARITY[operator]:
            return None
        try:
            values = array("d", operands)
            if operator == "c":
                flat = float(flatness) if flatness else 0.0
                values.extend((ctm[0], ctm[1], ctm[2], ctm[3], flat))
        except TypeError:
            return None
        ops += CODES[operator]
        coords.extend(values)
    return ops, coords


def flatten(commands, matrix, hypot=math.hypot):
    encoded = encode(commands)
    assert encoded is not None
    rows = array("d")
    flattened = flatten_path_commands(encoded[0], encoded[1], matrix, hypot, rows, 0.0)
    lines = numpy.frombuffer(rows, dtype=numpy.float64).reshape(-1, 5)[:, :4]
    return (*flattened, lines)


def identity(operator, operands):
    return (operator, operands, (1, 0, 0, 1, 0, 0), 0.0)


def run(case):
    encoded = encode(case["commands"])
    if encoded is None:
        return None
    rows = array("d", [7.0] * 5)
    try:
        xs, ys, spans, bbox, has_segments = flatten_path_commands(
            encoded[0], encoded[1], case["matrix"], math.hypot, rows, 2.5
        )
    except Exception as error:
        return ("raises", type(error).__name__)
    # The line table is appended to, never rewritten, and every row carries
    # the width it was given.
    assert rows[:5] == array("d", [7.0] * 5)
    lines = numpy.frombuffer(rows, dtype=numpy.float64).reshape(-1, 5)[1:]
    assert (lines[:, 4] == 2.5).all()
    lines = lines[:, :4]
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
    got = run(case)
    if got is None:
        assert case["expected"][0] == "raises"
        pytest.skip("PdfPath cannot hold this input")
    assert exact(got) == exact(case["expected"])


def test_only_inputs_that_raised_are_unencodable():
    unencodable = [case for case in GOLDEN if encode(case["commands"]) is None]
    assert unencodable
    assert all(case["expected"][0] == "raises" for case in unencodable)


def test_lines_come_from_consecutive_points_within_a_subpath():
    path = [
        identity("m", (0.0, 0.0)),
        identity("l", (10.0, 0.0)),
        identity("l", (10.0, 0.005)),
        identity("h", ()),
        identity("m", (50.0, 50.0)),
        identity("l", (50.0, 60.0)),
    ]
    _, _, spans, bbox, has_segments, lines = flatten(path, None)
    # No line across the subpath break, none for the closing edge, none for
    # the 0.005 step.
    assert lines.tolist() == [[0.0, 0.0, 10.0, 0.0], [50.0, 50.0, 50.0, 60.0]]
    assert spans == [(0, 3, True), (3, 5, False)]
    assert bbox == (0.0, 0.0, 50.0, 60.0)
    assert has_segments


def test_matrix_applies_to_points_lines_and_box():
    path = [
        identity("m", (1.0, 2.0)),
        identity("l", (3.0, 2.0)),
    ]
    xs, ys, _, bbox, _, lines = flatten(path, (2.0, 0.0, 0.0, -1.0, 10.0, 100.0))
    assert xs.tolist() == [12.0, 16.0]
    assert ys.tolist() == [98.0, 98.0]
    assert lines.tolist() == [[12.0, 98.0, 16.0, 98.0]]
    assert bbox == (12.0, 98.0, 16.0, 98.0)


def test_empty_and_pointless_paths():
    xs, ys, spans, bbox, has_segments, lines = flatten([], None)
    assert (xs.size, ys.size, spans, bbox, has_segments, lines.shape) == (
        0,
        0,
        [],
        None,
        False,
        (0, 4),
    )
    move = [identity("m", (4.0, 5.0))]
    _, _, spans, bbox, has_segments, lines = flatten(move, None)
    assert spans == [(0, 1, False)]
    assert bbox == (4.0, 5.0, 4.0, 5.0)
    assert not has_segments
    assert lines.shape == (0, 4)


def test_segment_count_goes_through_the_hypot_it_is_given():
    calls = []

    def counting_hypot(x, y):
        calls.append((x, y))
        return math.hypot(x, y)

    curve = [identity("c", (0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0))]
    flatten(curve, None, counting_hypot)
    # Two for the CTM's scale, three for the control polygon.
    assert len(calls) == 5


def test_arrays_are_float64():
    path = [
        identity("m", (0.0, 0.0)),
        identity("l", (1.0, 1.0)),
    ]
    xs, ys, _, _, _, lines = flatten(path, None)
    assert xs.dtype == ys.dtype == lines.dtype == numpy.float64


def test_no_line_table_collects_no_lines():
    encoded = encode([identity("m", (0.0, 0.0)), identity("l", (5.0, 0.0))])
    assert encoded is not None
    xs, _, _, _, has_segments = flatten_path_commands(
        encoded[0], encoded[1], None, math.hypot, None, 1.0
    )
    assert xs.tolist() == [0.0, 5.0]
    assert has_segments

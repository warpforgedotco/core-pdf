# SPDX-License-Identifier: AGPL-3.0-only

"""Outline edges: bitwise agreement with the numpy construction it replaced.

outline_golden.pkl.gz holds transformed glyph outlines captured from three
corpus pages, with the edge array the original produced -- a column_stack per
span, a separate 1x4 array per closing edge, and a concatenate over the lot.

The synthetic cases cover span shapes real glyphs do not produce: a ring whose
final point duplicates the first (dropped, no closing edge), an open ring (kept,
closing edge added), a bare two-point span, and spans too short to survive. The
drop happens before the length check, so a two-point span whose ends coincide
collapses to one point and is dropped rather than kept -- that ordering is the
easiest thing to get wrong in a port.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import outline_edges

GOLDEN_PATH = Path(__file__).parent / "outline_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))
REAL = GOLDEN["real"]
SYNTHETIC = GOLDEN["synthetic"]


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(REAL) == 150
    assert len(SYNTHETIC) == 6


@pytest.mark.parametrize("index", range(len(REAL)))
def test_kernel_reproduces_numpy_edges_bitwise(index):
    case = REAL[index]
    edges, kept, dropped = outline_edges(case["x"], case["y"], case["spans"])
    assert edges is not None
    assert edges.shape == case["edges"].shape
    assert numpy.array_equal(edges, case["edges"])


@pytest.mark.parametrize("index", range(len(SYNTHETIC)))
def test_synthetic_span_shapes(index):
    xs, ys, spans = SYNTHETIC[index]
    xs = numpy.ascontiguousarray(xs, dtype=numpy.float64)
    ys = numpy.ascontiguousarray(ys, dtype=numpy.float64)
    edges, kept, dropped = outline_edges(xs, ys, spans)
    if edges is None:
        assert not kept
        assert dropped
        return
    # Every kept span contributes one edge per segment, plus a closing edge
    # only when its endpoints differ after the duplicate-point drop.
    expected = sum((end - start - 1) + (1 if closes else 0) for start, end, closes in kept)
    assert edges.shape == (expected, 4)
    for start, end, closes in kept:
        assert end - start >= 2
        assert closes == (xs[start] != xs[end - 1] or ys[start] != ys[end - 1])


def test_a_two_point_span_with_equal_ends_is_dropped_not_kept():
    xs = numpy.array([4.0, 4.0])
    ys = numpy.array([9.0, 9.0])
    edges, kept, dropped = outline_edges(xs, ys, [(0, 2)])
    assert edges is None
    assert kept == []
    assert dropped


def test_render_commands_use_the_kernel():
    # Through translated_outline_edges, which runs outline_edges on the
    # translated columns.
    pytest.importorskip("core_pdf")
    from core_pdf.impl.render import commands
    from core_pdf_cythonized import translated_outline_edges

    assert commands.translated_outline_edges is translated_outline_edges

# SPDX-License-Identifier: AGPL-3.0-only

"""Signed-area coverage: the kernel must reproduce the numpy original exactly.

coverage_golden.pkl.gz was produced by the numpy implementation that this
kernel replaced, before that code was deleted. The cases are real edge arrays
captured while rasterizing three corpus pages, plus synthetic ones covering
branches real glyphs do not reach: empty input, purely horizontal edges (no
slope at all), purely vertical ones, spans clipped on both axes.

Equality here is bitwise, and that is not incidental. The original accumulated
through numpy.bincount over two concatenated index blocks, so every left-hand
weight landed before any right-hand weight. The kernel splits its accumulation
the same way; adding both weights per piece instead would still be a correct
rasterizer, and would round differently.
"""

import gzip
import pickle
from pathlib import Path

import numpy
import pytest

from core_pdf_cythonized import fill_glyph_coverage, signed_area_coverage

GOLDEN_PATH = Path(__file__).parent / "coverage_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 306
    assert any(case["edges"].shape[0] == 0 for case in GOLDEN)


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_numpy_output_bitwise(index):
    case = GOLDEN[index]
    got = signed_area_coverage(case["edges"], case["width"], case["height"])
    assert got.shape == case["expected"].shape
    assert numpy.array_equal(got, case["expected"])


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 4), (4, 0), (-1, 4), (4, -1)],
)
def test_degenerate_dimensions_yield_an_empty_buffer(width, height):
    edges = numpy.array([[0.0, 0.0, 4.0, 4.0]])
    out = signed_area_coverage(edges, width, height)
    assert out.shape == (max(height, 0), max(width, 0))


def test_coverage_is_bounded_to_unit_range():
    for case in GOLDEN:
        out = signed_area_coverage(case["edges"], case["width"], case["height"])
        assert out.size == 0 or (out.min() >= 0.0 and out.max() <= 1.0)


def test_render_target_uses_the_kernel():
    # These wire-up assertions need the consumer installed. The kernel tests
    # otherwise stand alone, so cibuildwheel can run the golden vectors
    # against a freshly built wheel with nothing else present.
    pytest.importorskip("core_pdf")
    from core_pdf.impl.render import paths, target

    assert target.fill_glyph_coverage is fill_glyph_coverage
    assert not hasattr(paths, "signed_area_coverage")
    # The device-space entry point is the one the golden vectors pin; core
    # reaches the shared core through the fused glyph front end instead.
    assert not hasattr(target, "signed_area_coverage")
    assert not hasattr(paths, "group_offsets")

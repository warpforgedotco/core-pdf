# SPDX-License-Identifier: AGPL-3.0-only

import gzip
import pickle
from pathlib import Path

import pytest

from core_pdf_cythonized import glyph_feature_cells

GOLDEN_PATH = Path(__file__).parent / "glyph_feature_cells_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 796
    origins = {case["origin"] for case in GOLDEN}
    assert origins == {"corpus/cff-repair", "corpus/outline", "fuzz", "declined"}
    assert sum(case["expected"] is None for case in GOLDEN) == 3
    assert sum(case["expected"] == () for case in GOLDEN) >= 2


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_golden_vector_or_declines(index):
    case = GOLDEN[index]
    assert glyph_feature_cells(case["contours"]) == case["expected"], case["origin"]


def test_cells_round_half_to_even():
    # 0.5 of a 17-cell span lands on 8.5, which rounds to 8, not 9.
    contours = (((0.0, 0.0), (8.5, 11.5), (17.0, 23.0)),)
    assert glyph_feature_cells(contours) == (((0, 0), (8, 12), (17, 23)), 17.0, 23.0)


def test_a_degenerate_extent_is_measured_as_one_unit():
    assert glyph_feature_cells((((3.0, 4.0), (3.0, 4.0)),)) == (((0, 0),), 1.0, 1.0)

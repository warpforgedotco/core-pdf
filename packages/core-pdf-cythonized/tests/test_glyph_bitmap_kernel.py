# SPDX-License-Identifier: AGPL-3.0-only


import gzip
import pickle
from pathlib import Path

import pytest

from core_pdf_cythonized import glyph_bitmap_rows

GOLDEN_PATH = Path(__file__).parent / "glyph_bitmap_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 1747
    origins = {case["origin"] for case in GOLDEN}
    assert origins == {"corpus/extract", "corpus/render", "fuzz", "declined"}
    assert sum(case["expected"] is None for case in GOLDEN) == 3
    assert any(case["expected"] and any(case["expected"]) for case in GOLDEN)


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_golden_vector_or_declines(index):
    case = GOLDEN[index]
    rows = glyph_bitmap_rows(case["contours"], case["width"], case["height"])
    assert rows == case["expected"], case["origin"]


def test_cells_are_sampled_at_their_centres():
    square = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    assert glyph_bitmap_rows((square,), 4, 3) == (0b000, 0b111, 0b111)


def test_empty_input_and_sizes_give_no_rows():
    assert glyph_bitmap_rows((), 18, 24) == ()
    assert glyph_bitmap_rows((((0.0, 0.0), (1.0, 1.0), (2.0, 0.0)),), 0, 24) == ()
    assert glyph_bitmap_rows((((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),), 18, 24) == ()

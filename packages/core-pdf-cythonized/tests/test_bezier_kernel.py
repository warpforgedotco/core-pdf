# SPDX-License-Identifier: AGPL-3.0-only


import json
import math
from pathlib import Path

import pytest

from core_pdf_cythonized import cubic_sample_times

GOLDEN = json.loads((Path(__file__).parent / "bezier_golden.json").read_text())


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 802


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_recorded_output(index):
    vector = GOLDEN[index]
    curve = tuple(tuple(point) for point in vector["curve"])
    assert cubic_sample_times(*curve) == tuple(vector["times"])


def test_sample_times_are_sorted_unique_and_bounded():
    for vector in GOLDEN:
        times = cubic_sample_times(*(tuple(p) for p in vector["curve"]))
        assert times == tuple(sorted(times))
        assert len(times) == len(set(times))
        assert times[-1] == 1.0
        assert all(0.0 < t <= 1.0 and math.isfinite(t) for t in times)


def test_core_reaches_the_sampler_through_the_type2_kernel():
    pytest.importorskip("core_pdf")
    from core_pdf.impl import fonts_font_program as font_program
    from core_pdf_cythonized import type2_glyph_geometry

    assert font_program.type2_glyph_geometry is type2_glyph_geometry

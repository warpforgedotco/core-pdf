# SPDX-License-Identifier: AGPL-3.0-only

"""The shading parameter: agreement with the Python it replaced.

shading_t_golden.pkl.gz holds 6,012 points against axial and radial
shadings, with the parameter core_pdf.impl.render.patterns.axial_shading_t
and radial_shading_t (both deleted) gave, or None: random coordinates and
points with signed zeros, tiny and huge values, infinities and NaNs among
them, degenerate axes, concentric and tangent circles, and the unit cases
those functions were tested with.
"""

import gzip
import math
import pickle
import struct
from pathlib import Path

from core_pdf_cythonized import shading_t

GOLDEN_PATH = Path(__file__).parent / "shading_t_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def exact(value: float | None) -> object:
    if value is None:
        return None
    return ("nan",) if math.isnan(value) else struct.pack("<d", value)


def test_golden_file_covers_the_cases_it_claims_to() -> None:
    assert len(GOLDEN) == 6012
    assert {case["kind"] for case in GOLDEN} == {2, 3}
    assert sum(case["expected"] is None for case in GOLDEN) == 2042


def test_every_point_is_the_pythons() -> None:
    for case in GOLDEN:
        x, y = case["point"]
        got = shading_t(case["kind"], case["coords"], x, y, 0.5)
        assert exact(got) == exact(case["expected"]), case

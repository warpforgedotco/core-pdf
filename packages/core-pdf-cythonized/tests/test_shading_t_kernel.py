# SPDX-License-Identifier: AGPL-3.0-only

"""The shading parameter: agreement with the Python it replaced, on this platform.

shading_t_golden.pkl.gz holds 6,012 points against axial and radial
shadings -- random coordinates and points with signed zeros, tiny and huge
values, infinities and NaNs among them, degenerate axes, concentric and
tangent circles, and the unit cases the deleted functions were tested
with -- and the parameter core_pdf.impl.render.patterns.axial_shading_t and
radial_shading_t gave on the machine that generated them.

The radial root is ``disc**0.5``, which CPython computes with the C
library's pow, and the kernel calls the same pow. pow is not correctly
rounded everywhere: macOS's differs from glibc's by an ulp on some inputs,
so a value recorded on one platform does not pin another. The deleted
functions are therefore kept here, verbatim, and the kernel is compared
with them on the platform the test runs on; the recorded values are
checked where the generating platform's pow agreed with this one's.
"""

import gzip
import math
import pickle
import struct
from pathlib import Path

from core_pdf_cythonized import shading_t

GOLDEN_PATH = Path(__file__).parent / "shading_t_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def axial_shading_t(coords, px, py):
    x0, y0, x1, y1 = coords[:4]
    dx = x1 - x0
    dy = y1 - y0
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return None
    return ((px - x0) * dx + (py - y0) * dy) / denom


def radial_shading_t(coords, px, py):
    x0, y0, r0, x1, y1, r1 = coords[:6]
    dx = x1 - x0
    dy = y1 - y0
    dr = r1 - r0
    qx = px - x0
    qy = py - y0
    a = dx * dx + dy * dy - dr * dr
    b = -2.0 * (qx * dx + qy * dy + r0 * dr)
    c = qx * qx + qy * qy - r0 * r0
    if abs(a) <= 1e-12:
        if abs(b) <= 1e-12:
            return None
        return -c / b
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    root = disc**0.5
    t0 = (-b - root) / (2.0 * a)
    t1 = (-b + root) / (2.0 * a)
    valid = [t for t in (t0, t1) if math.isfinite(t)]
    if not valid:
        return None
    in_range = [t for t in valid if 0.0 <= t <= 1.0]
    return max(in_range) if in_range else min(valid, key=lambda t: abs(t - 0.5))


def reference(case):
    x, y = case["point"]
    if case["kind"] == 2:
        return axial_shading_t(case["coords"], x, y)
    return radial_shading_t(case["coords"], x, y)


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
        assert exact(got) == exact(reference(case)), case


def test_the_recorded_values_are_the_reference_bar_pow() -> None:
    # Every recorded value is the reference's here, except where the
    # radial root came out of a pow that rounds differently on this
    # platform than on the one that recorded it -- an ulp at most.
    differing = [case for case in GOLDEN if exact(reference(case)) != exact(case["expected"])]
    assert len(differing) <= 3
    for case in differing:
        assert case["kind"] == 3
        got, recorded = reference(case), case["expected"]
        assert got is not None
        assert recorded is not None
        assert abs(got - recorded) <= 4 * math.ulp(recorded)

# SPDX-License-Identifier: AGPL-3.0-only


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
    differing = [case for case in GOLDEN if exact(reference(case)) != exact(case["expected"])]
    assert len(differing) <= 3
    for case in differing:
        assert case["kind"] == 3
        got, recorded = reference(case), case["expected"]
        assert got is not None
        assert recorded is not None
        assert abs(got - recorded) <= 4 * math.ulp(recorded)

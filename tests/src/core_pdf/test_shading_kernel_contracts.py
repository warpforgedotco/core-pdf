"""Axial and radial shadings painted by the kernels paint what paint_shading's loop did."""

import math
import random
from typing import Any

import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl.render import target as raster
from core_pdf.impl.render.patterns import axial_shading_t, radial_shading_t
from core_pdf_cythonized import shading_values


def loop_values(
    kind: int,
    coords: tuple[float, ...],
    box: tuple[int, int, int, int],
    allowed: numpy.ndarray,
    extend: tuple[bool, bool],
    domain: tuple[float, float],
    view: tuple[float, float, float],
) -> tuple[list[float], numpy.ndarray]:
    crop_x0, crop_y1, scale = view
    ix0, iy0, ix1, iy1 = box
    shading_t = axial_shading_t if kind == 2 else radial_shading_t
    painted = numpy.zeros((iy1 - iy0, ix1 - ix0), dtype=numpy.uint8)
    values: list[float] = []
    for py in range(iy0, iy1):
        page_y = crop_y1 - (py + 0.5) / scale
        for px in range(ix0, ix1):
            if not allowed[py - iy0, px - ix0]:
                continue
            unit_t = shading_t(coords, crop_x0 + (px + 0.5) / scale, page_y)
            if unit_t is None:
                continue
            if unit_t < 0.0:
                if not extend[0]:
                    continue
                unit_t = 0.0
            elif unit_t > 1.0:
                if not extend[1]:
                    continue
                unit_t = 1.0
            values.append(domain[0] + unit_t * (domain[1] - domain[0]))
            painted[py - iy0, px - ix0] = 1
    return values, painted


def same(left: list[float], right: list[float]) -> bool:
    return len(left) == len(right) and all(
        (math.isnan(a) and math.isnan(b)) or (a == b and math.copysign(1, a) == math.copysign(1, b))
        for a, b in zip(left, right, strict=True)
    )


@pytest.mark.parametrize("seed", range(40))
def test_values_are_the_loops(seed: int) -> None:
    rng = random.Random(seed)
    kind = rng.choice([2, 3])
    pick = [0.0, 1.0, 5.0, -3.0, 10.0, 1e-7, 50.5, 0.25]

    def number() -> float:
        return rng.choice(pick) if rng.random() < 0.4 else rng.uniform(-60.0, 60.0)

    coords = tuple(number() for _ in range(4 if kind == 2 else 6))
    if kind == 3 and rng.random() < 0.3:
        coords = (coords[0], coords[1], abs(coords[2]), coords[0], coords[1], abs(coords[5]))
    box = (rng.randint(0, 5), rng.randint(0, 5), rng.randint(6, 30), rng.randint(6, 30))
    allowed = numpy.random.default_rng(seed).random((box[3] - box[1], box[2] - box[0])) < 0.8
    allowed = allowed.astype(numpy.uint8)
    extend = (rng.random() < 0.5, rng.random() < 0.5)
    domain = (rng.choice([0.0, -1.0, 0.5]), rng.choice([1.0, 2.0, -0.0]))
    view = (rng.uniform(-10.0, 10.0), rng.uniform(20.0, 60.0), rng.choice([1.0, 0.75, 2.0]))
    expected, expected_painted = loop_values(kind, coords, box, allowed, extend, domain, view)
    values, painted = shading_values(
        kind,
        coords,
        view[0],
        view[1],
        view[2],
        box[0],
        box[1],
        box[2],
        box[3],
        allowed,
        extend[0],
        extend[1],
        domain[0],
        domain[1] - domain[0],
        0.5,
    )
    numpy.testing.assert_array_equal(painted, expected_painted)
    assert same(values[painted.view(numpy.bool_)].tolist(), expected)


def one_page_pdf(shading: bytes, content: bytes) -> bytes:
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 120 90] "
        b"/Resources << /Shading << /S 5 0 R >> /ExtGState << /G 6 0 R >> >> /Contents 4 0 R >>",
        4: b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        5: shading,
        6: b"<< /Type /ExtGState /ca 0.6 /BM /Normal >>",
    }
    data = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for number, body in objects.items():
        offsets[number] = len(data)
        data += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 7\n0000000000 65535 f \n"
    data += b"".join(b"%010d 00000 n \n" % offsets[number] for number in range(1, 7))
    data += b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref
    return bytes(data)


FUNCTION = b"<< /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0.5 1] /N 1 >>"
SHADINGS = [
    pytest.param(
        b"<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [10 10 100 70] "
        b"/Function " + FUNCTION + b" >>",
        id="axial",
    ),
    pytest.param(
        b"<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [10 10 100 70] "
        b"/Extend [true true] /Function " + FUNCTION + b" >>",
        id="axial-extended",
    ),
    pytest.param(
        b"<< /ShadingType 3 /ColorSpace /DeviceRGB /Coords [60 45 5 60 45 40] "
        b"/Extend [false true] /Function " + FUNCTION + b" >>",
        id="radial",
    ),
    pytest.param(
        b"<< /ShadingType 3 /ColorSpace /DeviceRGB /Coords [30 30 30 80 60 10] "
        b"/Extend [true false] /Function " + FUNCTION + b" >>",
        id="radial-cone",
    ),
]
CONTENTS = [
    pytest.param(b"/S sh", id="page"),
    pytest.param(b"q 20 15 70 50 re W n /S sh Q", id="clipped"),
    pytest.param(b"q /G gs 20 15 70 50 re W n /S sh Q", id="translucent"),
    pytest.param(b"q 1 0 0 1 0 0 cm 5 5 m 110 20 l 60 85 l h W n /G gs /S sh Q", id="path-clip"),
    pytest.param(b"q /G gs /S sh Q 0 0 1 rg 30 30 20 20 re f", id="then-fill"),
]


def rendered(data: bytes) -> bytes:
    with PdfDocument(data) as document:
        return document.pages[0].render().rasterize(scale=1.5).array().tobytes()


@pytest.mark.parametrize("shading", SHADINGS)
@pytest.mark.parametrize("content", CONTENTS)
def test_kernel_paint_is_the_loops(
    shading: bytes, content: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = one_page_pdf(shading, content)
    taken: list[bool] = []
    original = raster.RasterTarget.paint_shading_compiled

    def recorded(self: raster.RasterTarget, *args: Any) -> bool:
        result = original(self, *args)
        taken.append(result)
        return result

    monkeypatch.setattr(raster.RasterTarget, "paint_shading_compiled", recorded)
    compiled = rendered(data)
    assert taken
    assert all(taken)
    monkeypatch.setattr(raster.RasterTarget, "paint_shading_compiled", lambda *_: False)
    assert rendered(data) == compiled

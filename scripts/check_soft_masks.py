# SPDX-License-Identifier: AGPL-3.0-only

"""Check Luminosity soft masks against the specification's reference, byte for byte.

core_pdf.impl.spec.s_11_transparency.soft_masks holds the reference: a colour's luminosity
in its blending space (ISO 32000-2 11.5.3) and the mask value the transfer function gives it
(11.6.5.1). The renderer derives a Luminosity mask with the compiled luminosity_channel,
from a raster that holds the mask group composited over its BC colour in the space's form
(core_pdf.impl.render.raster_spaces): sRGB, a gray level in all three components, or a CMYK
group's two passes, the complements of C, M and Y and the complement of K.

This feeds luminosity_channel every gray level, every RGB colour whose components are edge
levels, every CMYK colour of edge levels, and random colours, in each space's form, and
compares each byte with the reference's luminosity of the colour the bytes stand for,
rounded to a byte, and the occurring bytes it reports with those it returned. It then
renders small pages whose content is masked by a Luminosity mask in each blending space --
an opaque fill inside the group's bounding box, with and without a BC colour and a transfer
function -- and checks the mask core builds inside the fill, inside the bounding box and
outside it against luminosity_mask_value of the fill's and the BC colour, through the mask's
own transfer function. It also checks that the kernel refuses what it cannot take.

uv run --all-packages python scripts/check_soft_masks.py [random colours per space]
"""

from __future__ import annotations

import itertools
import sys
from typing import Any

import numpy

from core_pdf import PdfDocument
from core_pdf.impl.capture.records import CapturedSoftMask
from core_pdf.impl.render import raster_dispatch
from core_pdf.impl.render.model import SoftMaskPlane
from core_pdf.impl.render.raster_spaces import LUMINOSITY_SPACES
from core_pdf.impl.render.target import RasterTarget
from core_pdf.impl.spec.s_11_transparency.blending_space import BlendingSpace
from core_pdf.impl.spec.s_11_transparency.soft_masks import (
    device_luminosity,
    luminosity_mask_color,
    luminosity_mask_value,
)
from core_pdf_cythonized import luminosity_channel

EDGE_LEVELS = (0, 1, 2, 63, 64, 127, 128, 191, 192, 253, 254, 255)
SPACES: tuple[BlendingSpace, ...] = ("DeviceGray", "DeviceRGB", "DeviceCMYK")
COMPONENTS = {"DeviceGray": 1, "DeviceRGB": 3, "DeviceCMYK": 4}
OPERATORS = {"DeviceGray": b"g", "DeviceRGB": b"rg", "DeviceCMYK": b"k"}


def reference_byte(space: BlendingSpace, levels: tuple[int, ...]) -> int:
    """The reference luminosity of the colour whose components are levels / 255, as a byte."""
    value = device_luminosity(space, tuple(level / 255.0 for level in levels))
    return max(0, min(255, round(value * 255.0)))


def colors(space: BlendingSpace, count: int, seed: int) -> list[tuple[int, ...]]:
    n = COMPONENTS[space]
    edge = list(itertools.product(EDGE_LEVELS, repeat=n))
    if space == "DeviceGray":
        edge = [(level,) for level in range(256)]
    random = numpy.random.default_rng(seed)
    drawn = [tuple(int(v) for v in row) for row in random.integers(0, 256, (count, n))]
    return edge + drawn


def raster_of(space: BlendingSpace, batch: list[tuple[int, ...]]) -> list[numpy.ndarray]:
    """The colours as one row of each pass the renderer paints a group of the space in."""
    levels = numpy.asarray(batch, dtype=numpy.uint8)
    pixels = numpy.zeros((1, len(batch), 4), dtype=numpy.uint8)
    pixels[..., 3] = 255
    if space == "DeviceGray":
        pixels[0, :, :3] = levels[:, :1]
        return [pixels]
    if space == "DeviceRGB":
        pixels[0, :, :3] = levels
        return [pixels]
    black = pixels.copy()
    pixels[0, :, :3] = 255 - levels[:, :3]
    black[0, :, :3] = 255 - levels[:, 3:4]
    return [pixels, black]


def check_kernel(count: int, failures: list[str]) -> int:
    checked = 0
    for index, space in enumerate(SPACES):
        batch = colors(space, count, 20261114 + index)
        passes = raster_of(space, batch)
        got, present = luminosity_channel(
            passes[0], passes[1] if len(passes) > 1 else None, LUMINOSITY_SPACES[space], True
        )
        want = [reference_byte(space, color) for color in batch]
        for color, value, expected in zip(batch, got[0].tolist(), want, strict=True):
            if value != expected:
                failures.append(f"luminosity_channel {space} {color}: {value} != {expected}")
        assert present is not None
        if set(numpy.flatnonzero(present).tolist()) != set(want):
            failures.append(f"luminosity_channel {space}: wrong occurring bytes")
        unflagged, absent = luminosity_channel(
            passes[0], passes[1] if len(passes) > 1 else None, LUMINOSITY_SPACES[space], False
        )
        if absent is not None or not numpy.array_equal(unflagged, got):
            failures.append(f"luminosity_channel {space}: differs without presence")
        checked += len(batch)
    return checked


def check_refusals(failures: list[str]) -> None:
    pixels = numpy.zeros((2, 2, 4), dtype=numpy.uint8)
    for args in (
        (numpy.zeros((2, 2, 3), dtype=numpy.uint8), None, 0, False),
        (pixels, None, 3, False),
        (pixels, None, -1, False),
        (pixels, None, 2, False),
        (pixels, numpy.zeros((2, 3, 4), dtype=numpy.uint8), 2, False),
    ):
        try:
            luminosity_channel(*args)
        except ValueError:
            continue
        failures.append(f"luminosity_channel accepted {args[2]} with shapes {args[0].shape}")


def check_references(failures: list[str]) -> None:
    # C = (1 - ag) C0 + ag Cg: the backdrop where the group has no alpha, the group's colour
    # where it is opaque.
    for backdrop, color in (((0.2,), (0.8,)), ((0.1, 0.5, 1.0), (1.0, 0.0, 0.25))):
        if luminosity_mask_color(backdrop, color, 0.0) != backdrop:
            failures.append("luminosity_mask_color: alpha 0 is not the backdrop")
        if luminosity_mask_color(backdrop, color, 1.0) != color:
            failures.append("luminosity_mask_color: alpha 1 is not the group's colour")
    if device_luminosity("DeviceCMYK", (1.0, 1.0, 1.0, 1.0)) != 0.0:
        failures.append("device_luminosity: CMYK is not clamped at black")
    if luminosity_mask_value("DeviceGray", (0.25,), lambda x: (1.0 - x,)) != 0.75:
        failures.append("luminosity_mask_value: transfer function not applied")


def page(space: BlendingSpace, fill: tuple[int, ...], backdrop: bytes, transfer: bytes) -> bytes:
    """A 40 by 40 page filled through a Luminosity mask whose group, of the space, has a
    bounding box from 10 to 30 and an opaque fill of the colour from 15 to 25."""
    operands = b" ".join(b"%.10f" % (level / 255.0) for level in fill)
    group = (
        b"<< /Type /XObject /Subtype /Form /BBox [10 10 30 30] "
        b"/Group << /S /Transparency /CS /" + space.encode() + b" >> /Length %d >>\nstream\n"
    )
    body = operands + b" " + OPERATORS[space] + b" 15 15 10 10 re f"
    smask = b"/S /Luminosity /G 5 0 R " + backdrop + b" " + transfer
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 40 40] /Contents 4 0 R "
        b"/Resources << /ExtGState << /M << /SMask << " + smask + b" >> >> >> >> >>",
        b"<< /Length 30 >>\nstream\n/M gs 1 0 0 rg 0 0 40 40 re f\nendstream",
        group % len(body) + body + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def mask_planes(data: bytes) -> list[tuple[CapturedSoftMask, SoftMaskPlane | None]]:
    seen: list[tuple[CapturedSoftMask, SoftMaskPlane | None]] = []
    original = raster_dispatch.resolve_soft_mask

    def record(target: RasterTarget, mask: CapturedSoftMask) -> SoftMaskPlane | None:
        plane = original(target, mask)
        seen.append((mask, plane))
        return plane

    raster_dispatch.resolve_soft_mask = record  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
    try:
        with PdfDocument(data) as document:
            document.pages[0].render().rasterize(scale=1.0)
    finally:
        raster_dispatch.resolve_soft_mask = original
    return seen


def plane_value(plane: SoftMaskPlane, row: int, column: int) -> tuple[int, Any]:
    byte = int(plane.alpha[row, column])
    return byte, None if plane.table is None else float(plane.table[byte])


TRANSFERS = {
    b"": None,
    b"/TR /Identity": None,
    b"/TR << /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >>": "invert",
    b"/TR << /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [1] /N 2.2 >>": "gamma",
}
FILLS: dict[BlendingSpace, tuple[tuple[int, ...], ...]] = {
    "DeviceGray": ((0,), (77,), (255,)),
    "DeviceRGB": ((255, 0, 0), (12, 200, 99), (255, 255, 255)),
    "DeviceCMYK": ((0, 0, 0, 0), (51, 102, 153, 25), (255, 255, 255, 255)),
}
BACKDROPS: dict[BlendingSpace, tuple[tuple[int, ...] | None, ...]] = {
    "DeviceGray": (None, (255,), (128,)),
    "DeviceRGB": (None, (255, 255, 255), (30, 60, 90)),
    "DeviceCMYK": (None, (0, 0, 0, 0), (10, 20, 30, 40)),
}
INITIAL: dict[BlendingSpace, tuple[int, ...]] = {
    "DeviceGray": (0,),
    "DeviceRGB": (0, 0, 0),
    "DeviceCMYK": (0, 0, 0, 255),
}
# Pixels (row, column) of the 40 by 40 raster: inside the fill, inside the bounding box
# only, and outside it.
SAMPLES = {"fill": (20, 20), "bbox": (27, 12), "outside": (2, 2)}


def check_pages(failures: list[str]) -> int:
    checked = 0
    for space in SPACES:
        for fill, backdrop, (transfer, _) in itertools.product(
            FILLS[space], BACKDROPS[space], TRANSFERS.items()
        ):
            entry = (
                b""
                if backdrop is None
                else b"/BC [%s]" % b" ".join(b"%.10f" % (level / 255.0) for level in backdrop)
            )
            planes = mask_planes(page(space, fill, entry, transfer))
            label = f"{space} fill {fill} BC {backdrop} {transfer.decode() or 'no TR'}"
            if len(planes) != 1 or planes[0][1] is None:
                failures.append(f"{label}: no mask plane")
                continue
            mask, plane = planes[0]
            assert plane is not None
            base = INITIAL[space] if backdrop is None else backdrop
            for name, (row, column) in SAMPLES.items():
                levels = fill if name == "fill" else base
                byte, value = plane_value(plane, row, column)
                want = reference_byte(space, levels)
                if byte != want:
                    failures.append(f"{label} {name}: luminosity {byte} != {want}")
                if mask.transfer is None:
                    if value is not None:
                        failures.append(f"{label} {name}: identity transfer has a table")
                    continue
                expected = numpy.float32(
                    luminosity_mask_value("DeviceGray", (want / 255.0,), mask.transfer)
                )
                if value != float(expected):
                    failures.append(f"{label} {name}: mask value {value} != {expected}")
                checked += 1
            checked += 1
    return checked


def main(argv: list[str]) -> int:
    count = int(argv[0]) if argv else 20000
    failures: list[str] = []
    check_references(failures)
    check_refusals(failures)
    in_kernel = check_kernel(count, failures)
    on_pages = check_pages(failures)
    print(
        f"{in_kernel} colours through luminosity_channel in {len(SPACES)} blending spaces, "
        f"{on_pages} mask samples on rendered pages: {len(failures)} failures"
    )
    for failure in failures[:20]:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

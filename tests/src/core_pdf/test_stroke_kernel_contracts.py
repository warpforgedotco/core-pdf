"""Strokes painted by stroke_polylines paint what stroke_path's primitives did.

stroke_path hands a deferred, undashed path under normal blending, with no
group planes and no clip or a rectangular one, to the kernel. The reference
is the same call with the kernel refused, so the Python loop paints it
primitive by primitive. Random paths cover every branch the kernel mirrors:
axis-aligned and degenerate segments, lines above and below 64 pixels,
miter and round joins, butt, round and square caps, closed subpaths with
and without a repeated closing point, coincident two-point subpaths,
translucent and transparent colours over empty, opaque and mixed backdrops,
clips, scales, paint windows, and coordinates that make floor or ceil raise.
"""

import math
import random
from typing import Any

import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl.capture.records import CapturedPath
from core_pdf.impl.render import target as raster
from core_pdf.impl.render.target import RasterTarget


def make_target(
    width: int, height: int, backdrop: str, scale: float, rng: random.Random
) -> RasterTarget:
    if backdrop == "empty":
        pixels = bytearray(width * height * 4)
    elif backdrop == "opaque":
        pixels = bytearray(bytes([30, 90, 200, 255]) * (width * height))
    else:
        pixels = bytearray(rng.randrange(256) for _ in range(width * height * 4))
    view = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(height, width, 4)
    clip = raster.ClipState(
        crop_x0=0, crop_y1=height / scale, scale=scale, width=width, height=height
    )
    return RasterTarget(
        pixels,
        None,
        clip=clip,
        width=width,
        height=height,
        scale=scale,
        crop_x0=0,
        crop_y0=0,
        crop_y1=height / scale,
        page_view=view,
    )


def random_path(rng: random.Random, size: float) -> CapturedPath:
    xs: list[float] = []
    ys: list[float] = []
    spans: list[tuple[int, int, bool]] = []
    for _ in range(rng.randint(1, 4)):
        start = len(xs)
        count = rng.choice([1, 2, 2, 3, 4, 6])
        x, y = rng.uniform(-2, size + 2), rng.uniform(-2, size + 2)
        for index in range(count):
            if index:
                step = rng.choice(["h", "v", "d", "same", "tiny"])
                if step == "h":
                    x += rng.uniform(-size / 2, size / 2)
                elif step == "v":
                    y += rng.uniform(-size / 2, size / 2)
                elif step == "d":
                    x += rng.uniform(-size / 2, size / 2)
                    y += rng.uniform(-size / 2, size / 2)
                elif step == "tiny":
                    x += 1e-7
                    y += 1e-7
            xs.append(x)
            ys.append(y)
        closed = rng.random() < 0.4
        if closed and count > 2 and rng.random() < 0.5:
            xs.append(xs[start])
            ys.append(ys[start])
        spans.append((start, len(xs), closed))
    columns = numpy.asarray(xs, dtype=numpy.float64), numpy.asarray(ys, dtype=numpy.float64)
    return CapturedPath.deferred_flattened(*columns, spans, None, True)


def outcome(target: RasterTarget, stroke: Any) -> tuple[Any, ...]:
    try:
        stroke()
        raised = None
    except Exception as error:  # noqa: BLE001 -- the comparison covers failures too
        raised = (type(error), str(error))
    window = None if target.paint_window is None else list(target.paint_window)
    return raised, bytes(target.pixels), window


@pytest.mark.parametrize("seed", range(300))
def test_the_kernel_strokes_as_the_primitives_did(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = random.Random(seed)
    width, height = rng.choice([(24, 20), (40, 36)])
    scale = rng.choice([1.0, 1.5, 0.75])
    size = min(width, height) / scale
    path = random_path(rng, size)
    line_width = rng.choice([0.0, 0.3, 1.0, 2.5, 7.0])
    rgba = rng.choice(
        [(0, 0, 0, 255), (200, 40, 90, 255), (10, 200, 60, 128), (5, 6, 7, 0), (250, 250, 5, 17)]
    )
    cap = rng.choice([0, 1, 2])
    join = rng.choice([0, 1, 2])
    backdrop = rng.choice(["empty", "opaque", "mixed"])
    clip = rng.random() < 0.4
    window = rng.random() < 0.5
    pixels_seed = rng.random()
    outcomes = []
    for native in (True, False):
        target = make_target(width, height, backdrop, scale, random.Random(pixels_seed))
        if clip:
            target.clip.push(
                CapturedPath.deferred_flattened(
                    numpy.array([2.3, size - 3.1, size - 3.1, 2.3]),
                    numpy.array([1.7, 1.7, size - 2.2, size - 2.2]),
                    [(0, 4, True)],
                    None,
                    True,
                ),
                "nonzero",
            )
        if window:
            target.paint_window = []
        if not native:
            monkeypatch.setattr(RasterTarget, "stroke_natively", lambda *_: False)
        taken: list[bool] = []
        original = RasterTarget.stroke_natively

        def watched(
            self: RasterTarget, *args: Any, _original: Any = original, _taken: list[bool] = taken
        ) -> bool:
            result = _original(self, *args)
            _taken.append(result)
            return result

        if native:
            monkeypatch.setattr(RasterTarget, "stroke_natively", watched)
        outcomes.append(
            outcome(
                target,
                lambda target=target: target.stroke_path(
                    path, line_width, rgba, None, None, cap, join
                ),
            )
        )
        if native:
            assert taken == [True]
        monkeypatch.undo()
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan, 1e300])
def test_a_box_floor_refuses_raises_as_the_primitives_did(
    value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = []
    for native in (True, False):
        path = CapturedPath.deferred_flattened(
            numpy.array([2.0, 9.0, 9.0, value, 4.0]),
            numpy.array([2.0, 3.0, 9.0, 5.0, 12.0]),
            [(0, 3, False), (3, 5, False)],
            None,
            True,
        )
        target = make_target(20, 20, "mixed", 1.0, random.Random(3))
        target.paint_window = []
        if not native:
            monkeypatch.setattr(RasterTarget, "stroke_natively", lambda *_: False)
        outcomes.append(
            outcome(
                target,
                lambda target=target, path=path: target.stroke_path(
                    path, 1.5, (9, 9, 9, 200), None, None, 1, 1
                ),
            )
        )
        monkeypatch.undo()
    assert outcomes[0] == outcomes[1]


CONTENT = (
    b"0.9 0.9 0.2 rg 0 0 200 150 re f "
    b"1 J 1 j 2 w 0 0 1 RG 10 10 m 190 140 l 30 130 l S 10 10 m 10 10 l S "
    b"0 J 0 j 0.5 w 1 0 0 RG 20 20 m 180 20 l 180 120 l 20 120 l h S "
    b"2 J 4 w 0 0.5 0 RG 50 60 m 150 90 l S 50 70 m 150 100 l S 50 80 m 150 110 l S "
    b"q 30 30 120 80 re W n 3 w 0.2 0.2 0.2 RG 0 75 m 200 75 l 100 0 l 100 150 l S Q "
    b"/G gs 6 w 1 0 1 RG 10 140 m 190 10 l S "
    b"q 40 40 m 160 50 l 100 130 l h W n 2 w 0 0 0 RG 0 0 m 200 150 l S Q"
)


def one_page_pdf(content: bytes) -> bytes:
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 150] "
        b"/Resources << /ExtGState << /G 5 0 R >> >> /Contents 4 0 R >>",
        4: b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        5: b"<< /Type /ExtGState /CA 0.5 >>",
    }
    data = bytearray(b"%PDF-1.7\n")
    offsets = {}
    for number, body in objects.items():
        offsets[number] = len(data)
        data += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 6\n0000000000 65535 f \n"
    data += b"".join(b"%010d 00000 n \n" % offsets[number] for number in range(1, 6))
    data += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref
    return bytes(data)


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_a_page_strokes_as_the_primitives_did(
    scale: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = one_page_pdf(CONTENT)

    def rendered() -> bytes:
        with PdfDocument(data) as document:
            return document.pages[0].render().rasterize(scale=scale).array().tobytes()

    taken: list[bool] = []
    original = RasterTarget.stroke_natively

    def watched(self: RasterTarget, *args: Any) -> bool:
        result = original(self, *args)
        taken.append(result)
        return result

    monkeypatch.setattr(RasterTarget, "stroke_natively", watched)
    native = rendered()
    assert any(taken)
    assert not all(taken)
    monkeypatch.setattr(RasterTarget, "stroke_natively", lambda *_: False)
    assert rendered() == native


@pytest.mark.parametrize("seed", range(60))
def test_pixel_aligned_strokes_blend_as_the_primitives_did(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Axis-aligned segments on half-pixel centres at width 1, or on whole
    # pixels at width 2, give boxes whose edges fall on pixels: the solid
    # fills, opaque and in each of the translucent regimes.
    rng = random.Random(seed)
    half_width = rng.choice([0.5, 1.0])
    offset = 0.5 if half_width == 0.5 else 0.0
    xs: list[float] = []
    ys: list[float] = []
    spans: list[tuple[int, int, bool]] = []
    for _ in range(rng.randint(1, 4)):
        start = len(xs)
        x, y = rng.randint(2, 16) + offset, rng.randint(2, 16) + offset
        xs.append(x)
        ys.append(y)
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.5:
                x = rng.randint(2, 16) + offset
            else:
                y = rng.randint(2, 16) + offset
            xs.append(x)
            ys.append(y)
        spans.append((start, len(xs), rng.random() < 0.3))
    path = CapturedPath.deferred_flattened(
        numpy.asarray(xs, dtype=numpy.float64),
        numpy.asarray(ys, dtype=numpy.float64),
        spans,
        None,
        True,
    )
    rgba = rng.choice([(200, 40, 90, 255), (10, 200, 60, 128), (250, 250, 5, 17), (5, 6, 7, 0)])
    backdrop = rng.choice(["empty", "opaque", "mixed"])
    cap = rng.choice([0, 2])
    outcomes = []
    for native in (True, False):
        target = make_target(20, 20, backdrop, 1.0, random.Random(seed))
        target.paint_window = []
        if not native:
            monkeypatch.setattr(RasterTarget, "stroke_natively", lambda *_: False)
        outcomes.append(
            outcome(
                target,
                lambda target=target: target.stroke_path(
                    path, half_width * 2, rgba, None, None, cap, 0
                ),
            )
        )
        monkeypatch.undo()
    assert outcomes[0] == outcomes[1]

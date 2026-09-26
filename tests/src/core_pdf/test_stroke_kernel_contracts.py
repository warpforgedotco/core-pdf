"""Strokes walked by stroke_polylines paint what stroke_path's Python walk did.

The reference below is that walk as it was: each subpath as fill_line,
fill_join and fill_cap calls, a coincident two-point subpath under a round
cap as a circle path, a dashed path as its dash pieces. The kernel now walks
every stroke, painting the primitives itself under normal blending with no
group planes and calling the Python ones back otherwise. Random paths cover
every branch it mirrors: axis-aligned and degenerate segments, lines above
and below 64 pixels, miter and round joins, butt, round and square caps,
closed subpaths with and without a repeated closing point, coincident
two-point subpaths, built paths with shared and NaN coordinates, dashes,
translucent and transparent colours over empty, opaque and mixed backdrops,
no clip, rectangular and path clips, blend modes, group planes, scales,
paint windows, and coordinates that make floor or ceil raise.
"""

import math
import random
from typing import Any

import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl.capture.records import CapturedPath, CapturedSubpath
from core_pdf.impl.render import target as raster
from core_pdf.impl.render.model import LineCap
from core_pdf.impl.render.paths import circle_path, dash_subpath, intersect_box
from core_pdf.impl.render.target import RasterTarget
from core_pdf_cythonized import path_bounds


def deferred(xs: Any, ys: Any, spans: list[tuple[int, int, bool]]) -> CapturedPath:
    """A flattened path as capture makes one, its box and segment flag with it."""
    columns = numpy.asarray(xs, dtype=numpy.float64), numpy.asarray(ys, dtype=numpy.float64)
    return CapturedPath.deferred_flattened(*columns, spans, *path_bounds(*columns, spans))


def reference_stroke_path(
    self: RasterTarget,
    path: CapturedPath,
    line_width: float,
    rgba: tuple[int, int, int, int],
    dash_pattern: tuple[list[float], float] | None = None,
    blend_mode: str | None = None,
    line_cap: int = 0,
    line_join: int = 0,
) -> None:
    scale = self.scale
    if self.clip.regions:
        clip_box = self.clip.current_clip()
        path_box = self.clip.path_bbox(path)
        if clip_box is not None and path_box is not None:
            stroke_pad = max(0.5 / scale, float(line_width) * 0.5)
            stroke_box = (
                path_box[0] - stroke_pad,
                path_box[1] - stroke_pad,
                path_box[2] + stroke_pad,
                path_box[3] + stroke_pad,
            )
            if intersect_box(stroke_box, clip_box) is None:
                return
    for subpath in path.subpaths:
        if dash_pattern and dash_pattern[0]:
            reference_stroke_path(
                self,
                CapturedPath(dash_subpath(subpath, dash_pattern)),
                line_width,
                rgba,
                None,
                blend_mode,
                line_cap,
                line_join,
            )
            continue
        points = subpath.points
        if len(points) < 2:
            continue
        if len(points) == 2 and not subpath.closed:
            (x0, y0), (x1, y1) = points
            if (x0, y0) == (x1, y1):
                if line_cap == LineCap.ROUND:
                    radius = line_width * 0.5 if line_width > 0.0 else 0.5 / scale
                    self.fill_path(circle_path(x0, y0, radius), rgba, blend_mode)
                continue
            self.fill_line(x0, y0, x1, y1, line_width, rgba, blend_mode)
            if line_cap != 0:
                self.fill_cap(x0, y0, line_width, rgba, line_cap, blend_mode)
                self.fill_cap(x1, y1, line_width, rgba, line_cap, blend_mode)
            continue
        for index in range(len(points) - 1):
            x0, y0 = points[index]
            x1, y1 = points[index + 1]
            self.fill_line(x0, y0, x1, y1, line_width, rgba, blend_mode)
        if subpath.closed and points[0] != points[-1]:
            x0, y0 = points[-1]
            x1, y1 = points[0]
            self.fill_line(x0, y0, x1, y1, line_width, rgba, blend_mode)
        for x, y in points[1:-1]:
            self.fill_join(x, y, line_width, rgba, line_join, blend_mode)
        if subpath.closed:
            x, y = points[0]
            self.fill_join(x, y, line_width, rgba, line_join, blend_mode)
        elif line_cap != 0:
            self.fill_cap(points[0][0], points[0][1], line_width, rgba, line_cap, blend_mode)
            self.fill_cap(points[-1][0], points[-1][1], line_width, rgba, line_cap, blend_mode)


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


def random_columns(
    rng: random.Random, size: float
) -> tuple[list[float], list[float], list[tuple[int, int, bool]]]:
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
    return xs, ys, spans


def random_path(rng: random.Random, size: float, *, built: bool = False) -> CapturedPath:
    xs, ys, spans = random_columns(rng, size)
    if built:
        # Built subpaths share their tuples, as dash pieces and repeated
        # closing points do, so tuple comparison meets identity.
        points = list(zip(xs, ys, strict=True))
        return CapturedPath(
            [
                CapturedSubpath(
                    [points[start], *points[start + 1 : end - 1], points[start]]
                    if closed and end - start > 2 and rng.random() < 0.3
                    else points[start:end],
                    closed=closed,
                )
                for start, end, closed in spans
            ]
        )
    return deferred(xs, ys, spans)


def push_clip(target: RasterTarget, size: float, shape: str) -> None:
    if shape == "rect":
        xs = [2.3, size - 3.1, size - 3.1, 2.3]
        ys = [1.7, 1.7, size - 2.2, size - 2.2]
    elif shape == "wide":
        # Not a rectangle, but rows of nearly the full width: long spans.
        xs = [0.5, size - 0.5, size - 0.5, size / 2, 0.5]
        ys = [0.5, 0.5, size - 0.5, size - 6.0, size - 0.5]
    else:
        xs = [1.0, size - 1.5, size / 2, 3.5]
        ys = [2.0, 4.0, size - 1.0, size / 2]
    target.clip.push(deferred(xs, ys, [(0, len(xs), True)]), "nonzero")
    region = target.clip.current_region()
    assert region is not None
    assert not region.empty
    assert region.rectangular == (shape == "rect")


def outcome(target: RasterTarget, stroke: Any) -> tuple[Any, ...]:
    try:
        stroke()
        raised = None
    except Exception as error:  # noqa: BLE001 -- the comparison covers failures too
        raised = (type(error), str(error))
    window = None if target.paint_window is None else list(target.paint_window)
    planes = tuple(
        None if plane is None else plane.tobytes()
        for plane in (target.group_source_alpha, target.group_source_shape)
    )
    return raised, bytes(target.pixels), window, planes


def compare(build: Any, stroke: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = []
    for reference in (False, True):
        target = build()
        if reference:
            monkeypatch.setattr(RasterTarget, "stroke_path", reference_stroke_path)
        outcomes.append(outcome(target, lambda target=target: stroke(target)))
        monkeypatch.undo()
    assert outcomes[0] == outcomes[1]


COLOURS = [
    (0, 0, 0, 255),
    (200, 40, 90, 255),
    (10, 200, 60, 128),
    (5, 6, 7, 0),
    (250, 250, 5, 17),
]


@pytest.mark.parametrize("seed", range(400))
def test_the_kernel_strokes_as_the_python_walk_did(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = random.Random(seed)
    width, height = rng.choice([(24, 20), (40, 36)])
    scale = rng.choice([1.0, 1.5, 0.75])
    size = min(width, height) / scale
    path = random_path(rng, size, built=rng.random() < 0.25)
    line_width = rng.choice([0.0, 0.3, 1.0, 2.5, 7.0])
    rgba = rng.choice(COLOURS)
    cap = rng.choice([0, 1, 2])
    join = rng.choice([0, 1, 2])
    backdrop = rng.choice(["empty", "opaque", "mixed"])
    clip = rng.choice([None, None, "rect", "path", "wide"])
    window = rng.random() < 0.5
    dash = (
        ([rng.uniform(0.5, 4.0), rng.uniform(0.5, 3.0)], rng.uniform(0, 2))
        if rng.random() < 0.15
        else None
    )
    blend = rng.choice([None] * 8 + ["Multiply", "Screen", "Normal"])
    planes = rng.random() < 0.1
    pixels_seed = rng.random()

    def build() -> RasterTarget:
        target = make_target(width, height, backdrop, scale, random.Random(pixels_seed))
        if clip is not None:
            push_clip(target, size, clip)
        if window:
            target.paint_window = []
        if planes:
            target.group_source_alpha = numpy.full((height, width), 0.25, dtype=numpy.float32)
            target.paint_window = []
        return target

    compare(
        build,
        lambda target: target.stroke_path(path, line_width, rgba, dash, blend, cap, join),
        monkeypatch,
    )


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan, 1e300])
@pytest.mark.parametrize("clip", [None, "rect", "path"])
@pytest.mark.parametrize("built", [False, True])
def test_a_box_floor_refuses_raises_as_the_python_walk_did(
    value: float, clip: str | None, built: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    xs = [2.0, 9.0, 9.0, value, 4.0]
    ys = [2.0, 3.0, 9.0, 5.0, 12.0]
    spans = [(0, 3, False), (3, 5, False)]
    if built:
        path = CapturedPath(
            [
                CapturedSubpath(list(zip(xs[s:e], ys[s:e], strict=True)), closed=c)
                for s, e, c in spans
            ]
        )
    else:
        path = deferred(xs, ys, spans)

    def build() -> RasterTarget:
        target = make_target(20, 20, "mixed", 1.0, random.Random(3))
        if clip is not None:
            push_clip(target, 20.0, clip)
        target.paint_window = []
        return target

    compare(
        build,
        lambda target: target.stroke_path(path, 1.5, (9, 9, 9, 200), None, None, 1, 1),
        monkeypatch,
    )


def test_a_nan_shared_between_points_compares_as_the_same_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nan = math.nan
    point = (nan, 3.0)
    paths = [
        CapturedPath([CapturedSubpath([point, point])]),
        CapturedPath([CapturedSubpath([point, (4.0, 5.0), point], closed=True)]),
        CapturedPath([CapturedSubpath([(nan, 3.0), (nan, 3.0)])]),
    ]
    for path in paths:
        compare(
            lambda: make_target(10, 10, "empty", 1.0, random.Random(0)),
            lambda target, path=path: target.stroke_path(
                path, 2.0, (1, 2, 3, 255), None, None, 1, 0
            ),
            monkeypatch,
        )


@pytest.mark.parametrize("seed", range(60))
def test_pixel_aligned_strokes_blend_as_the_python_walk_did(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Axis-aligned segments on half-pixel centres at width 1, or on whole
    # pixels at width 2, give boxes whose edges fall on pixels: the solid
    # fills, opaque and in each of the translucent regimes, and under a clip
    # path the solid spans of 32 pixels or more.
    rng = random.Random(seed)
    half_width = rng.choice([0.5, 1.0])
    offset = 0.5 if half_width == 0.5 else 0.0
    xs: list[float] = []
    ys: list[float] = []
    spans: list[tuple[int, int, bool]] = []
    for _ in range(rng.randint(1, 4)):
        start = len(xs)
        x, y = rng.randint(2, 44) + offset, rng.randint(2, 44) + offset
        xs.append(x)
        ys.append(y)
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.5:
                x = rng.randint(2, 44) + offset
            else:
                y = rng.randint(2, 44) + offset
            xs.append(x)
            ys.append(y)
        spans.append((start, len(xs), rng.random() < 0.3))
    path = deferred(xs, ys, spans)
    rgba = rng.choice(COLOURS)
    backdrop = rng.choice(["empty", "opaque", "mixed"])
    cap = rng.choice([0, 2])
    clip = rng.choice([None, "path", "wide", "wide"])

    def build() -> RasterTarget:
        target = make_target(48, 48, backdrop, 1.0, random.Random(seed))
        if clip is not None:
            push_clip(target, 48.0, clip)
        target.paint_window = []
        return target

    compare(
        build,
        lambda target: target.stroke_path(path, half_width * 2, rgba, None, None, cap, 0),
        monkeypatch,
    )


CONTENT = (
    b"0.9 0.9 0.2 rg 0 0 200 150 re f "
    b"1 J 1 j 2 w 0 0 1 RG 10 10 m 190 140 l 30 130 l S 10 10 m 10 10 l S "
    b"0 J 0 j 0.5 w 1 0 0 RG 20 20 m 180 20 l 180 120 l 20 120 l h S "
    b"2 J 4 w 0 0.5 0 RG 50 60 m 150 90 l S 50 70 m 150 100 l S 50 80 m 150 110 l S "
    b"q 30 30 120 80 re W n 3 w 0.2 0.2 0.2 RG 0 75 m 200 75 l 100 0 l 100 150 l S Q "
    b"q [6 3] 0 d 1 J 3 w 0 0 0 RG 15 100 m 185 40 l 185 140 l S Q "
    b"q /G gs 6 w 1 0 1 RG 10 140 m 190 10 l S Q "
    b"q /M gs 5 w 0 1 1 RG 20 30 m 180 130 l S Q "
    b"q 40 40 m 160 50 l 100 130 l h W n 2 w 0 0 0 RG 0 0 m 200 150 l S "
    b"1 j 8 w 0.4 0.1 0.7 RG 60 20 m 140 20 l 100 140 l h S Q"
)


def one_page_pdf(content: bytes) -> bytes:
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 150] "
        b"/Resources << /ExtGState << /G 5 0 R /M 6 0 R >> >> /Contents 4 0 R >>",
        4: b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        5: b"<< /Type /ExtGState /CA 0.5 >>",
        6: b"<< /Type /ExtGState /CA 0.8 /BM /Multiply >>",
    }
    data = bytearray(b"%PDF-1.7\n")
    offsets = {}
    for number, body in objects.items():
        offsets[number] = len(data)
        data += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 7\n0000000000 65535 f \n"
    data += b"".join(b"%010d 00000 n \n" % offsets[number] for number in range(1, 7))
    data += b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref
    return bytes(data)


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_a_page_strokes_as_the_python_walk_did(
    scale: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = one_page_pdf(CONTENT)

    def rendered() -> bytes:
        with PdfDocument(data) as document:
            return document.pages[0].render().rasterize(scale=scale).array().tobytes()

    native = rendered()
    monkeypatch.setattr(RasterTarget, "stroke_path", reference_stroke_path)
    assert rendered() == native

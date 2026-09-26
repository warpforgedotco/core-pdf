"""Small fills and stroke segments in any blend paint what their per-pixel loops did.

fill_path and fill_line each ended in a Python loop for what their kernels
did not take -- a blend other than normal, a group recording planes, a clip
that is not rectangles: sample each pixel 4x4, then blend_coverage_pixel.
The loops are gone. Their counts now come from supersampled_coverage_plane
and stroke_segment_samples, checked here against the sampling loops as they
were, and their blending from blend_coverage_counts, checked against
blend_coverage_pixel's blend_px call for each pixel, in every blend mode,
with and without group planes, including where blend_px raises part way.
"""

import random
from typing import Any

import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl.render import target as raster
from core_pdf.impl.render.paths import fill_path_crossing_spans
from core_pdf.impl.render.target import RasterTarget
from core_pdf_cythonized import stroke_segment_samples, supersampled_coverage_plane

OFFSETS = (0.125, 0.375, 0.625, 0.875)


def loop_path_counts(edges, box, view, fill_rule):
    """fill_path's sampling: four scanlines a row, four samples a pixel on each."""
    crop_x0, crop_y1, scale = view
    ix0, iy0, ix1, iy1 = box
    segments = [
        (ex0, ey0, ex1, ey1, min(ey1, ey0), max(ey0, ey1))
        for ex0, ey0, ex1, ey1 in edges
        if ey0 != ey1
    ]
    counts = numpy.zeros((iy1 - iy0, ix1 - ix0), dtype=numpy.uint8)
    for py in range(iy0, iy1):
        sample_spans = []
        for sy in range(4):
            page_y = crop_y1 - (py + (sy + 0.5) / 4) / scale
            crossings = []
            for ex0, ey0, ex1, ey1, low, high in segments:
                if low <= page_y < high:
                    t = (page_y - ey0) / (ey1 - ey0)
                    crossings.append((ex0 + t * (ex1 - ex0), 1 if ey1 > ey0 else -1))
            sample_spans.append(fill_path_crossing_spans(crossings, fill_rule))
        for px in range(ix0, ix1):
            covered = 0
            sample_x0 = crop_x0 + (px + 0.5 / 4) / scale
            sample_step = 1.0 / (4 * scale)
            for spans in sample_spans:
                for sx in range(4):
                    page_x = sample_x0 + sx * sample_step
                    if any(start <= page_x < end for start, end in spans):
                        covered += 1
            counts[py - iy0, px - ix0] = covered
    return counts


def loop_segment_counts(segment, half, round_cap, box, view, cap_extension):
    """fill_line's sampling of a diagonal segment."""
    crop_x0, crop_y1, scale = view
    x0, y0, x1, y1 = segment
    ix0, iy0, ix1, iy1 = box
    dx, dy = x1 - x0, y1 - y0
    seg_len2 = dx * dx + dy * dy
    half2 = half * half
    inv_seg_len2 = 1.0 / seg_len2
    projection_extension = cap_extension * seg_len2**0.5
    cross_limit = half2 * seg_len2
    counts = numpy.zeros((iy1 - iy0, ix1 - ix0), dtype=numpy.uint8)
    for py in range(iy0, iy1):
        page_ys = [crop_y1 - (py + offset) / scale for offset in OFFSETS]
        for px in range(ix0, ix1):
            page_xs = [crop_x0 + (px + offset) / scale for offset in OFFSETS]
            covered = 0
            for page_y in page_ys:
                offset_y = page_y - y0
                for page_x in page_xs:
                    offset_x = page_x - x0
                    cross = offset_x * dy - offset_y * dx
                    if not round_cap:
                        projection = offset_x * dx + offset_y * dy
                        if -projection_extension <= projection <= seg_len2 + projection_extension:
                            covered += cross * cross <= cross_limit
                        continue
                    t = (offset_x * dx + offset_y * dy) * inv_seg_len2
                    if 0.0 <= t <= 1.0:
                        covered += cross * cross <= cross_limit
                    elif t < 0.0:
                        covered += offset_x * offset_x + offset_y * offset_y <= half2
                    else:
                        end_x, end_y = page_x - x1, page_y - y1
                        covered += end_x * end_x + end_y * end_y <= half2
            counts[py - iy0, px - ix0] = covered
    return counts


@pytest.mark.parametrize("seed", range(60))
def test_path_counts_are_the_loops(seed: int) -> None:
    rng = random.Random(seed)
    points = [(rng.uniform(-2, 24), rng.uniform(-2, 24)) for _ in range(rng.randint(3, 9))]
    if rng.random() < 0.3:
        points = [(round(x), round(y)) for x, y in points]
    edges = [(*a, *b) for a, b in zip(points, points[1:] + points[:1], strict=True)]
    box = (rng.randint(0, 4), rng.randint(0, 4), rng.randint(10, 22), rng.randint(10, 22))
    view = (rng.uniform(-3, 3), rng.uniform(18, 26), rng.choice([1.0, 0.75, 1.5]))
    fill_rule = rng.choice(["nonzero", "evenodd"])
    expected = loop_path_counts(edges, box, view, fill_rule)
    sampled = supersampled_coverage_plane(
        numpy.asarray(edges, dtype=numpy.float64), *view, *box, fill_rule == "evenodd"
    )
    counts = numpy.zeros_like(expected)
    if sampled is not None:
        rows, first_row = sampled
        counts[first_row : first_row + len(rows)] = rows
    numpy.testing.assert_array_equal(counts, expected)


@pytest.mark.parametrize("seed", range(60))
def test_segment_counts_are_the_loops(seed: int) -> None:
    rng = random.Random(seed)
    segment = (rng.uniform(0, 12), rng.uniform(0, 12), rng.uniform(0, 12), rng.uniform(0, 12))
    view = (0.0, 12.0, rng.choice([1.0, 2.0, 0.75]))
    half = rng.choice([0.5, 1.0, 2.5])
    line_cap = rng.choice([0, 1, 2])
    cap_extension = half if line_cap == 2 else 0.0
    box = (0, 0, int(12 * view[2]), int(12 * view[2]))
    dx, dy = segment[2] - segment[0], segment[3] - segment[1]
    seg_len2 = dx * dx + dy * dy
    expected = loop_segment_counts(segment, half, line_cap == 1, box, view, cap_extension)
    counts = numpy.zeros_like(expected)
    stroke_segment_samples(
        numpy.zeros((box[3], box[2], 4), dtype=numpy.uint8),
        0,
        0,
        *box,
        view[0],
        view[1],
        view[2],
        *segment,
        dx,
        dy,
        seg_len2,
        1.0 / seg_len2,
        half * half,
        cap_extension * seg_len2**0.5,
        line_cap == 1,
        10,
        20,
        30,
        200,
        None,
        counts,
    )
    numpy.testing.assert_array_equal(counts, expected)


def reference_blend_counts(
    self: RasterTarget,
    counts: Any,
    left: int,
    top: int,
    allowed: bytearray | None,
    rgba: tuple[int, int, int, int],
    blend_mode: str | None,
) -> None:
    """blend_coverage_pixel for each covered pixel the clip lets through."""
    mode = self.resolved_blend(blend_mode)
    track_shape = self.group_source_shape is not None
    rows, columns = counts.shape
    for r in range(rows):
        for c in range(columns):
            covered = int(counts[r, c])
            if not covered or (allowed is not None and not allowed[r * columns + c]):
                continue
            alpha = max(0, min(255, round(rgba[3] * covered / 16)))
            shape = round(255 * covered / 16) if track_shape else 255
            offset = ((top + r) * self.width + left + c) * 4
            self.blend_px(offset, (rgba[0], rgba[1], rgba[2], alpha), mode, shape=shape)


def make_target(width: int, height: int, *, planes: bool) -> RasterTarget:
    pixels = bytearray(bytes([10, 200, 90, 180]) * (width * height))
    view = numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(height, width, 4)
    clip = raster.ClipState(crop_x0=0, crop_y1=height, scale=1, width=width, height=height)
    target = RasterTarget(
        pixels,
        None,
        clip=clip,
        width=width,
        height=height,
        scale=1,
        crop_x0=0,
        crop_y0=0,
        crop_y1=height,
        page_view=view,
    )
    if planes:
        target.group_source_alpha = numpy.full((height, width), 0.25, dtype=numpy.float32)
        target.group_source_shape = numpy.full((height, width), 0.5, dtype=numpy.float32)
        target.paint_window = []
    return target


def state(target: RasterTarget) -> tuple[Any, ...]:
    return (
        bytes(target.pixels),
        None if target.group_source_alpha is None else target.group_source_alpha.tobytes(),
        None if target.group_source_shape is None else target.group_source_shape.tobytes(),
        None if target.paint_window is None else list(target.paint_window),
    )


MODES = [None, "Normal", "Multiply", "Screen", "ColorDodge", "ColorBurn", "Overlay"]


@pytest.mark.parametrize("planes", [False, True])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("version", [None, (1, 4), (2, 0)])
@pytest.mark.parametrize("masked", [False, True])
def test_blended_counts_are_blend_coverage_pixels(
    planes: bool, mode: str | None, version: tuple[int, int] | None, masked: bool
) -> None:
    from core_pdf_spec.standards import PdfVersion, SemanticContext

    rng = numpy.random.default_rng(len(MODES) * planes + (version or (0, 0))[0])
    counts = rng.integers(0, 17, size=(5, 7), dtype=numpy.uint8)
    counts[rng.random((5, 7)) < 0.3] = 0
    allowed = bytearray((rng.random(35) < 0.7).astype(numpy.uint8).tobytes()) if masked else None
    outcomes = []
    for blend in (RasterTarget.blend_counts, reference_blend_counts):
        target = make_target(10, 8, planes=planes)
        target.semantic_context = SemanticContext(None if version is None else PdfVersion(*version))
        try:
            blend(target, counts, 2, 1, allowed, (250, 30, 120, 170), mode)
            raised = None
        except Exception as error:  # noqa: BLE001 -- the comparison covers failures too
            raised = (type(error), str(error))
        outcomes.append((raised, *state(target)))
    assert outcomes[0] == outcomes[1]


def one_page_pdf(content: bytes, form: bytes) -> bytes:
    resources = b"/ExtGState << /M 6 0 R /D 7 0 R /S 8 0 R >>"
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 120 90] /Resources << "
        + resources
        + b" /XObject << /X 5 0 R >> >> /Contents 4 0 R >>",
        4: b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        5: b"<< /Type /XObject /Subtype /Form /BBox [0 0 120 90] /Group << /S /Transparency "
        b"/I false /K true >> /Resources << "
        + resources
        + b" >> /Length %d >>\nstream\n" % len(form)
        + form
        + b"\nendstream",
        6: b"<< /Type /ExtGState /ca 0.8 /CA 0.7 /BM /Multiply >>",
        7: b"<< /Type /ExtGState /ca 0.7 /CA 0.9 /BM /ColorDodge >>",
        8: b"<< /Type /ExtGState /BM /Screen >>",
    }
    data = bytearray(b"%PDF-1.7\n")
    offsets = {}
    for number, body in objects.items():
        offsets[number] = len(data)
        data += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 9\n0000000000 65535 f \n"
    data += b"".join(b"%010d 00000 n \n" % offsets[number] for number in range(1, 9))
    data += b"trailer\n<< /Size 9 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref
    return bytes(data)


BACKDROP = b"0.2 0.4 0.6 rg 0 0 120 90 re f 0.9 0.8 0.1 rg 30 20 50 40 re f "
SHAPES = (
    b"0.8 0.1 0.3 rg 10 10 m 40 70 l 55 12 l h f "
    b"0.1 0.7 0.2 rg 60 10 m 110 80 l 70 60 l 100 20 l h f* "
    b"0 0 1 RG 1.5 w 1 J 5 5 m 115 85 l 20 80 l S "
    b"2 J 0.4 w 10 60 m 80 5 l S "
)
CONTENTS = [
    pytest.param(BACKDROP + b"q /M gs " + SHAPES + b"Q", b"", id="multiply"),
    pytest.param(BACKDROP + b"q /D gs " + SHAPES + b"Q", b"", id="color-dodge"),
    pytest.param(
        BACKDROP + b"q 60 45 m 110 85 l 115 5 l h W n /S gs " + SHAPES + b"Q",
        b"",
        id="screen-under-path-clip",
    ),
    pytest.param(
        BACKDROP + b"q 60 45 m 110 85 l 115 5 l h W n " + SHAPES + b"Q", b"", id="path-clip"
    ),
    pytest.param(BACKDROP + b"/X Do", SHAPES + b"/M gs " + SHAPES, id="knockout-group"),
]


def rendered(data: bytes) -> bytes:
    with PdfDocument(data) as document:
        return document.pages[0].render().rasterize(scale=1.5).array().tobytes()


@pytest.mark.parametrize(("content", "form"), CONTENTS)
def test_pages_paint_as_blend_coverage_pixel_did(
    content: bytes, form: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = one_page_pdf(content, form)
    compiled = rendered(data)
    calls: list[object] = []
    original = RasterTarget.blend_counts

    def counted(self: RasterTarget, *args: Any) -> None:
        calls.append(args)
        original(self, *args)

    monkeypatch.setattr(RasterTarget, "blend_counts", counted)
    assert rendered(data) == compiled
    assert calls
    monkeypatch.setattr(RasterTarget, "blend_counts", reference_blend_counts)
    assert rendered(data) == compiled

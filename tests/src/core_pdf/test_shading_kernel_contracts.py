"""Shadings painted by the kernels paint what paint_shading's per-pixel loop did.

The reference below is that loop as it was, with the parameter taken from
shading_t -- pinned to the deleted Python functions by golden vectors -- and
blending by blend_px, which the rest of the rasterizer still uses. Pages
cover axial and radial shadings, extends, clips, opacity, every blend mode
blend_px treats apart, and knockout and non-isolated groups that record
source alpha and shape; failures part way are checked on a target directly.
"""

import math
import random
from typing import Any

import numpy
import pytest

from core_pdf.impl import render_target as raster
from core_pdf.impl.graphics_shading import prepare_shading
from core_pdf.impl.render_target import RasterTarget
from core_pdf_cythonized import shading_t, shading_values
from core_pdf_spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING
from tests.src.core_pdf.pdf_bytes import serialize_pdf
from tests.src.core_pdf.raster_support import make_backdrop_target, rendered


def reference_paint_shading(
    self: RasterTarget, data: dict[str, Any], blend_mode: str | None
) -> None:
    shading = prepare_shading(
        data.get("dictionary"), rendering=data.get("color_rendering", DEFAULT_COLOR_RENDERING)
    )
    if shading is None:
        return
    clipped_box = self.clip.clipped_pixel_box(self.shading_box(data, shading))
    if clipped_box is None:
        return
    ix0, iy0, ix1, iy1 = clipped_box[1]
    soft_mask_alpha = data.get("soft_mask_alpha")
    fill_opacity = data.get("fill_opacity")
    shading_alpha = float(soft_mask_alpha) if is_pdf_number(soft_mask_alpha) else None
    domain = shading.domain
    domain_span = domain[1] - domain[0]
    mode = self.resolved_blend(blend_mode)
    kind = shading.shading_type
    cache: dict[float, tuple[int, int, int, int]] = {}
    for py in range(iy0, iy1):
        page_y = self.crop_y1 - (py + 0.5) / self.scale
        row = py * self.width * 4
        for span_start, span_end in self.clip.clip_row_visible_spans(py):
            for px in range(max(ix0, span_start), min(ix1, span_end)):
                page_x = self.crop_x0 + (px + 0.5) / self.scale
                unit_t = shading_t(kind, shading.coords, page_x, page_y, 0.5)
                if unit_t is None:
                    continue
                if unit_t < 0.0:
                    if not shading.extend_start:
                        continue
                    unit_t = 0.0
                elif unit_t > 1.0:
                    if not shading.extend_end:
                        continue
                    unit_t = 1.0
                value = domain[0] + unit_t * domain_span
                rgba = cache.get(value)
                if rgba is None:
                    rgba = cache[value] = raster.shading_rgba(
                        shading.color_model,
                        shading.evaluator(value),
                        fill_opacity,
                        shading.color_rendering,
                        shading_alpha,
                    )
                self.blend_px(row + px * 4, rgba, mode)


def loop_values(kind, coords, box, allowed, extend, domain, view):
    crop_x0, crop_y1, scale = view
    ix0, iy0, ix1, iy1 = box
    painted = numpy.zeros((iy1 - iy0, ix1 - ix0), dtype=numpy.uint8)
    values: list[float] = []
    for py in range(iy0, iy1):
        page_y = crop_y1 - (py + 0.5) / scale
        for px in range(ix0, ix1):
            if not allowed[py - iy0, px - ix0]:
                continue
            unit_t = shading_t(kind, coords, crop_x0 + (px + 0.5) / scale, page_y, 0.5)
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


def one_page_pdf(shading: bytes, content: bytes, form: bytes) -> bytes:
    resources = b"/Shading << /S 5 0 R >> /ExtGState << /G 6 0 R /M 8 0 R /K 9 0 R >>"
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 120 90] /Resources << "
        + resources
        + b" /XObject << /X 7 0 R >> >> /Contents 4 0 R >>",
        4: b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        5: shading,
        6: b"<< /Type /ExtGState /ca 0.6 /BM /Normal >>",
        7: b"<< /Type /XObject /Subtype /Form /BBox [0 0 120 90] /Group << /S /Transparency "
        b"/I false /K true >> /Resources << "
        + resources
        + b" >> /Length %d >>\nstream\n" % len(form)
        + form
        + b"\nendstream",
        8: b"<< /Type /ExtGState /ca 0.8 /BM /Multiply >>",
        9: b"<< /Type /ExtGState /ca 0.7 /BM /ColorDodge >>",
    }
    return serialize_pdf(objects, b"1.7")


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
BACKDROP = b"0.2 0.4 0.6 rg 0 0 120 90 re f 0.9 0.8 0.1 rg 30 20 50 40 re f "
CONTENTS = [
    pytest.param(b"/S sh", b"", id="page"),
    pytest.param(b"q 20 15 70 50 re W n /S sh Q", b"", id="clipped"),
    pytest.param(b"q /G gs 20 15 70 50 re W n /S sh Q", b"", id="translucent"),
    pytest.param(BACKDROP + b"q /M gs /S sh Q", b"", id="multiply"),
    pytest.param(BACKDROP + b"q /K gs /S sh Q", b"", id="color-dodge"),
    pytest.param(BACKDROP + b"/X Do", b"/G gs /S sh", id="knockout-group"),
    pytest.param(
        BACKDROP + b"/X Do", b"q 10 10 60 60 re W n /M gs /S sh Q /S sh", id="group-modes"
    ),
]


@pytest.mark.parametrize("shading", SHADINGS)
@pytest.mark.parametrize(("content", "form"), CONTENTS)
def test_kernel_paint_is_the_loops(
    shading: bytes, content: bytes, form: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = one_page_pdf(shading, content, form)
    compiled = rendered(data)
    monkeypatch.setattr(RasterTarget, "paint_shading", reference_paint_shading)
    assert rendered(data) == compiled


def plane_bytes(plane: numpy.ndarray | None) -> bytes | None:
    return None if plane is None else plane.tobytes()


SHADING = {
    "ShadingType": 2,
    "ColorSpace": "DeviceRGB",
    "Coords": [0, 0, 12, 0],
    "Extend": [True, True],
    "Function": {"FunctionType": 2, "Domain": [0, 1], "C0": [0, 0, 0], "C1": [1, 1, 1], "N": 1},
}


@pytest.mark.parametrize("planes", [False, True])
@pytest.mark.parametrize("opacity", [None, 0.0, 0.5])
def test_a_colour_failing_part_way_paints_what_the_loop_painted(
    planes: bool, opacity: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core_pdf.impl.types import PdfName

    shading = {PdfName.of(k.encode()): v for k, v in SHADING.items()}
    original = raster.shading_rgba
    seen: list[object] = []

    def failing(model: str, components: Any, *rest: Any) -> Any:
        seen.append(components)
        if len(seen) == 5:
            raise ArithmeticError("no colour")
        return original(model, components, *rest)

    monkeypatch.setattr(raster, "shading_rgba", failing)
    outcomes = []
    for paint in (RasterTarget.paint_shading, reference_paint_shading):
        seen.clear()
        target = make_backdrop_target(12, 3, planes=planes)
        with pytest.raises(ArithmeticError, match="no colour"):
            paint(target, {"dictionary": shading, "fill_opacity": opacity}, "Multiply")
        outcomes.append(
            (
                bytes(target.pixels),
                plane_bytes(target.group_source_alpha),
                plane_bytes(target.group_source_shape),
                list(target.paint_window) if target.paint_window is not None else None,
            )
        )
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("planes", [False, True])
@pytest.mark.parametrize("mode", ["ColorDodge", "ColorBurn", "Screen"])
@pytest.mark.parametrize("version", [None, (1, 4), (2, 0)])
def test_blending_rules_follow_the_documents_version(
    planes: bool, mode: str, version: tuple[int, int] | None
) -> None:
    from core_pdf.impl.types import PdfName
    from core_pdf_spec.standards import PdfVersion, SemanticContext

    shading = {PdfName.of(k.encode()): v for k, v in SHADING.items()}
    context = SemanticContext(None if version is None else PdfVersion(*version))
    outcomes = []
    for paint in (RasterTarget.paint_shading, reference_paint_shading):
        target = make_backdrop_target(12, 3, planes=planes)
        target.semantic_context = context
        try:
            paint(target, {"dictionary": shading, "fill_opacity": 0.8}, mode)
            raised = None
        except Exception as error:  # noqa: BLE001 -- the comparison covers failures too
            raised = (type(error), str(error))
        outcomes.append(
            (
                raised,
                bytes(target.pixels),
                plane_bytes(target.group_source_alpha),
                plane_bytes(target.group_source_shape),
                list(target.paint_window) if target.paint_window is not None else None,
            )
        )
    assert outcomes[0] == outcomes[1]
    if version is None and mode != "Screen":
        assert outcomes[0][0] is not None


def test_a_shading_painted_again_is_prepared_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from core_pdf.impl.types import PdfName

    shading = {PdfName.of(k.encode()): v for k, v in SHADING.items()}
    prepared: list[object] = []
    original = raster.prepare_shading

    def counting(dictionary: object, **options: Any) -> Any:
        prepared.append(dictionary)
        return original(dictionary, **options)

    monkeypatch.setattr(raster, "prepare_shading", counting)
    target = make_backdrop_target(12, 3, planes=False)
    for _ in range(3):
        target.paint_shading({"dictionary": shading}, None)
    assert prepared == [shading]
    sibling, _ = target.blank_sibling()
    assert sibling.prepared_shading_cache is target.prepared_shading_cache


def test_a_shading_painted_again_is_captured_once() -> None:
    from core_pdf import PdfDocument

    shading = (
        b"<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [10 10 100 70] "
        b"/Function " + FUNCTION + b" >>"
    )
    data = one_page_pdf(shading, b"q 20 15 70 50 re W n /S sh Q /S sh", b"")
    with PdfDocument(data) as document:
        program = document.pages[0].get_page_program()
    dictionaries = [drawing.dictionary for drawing in program.drawings if drawing.kind == "shading"]
    assert len(dictionaries) == 2
    assert dictionaries[0] is dictionaries[1]

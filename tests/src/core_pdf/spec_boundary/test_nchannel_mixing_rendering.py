# SPDX-License-Identifier: AGPL-3.0-only
"""The selected NChannel screen approximation reaches complete PDF paint paths."""

from dataclasses import dataclass
from typing import Any

import imagecodecs
import numpy
import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.graphics.device_profiles import default_cmyk_transform
from core_pdf.impl._impl.render.model import PathPaintItem
from core_pdf_spec.s_08_graphics.color_rendering import ColorRendering

PAINTS = (
    "path",
    "image8",
    "image16",
    "colored-pattern",
    "uncolored-pattern",
    "indexed-path",
    "indexed-image",
    "shading",
    "form",
)


@dataclass(frozen=True)
class internal_MixCase:
    space: bytes
    values: tuple[float, ...]
    expected: tuple[int, int, int]


def internal_spot(name: bytes, endpoint: bytes) -> bytes:
    return (
        b"/"
        + name
        + b" [/Separation /"
        + name
        + b" /DeviceRGB << /FunctionType 2 /Domain [0 1] /C0 [1 1 1] /C1 ["
        + endpoint
        + b"] /N 1 >>]"
    )


def internal_process_space(process: bytes = b"/DeviceRGB", *, hints: bytes = b"") -> bytes:
    cmyk = process != b"/DeviceRGB"
    names = b"/Black /YellowSpot" if cmyk else b"/Red /Green /Blue /YellowSpot"
    components = b"/Cyan /Magenta /Yellow /Black" if cmyk else b"/Red /Green /Blue"
    return (
        b"[/DeviceN ["
        + names
        + b"] /DeviceRGB 8 0 R << /Subtype /NChannel /Process << /ColorSpace "
        + process
        + b" /Components ["
        + components
        + b"] >> /Colorants << "
        + internal_spot(b"YellowSpot", b"1 1 0")
        + b" >> "
        + hints
        + b" >>]"
    )


INTERIOR = internal_MixCase(internal_process_space(), (0.4, 0.8, 0.6, 0.6), (102, 204, 61))
RED = internal_MixCase(internal_process_space(), (1, 0, 0, 1), (255, 0, 0))
ALL_SPOT = internal_MixCase(
    b"[/DeviceN [/CyanSpot /MagentaSpot] /DeviceRGB 8 0 R << /Subtype /NChannel /Colorants << "
    + internal_spot(b"CyanSpot", b"0 1 1")
    + b" "
    + internal_spot(b"MagentaSpot", b"1 0 1")
    + b" >> >>]",
    (0.4, 0.6),
    (153, 102, 255),
)


def internal_stream(data: bytes, entries: bytes = b"") -> bytes:
    return b"<< " + entries + f" /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"


def internal_numbers(values: tuple[float, ...]) -> bytes:
    return b" ".join(f"{value:.12g}".encode() for value in values)


def internal_document(
    case: internal_MixCase,
    paint: str,
    *,
    opacity: float = 1,
    repeated: bool = False,
    image_intent: bool = False,
    content_prefix: bytes = b"",
    after_paint: bytes = b"",
    pattern_repeats: int = 1,
    pattern_step: int = 10,
    pattern_prefix: bytes = b"",
    extra_extgstate: bytes = b"",
) -> bytes:
    space, values = case.space, case.values
    count = len(values)
    if paint.startswith("indexed-"):
        palette = bytes(round(value * 255) for value in values).hex().encode()
        space = b"[/Indexed " + space + b" 0 <" + palette + b">]"
        values = (0,)
    operands = internal_numbers(values)
    resources = b""
    extra = b"<< >>"
    operation = b"/C cs " + operands + b" scn 0 0 10 10 re f"
    if paint in {"image8", "image16", "indexed-image"}:
        depth = 16 if paint == "image16" else 8
        data = b"".join(
            round(value * ((1 << depth) - 1)).to_bytes(depth // 8, "big") for value in values
        )
        extra = internal_stream(
            data,
            f"/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent {depth} ".encode()
            + b"/ColorSpace 5 0 R "
            + (b"/Intent /AbsoluteColorimetric" if image_intent else b""),
        )
        resources = b"/XObject << /Im 6 0 R >>"
        operation = b"10 0 0 10 0 0 cm /Im Do"
    elif paint in {"colored-pattern", "uncolored-pattern"}:
        colored = paint == "colored-pattern"
        extra = internal_stream(
            pattern_prefix
            + (b"/C cs " + operands + b" scn " if colored else b"")
            + b"0 0 10 10 re f " * pattern_repeats,
            b"/Type /Pattern /PatternType 1 "
            + (b"/PaintType 1 " if colored else b"/PaintType 2 ")
            + f"/TilingType 1 /BBox [0 0 10 10] /XStep {pattern_step} /YStep 10 ".encode()
            + b"/Resources << /ColorSpace << /C 5 0 R >> /ExtGState << "
            + extra_extgstate
            + b" >> >>",
        )
        resources = b"/Pattern << /P 6 0 R >>"
        operation = (
            b"/Pattern cs /P scn " if colored else b"/U cs " + operands + b" /P scn "
        ) + b"0 0 10 10 re f"
    elif paint == "shading":
        extra = (
            b"<< /ShadingType 2 /ColorSpace 5 0 R /Coords [0 0 10 0] /Extend [true true] "
            b"/Function << /FunctionType 2 /Domain [0 1] /C0 ["
            + operands
            + b"] /C1 ["
            + operands
            + b"] /N 1 >> >>"
        )
        resources = b"/Shading << /S 6 0 R >>"
        operation = b"0 0 10 10 re W n /S sh"
    elif paint == "form":
        extra = internal_stream(
            operation + b" /AbsoluteColorimetric ri",
            b"/Type /XObject /Subtype /Form /BBox [0 0 10 10] "
            b"/Resources << /ColorSpace << /C 5 0 R >> >>",
        )
        resources = b"/XObject << /F 6 0 R >>"
        operation = b"/F Do"
    elif paint == "text":
        extra = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        resources = b"/Font << /F 6 0 R >>"
        operation = b"/C cs " + operands + b" scn BT /F 10 Tf 1 0 0 1 0 1 Tm (H) Tj ET"
    content = (
        b"/Opacity gs "
        + content_prefix
        + (
            b" ".join(
                f"q /{setting} gs 1 0 0 1 {index * 10} 0 cm ".encode() + operation + b" Q"
                for index, setting in enumerate(("Off", "On", "Off", "Abs"))
            )
            if repeated
            else operation
        )
        + after_paint
    )
    profile = default_cmyk_transform()
    assert profile is not None
    bodies = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 40 10] /Resources << "
        b"/ColorSpace << /C 5 0 R /U [/Pattern 5 0 R] >> "
        b"/ExtGState << /Off << /UseBlackPtComp /OFF >> /On << /UseBlackPtComp /ON >> "
        b"/Abs << /RI /AbsoluteColorimetric /UseBlackPtComp /ON >> /Opacity << /ca "
        + f"{opacity}".encode()
        + b" >> "
        + extra_extgstate
        + b" >> "
        + resources
        + b" >> /Contents 4 0 R >>",
        internal_stream(content),
        space,
        extra,
        internal_stream(profile.profile, b"/N 4 /Alternate /DeviceCMYK"),
        internal_stream(
            b"\x00\x00\xff",
            b"/FunctionType 0 /Domain ["
            + b"0 1 " * count
            + b"] /Size ["
            + b"1 " * count
            + b"] /BitsPerSample 8 /Encode ["
            + b"0 0 " * count
            + b"] /Range [0 1 0 1 0 1]",
        ),
    ]
    result = b"%PDF-2.0\n"
    offsets = [0]
    for number, body in enumerate(bodies, 1):
        offsets.append(len(result))
        result += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(result)
    result += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    result += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    return (
        result
        + f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )


def internal_raster(data: bytes) -> numpy.ndarray:
    with PdfDocument(data) as document:
        return document.pages[0].render().rasterize(background=(255, 255, 255, 255)).array().copy()


@pytest.mark.parametrize(
    "case", [INTERIOR, RED, ALL_SPOT], ids=["interior-process-spot", "red-yellow", "all-spot"]
)
@pytest.mark.parametrize("paint", PAINTS)
def test_mixed_and_all_spot_colors_reach_pixels(case: internal_MixCase, paint: str) -> None:
    assert case.expected != (0, 0, 255)  # Deliberately contradict the global tint.
    actual = internal_raster(internal_document(case, paint))[5, 5, :3]
    numpy.testing.assert_array_equal(actual, case.expected)


@pytest.mark.parametrize("case", [INTERIOR, ALL_SPOT], ids=["mixed", "all-spot"])
def test_text_uses_mixed_color_at_glyph_paint_time(case: internal_MixCase) -> None:
    raster = internal_raster(internal_document(case, "text"))
    direct = internal_MixCase(
        b"/DeviceRGB", tuple(value / 255 for value in case.expected), case.expected
    )
    expected = internal_raster(internal_document(direct, "text"))
    assert numpy.any(raster[:, :10, :3] != 255)
    numpy.testing.assert_array_equal(raster, expected)


@pytest.mark.parametrize("paint", PAINTS)
def test_mixed_color_object_opacity_is_applied_once(paint: str) -> None:
    actual = internal_raster(internal_document(INTERIOR, paint, opacity=0.5))[5, 5, :3]
    expected = (numpy.asarray(INTERIOR.expected) + 255) / 2
    numpy.testing.assert_allclose(actual, expected, atol=1, rtol=0)


@pytest.mark.parametrize("paint", PAINTS)
def test_mixed_icc_process_preserves_intent_bpc_and_form_inheritance(
    paint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = internal_MixCase(internal_process_space(b"[/ICCBased 7 0 R]"), (1, 0.6), (0, 0, 0))
    profile = default_cmyk_transform()
    assert profile is not None
    settings = [
        ColorRendering(black_point_compensation="OFF"),
        ColorRendering(black_point_compensation="ON"),
        ColorRendering(black_point_compensation="OFF"),
        ColorRendering("AbsoluteColorimetric", "ON"),
    ]
    expected = []
    for setting in settings:
        process = profile.apply_uint16(
            numpy.array([[0, 0, 0, 65535]], dtype=numpy.uint16), rendering=setting
        )[0]
        expected.append(numpy.rint(process * numpy.array([1, 1, 0.4])).astype(numpy.uint8))
    calls: list[tuple[int, int]] = []
    original = imagecodecs.cms_transform

    def record(*args: Any, **kwargs: Any) -> Any:
        calls.append((kwargs["intent"], kwargs["flags"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(imagecodecs, "cms_transform", record)
    raster = internal_raster(internal_document(case, paint, repeated=True))
    numpy.testing.assert_array_equal(raster[5, [5, 15, 25, 35], :3], expected)
    assert {(1, 0x0100), (1, 0x2100), (3, 0x0100)} <= set(calls)


@pytest.mark.parametrize("depth", [8, 16])
def test_mixed_image_intent_override_disables_compensation(
    depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = internal_MixCase(internal_process_space(b"[/ICCBased 7 0 R]"), (1, 0.6), (0, 0, 0))
    calls: list[tuple[int, int]] = []
    original = imagecodecs.cms_transform

    def record(*args: Any, **kwargs: Any) -> Any:
        calls.append((kwargs["intent"], kwargs["flags"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(imagecodecs, "cms_transform", record)
    raster = internal_raster(
        internal_document(case, f"image{depth}", repeated=True, image_intent=True)
    )
    assert calls
    assert set(calls) == {(3, 0x0100)}
    for position in (15, 25, 35):
        numpy.testing.assert_array_equal(raster[5, position, :3], raster[5, 5, :3])


def test_form_local_rendering_intent_does_not_leak_to_following_paint() -> None:
    case = internal_MixCase(internal_process_space(b"[/ICCBased 7 0 R]"), (1, 0.6), (0, 0, 0))
    # The Form switches to AbsoluteColorimetric after painting. The following
    # parent path must still use its original relative/OFF settings.
    data = internal_document(
        case,
        "form",
        content_prefix=b"/Off gs ",
        after_paint=b" 1 0 0 1 10 0 cm /C cs 1 0.6 scn 0 0 10 10 re f",
    )
    raster = internal_raster(data)
    profile = default_cmyk_transform()
    assert profile is not None
    process = profile.apply_uint16(
        numpy.array([[0, 0, 0, 65535]], dtype=numpy.uint16),
        rendering=ColorRendering(black_point_compensation="OFF"),
    )[0]
    expected = numpy.rint(process * numpy.array([1, 1, 0.4])).astype(numpy.uint8)
    numpy.testing.assert_array_equal(raster[5, 5, :3], expected)
    numpy.testing.assert_array_equal(raster[5, 15, :3], expected)


@pytest.mark.parametrize("hints", [b"", b"/MixingHints << >>"])
def test_empty_mixing_hints_retain_selected_screen_policy(hints: bytes) -> None:
    case = internal_MixCase(internal_process_space(hints=hints), INTERIOR.values, INTERIOR.expected)
    numpy.testing.assert_array_equal(
        internal_raster(internal_document(case, "path"))[5, 5, :3], case.expected
    )


def test_nonempty_mixing_hints_select_whole_global_fallback() -> None:
    hints = (
        b"/MixingHints << /Solidities << /YellowSpot 1 >> "
        b"/PrintingOrder [/Red /Green /Blue /YellowSpot] >>"
    )
    case = internal_MixCase(internal_process_space(hints=hints), INTERIOR.values, (0, 0, 255))
    for paint in ("path", "image8", "image16", "indexed-path", "shading"):
        numpy.testing.assert_array_equal(
            internal_raster(internal_document(case, paint))[5, 5, :3], case.expected
        )


@pytest.mark.parametrize("paint", ["colored-pattern", "uncolored-pattern"])
@pytest.mark.parametrize(
    "case",
    [INTERIOR, internal_MixCase(b"/DeviceRGB", (0.4, 0.8, 0.6), (102, 204, 153))],
    ids=["mixed-nchannel", "device-rgb"],
)
def test_tiling_overlap_gets_one_outer_opacity_and_preserves_clip(
    case: internal_MixCase, paint: str
) -> None:
    # Both the cell's marks and adjacent cells overlap; outer opacity still
    # applies once to their complete source, not separately to each mark/tile.
    raster = internal_raster(
        internal_document(
            case,
            paint,
            opacity=0.5,
            pattern_repeats=2,
            pattern_step=5,
            content_prefix=b"2 2 6 6 re W n ",
        )
    )
    expected = (numpy.asarray(case.expected) + 255) / 2
    numpy.testing.assert_allclose(raster[5, 5, :3], expected, atol=1, rtol=0)
    numpy.testing.assert_array_equal(raster[1, 1], [255, 255, 255, 255])
    numpy.testing.assert_array_equal(raster[5, 15], [255, 255, 255, 255])


@pytest.mark.parametrize("paint", ["colored-pattern", "uncolored-pattern"])
def test_tiling_scalar_soft_mask_multiplies_outer_opacity_once(paint: str) -> None:
    data = internal_document(INTERIOR, paint, opacity=0.5, pattern_repeats=2)
    with PdfDocument(data) as document:
        page = document.pages[0].render()
        # Exercise the renderer's existing scalar soft-mask input independently
        # of source soft-mask interpretation (which is a separate capture path).
        item = next(
            item
            for item in page.display_list.items
            if isinstance(item, PathPaintItem) and item.fill_pattern is not None
        )
        item.soft_mask_alpha = 0.5
        raster = page.rasterize(background=(255, 255, 255, 255)).array()
    expected = numpy.asarray(INTERIOR.expected) * 0.25 + 255 * 0.75
    numpy.testing.assert_allclose(raster[5, 5, :3], expected, atol=1, rtol=0)


@pytest.mark.parametrize("paint", ["colored-pattern", "uncolored-pattern"])
def test_tiling_outer_blend_applies_after_overlapping_normal_marks(paint: str) -> None:
    case = internal_MixCase(b"/DeviceRGB", (0.4, 0.8, 0.6), (102, 204, 153))
    raster = internal_raster(
        internal_document(
            case,
            paint,
            pattern_repeats=2,
            extra_extgstate=b"/Screen << /BM /Screen >>",
            content_prefix=b"0 0 1 rg 0 0 40 10 re f /Screen gs ",
        )
    )
    # Screen against blue preserves red/green and saturates blue. Applying it
    # to each overlapping mark would incorrectly boost red and green twice.
    numpy.testing.assert_array_equal(raster[5, 5, :3], [102, 204, 255])


@pytest.mark.parametrize("paint", ["colored-pattern", "uncolored-pattern"])
def test_nonnormal_pattern_children_preserve_backdrop_dependent_path(paint: str) -> None:
    case = internal_MixCase(b"/DeviceRGB", (0.4, 0.8, 0.6), (102, 204, 153))
    raster = internal_raster(
        internal_document(
            case,
            paint,
            pattern_prefix=b"/Screen gs ",
            extra_extgstate=b"/Screen << /BM /Screen >>",
            content_prefix=b"0 0 1 rg 0 0 40 10 re f ",
        )
    )
    # A non-Normal cell must still see its blue backdrop; a fresh isolated
    # buffer would turn this into [102,204,153]. Outer-alpha support for this
    # general non-isolated case remains a separate renderer limitation.
    numpy.testing.assert_array_equal(raster[5, 5, :3], [102, 204, 255])

# SPDX-License-Identifier: AGPL-3.0-only
"""Knockout metadata and independent paint boundaries survive reader capture."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.capture.program import CapturedProgram
from core_pdf.impl._impl.capture.records import CapturedDrawing, CapturedPath, TilingPattern
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf.impl._impl.model.glyphs import GlyphObservation
from core_pdf.impl._impl.render.commands import append_captured_program
from core_pdf.impl._impl.render.display import DisplayList
from core_pdf.impl._impl.render.model import DisplayListItem, ImagePaintItem, PathPaintItem
from core_pdf_spec.s_07_content.model import TilingPattern as PdfTilingPattern
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.types import PdfName


def internal_state() -> TextState:
    resolver = ObjectResolver(b"", {})
    return TextState(SimpleNamespace(resolver=resolver, resolve=resolver.resolve))


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("knockout", [False, True])
@pytest.mark.parametrize("initial_AIS", [False, True])
def test_group_capture_retains_independent_K_and_outer_AIS(
    isolated: bool, knockout: bool, initial_AIS: bool
) -> None:
    state = internal_state()
    state.graphics.alpha_is_shape = initial_AIS
    state.graphics.fill_opacity = 0.3
    form = PdfStream(
        raw_data=b"0 0 1 1 re f /Other gs 0 0 1 1 re f",
        dictionary={
            "Subtype": PdfName.of("Form"),
            "BBox": [0, 0, 1, 1],
            "Group": {"S": PdfName.of("Transparency"), "I": isolated, "K": knockout},
            "Resources": {"ExtGState": {"Other": {"AIS": not initial_AIS}}},
        },
    )
    state.stream_executor.consume(
        PdfStream(raw_data=b"/F Do"), {"XObject": {"F": form}}, IDENTITY_MATRIX, 0
    )
    _, begin, first, second, end, _ = state.drawings
    assert begin.group_knockout is end.group_knockout is knockout
    assert begin.group_isolated is end.group_isolated is isolated
    assert begin.alpha_is_shape is first.alpha_is_shape is end.alpha_is_shape is initial_AIS
    assert second.alpha_is_shape is not initial_AIS
    assert begin.fill_opacity == end.fill_opacity == 0.3
    assert first.fill_opacity == second.fill_opacity == 1.0
    assert state.graphics.alpha_is_shape is initial_AIS
    display = DisplayList(1, 1)
    for drawing in state.drawings:
        display.append_captured_drawing(drawing)
    groups = [item for item in display.items if item.kind in {"group-begin", "group-end"}]
    assert len(groups) == 2
    for item in groups:
        assert isinstance(item, DisplayListItem)
        assert item.data["group_knockout"] is knockout
        assert item.data["alpha_is_shape"] is initial_AIS


def internal_stroke(seqno: int, *, alpha_is_shape: bool = False) -> CapturedDrawing:
    path = CapturedPath()
    path.rect(0, 0, 2, 2)
    return CapturedDrawing(
        seqno=seqno,
        kind="stroke",
        fill=None,
        fill_opacity=None,
        path=path,
        stroke_color=(1, 0, 0),
        stroke_opacity=0.5,
        alpha_is_shape=alpha_is_shape,
    )


@pytest.mark.parametrize("knockout", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_stroke_coalescing_retains_objects_under_any_knockout_ancestor(
    knockout: bool, nested: bool
) -> None:
    # ISO 32000-2 11.4.6: distinct elements knock out one another, so merging
    # their paths would discard the compositing boundary in an overlap.
    display = DisplayList(2, 2)
    display.append("group-begin", 0, group_knockout=knockout)
    if nested:
        display.append("scope-begin", 1)
        display.append("group-begin", 2, group_knockout=False)
    display.append_captured_drawing(internal_stroke(3))
    display.append_captured_drawing(internal_stroke(4))
    assert sum(isinstance(item, PathPaintItem) for item in display.items) == (2 if knockout else 1)
    if nested:
        display.append("group-end", 5)
        display.append("scope-end", 6)
    display.append("group-end", 7)
    display.append_captured_drawing(internal_stroke(8))
    display.append_captured_drawing(internal_stroke(9))
    assert isinstance(display.items[-1], PathPaintItem)
    assert display.items[-1].coalesced_path


def test_coalescing_restores_scope_group_floor_after_unbalanced_group_markers() -> None:
    display = DisplayList(2, 2)
    display.append("group-begin", 0, group_knockout=True)
    display.append("scope-begin", 1)
    display.append("group-end", 2)  # Cannot close the parent group's state.
    display.append_captured_drawing(internal_stroke(3))
    display.append_captured_drawing(internal_stroke(4))
    assert sum(isinstance(item, PathPaintItem) for item in display.items) == 2
    display.append("scope-end", 5)
    display.append("group-end", 6)
    display.append("scope-begin", 7)
    display.append("group-begin", 8, group_knockout=True)
    display.append("scope-end", 9)  # Discard an unfinished local group.
    display.append_captured_drawing(internal_stroke(10))
    display.append_captured_drawing(internal_stroke(11))
    assert isinstance(display.items[-1], PathPaintItem)
    assert display.items[-1].coalesced_path


def test_stroke_coalescing_compares_AIS_and_handles_prepopulated_group_items() -> None:
    display = DisplayList(2, 2)
    display.append_captured_drawing(internal_stroke(0))
    display.append_captured_drawing(internal_stroke(1, alpha_is_shape=True))
    assert len(display.items) == 2
    prepopulated = DisplayList(2, 2, [DisplayListItem("group-begin", 0, {"group_knockout": True})])
    prepopulated.append_captured_drawing(internal_stroke(1))
    prepopulated.append_captured_drawing(internal_stroke(2))
    assert len(prepopulated.items) == 3


def internal_document() -> bytes:
    content = (
        b"q /Shape gs 1 0 0 rg 0 0 1 1 re f /Im Do "
        b"BI /W 1 /H 1 /BPC 8 /CS /RGB ID \xff\x00\x00 EI "
        b"/Sh sh BT /F 8 Tf 1 0 0 1 2 2 Tm (A) Tj ET Q 0 0 1 1 re f"
    )
    resources = (
        b"/ExtGState << /Shape << /AIS true /ca 0.4 >> >> "
        b"/Font << /F << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> "
        b"/XObject << /Im 5 0 R >> "
        b"/Shading << /Sh << /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 10 0] "
        b"/Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 1 0] /N 1 >> >> >>"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 20 20] /Resources << "
        + resources
        + b" >> /Contents 4 0 R >>",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
        b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB "
        b"/BitsPerComponent 8 /Length 3 >>\nstream\n\xff\x00\x00\nendstream",
    ]
    result = b"%PDF-1.7\n"
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(result)
    result += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    result += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    return (
        result
        + f"trailer << /Root 1 0 R /Size {len(offsets)} >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )


def test_AIS_reaches_paths_images_inline_images_shading_and_real_glyphs() -> None:
    with PdfDocument(internal_document()) as document:
        page = document.pages[0]
        program = page.get_page_program()
        assert [drawing.alpha_is_shape for drawing in program.drawings] == [True, True, True, False]
        assert len(program.inline_images) == 1
        assert program.inline_images[0].alpha_is_shape is True
        assert len(program.glyphs) == 1
        assert program.glyphs[0].alpha_is_shape is True
        display = page.render().display_list
        images = [item for item in display.items if isinstance(item, ImagePaintItem)]
        assert len(images) == 2
        assert all(item.alpha_is_shape for item in images)
        paths = [item for item in display.items if isinstance(item, PathPaintItem)]
        assert len(paths) >= 2
        assert paths[0].alpha_is_shape is True
        assert paths[-1].alpha_is_shape is False
        shading = next(item for item in display.items if item.kind == "shading")
        assert isinstance(shading, DisplayListItem)
        assert shading.data["alpha_is_shape"] is True


def test_bitmap_glyph_fallback_retains_AIS_and_existing_transparency_parameters() -> None:
    glyph = GlyphObservation(
        text="A",
        ink_bbox=(0, 0, 1, 1),
        advance_bbox=(0, 0, 1, 1),
        seqno=0,
        bitmap=(1,),
        bitmap_width=1,
        bitmap_height=1,
        alpha_is_shape=True,
        fill_opacity=0.4,
        blend_mode="Multiply",
        soft_mask_alpha=0.5,
    )
    display = DisplayList(1, 1)
    append_captured_program(display, CapturedProgram(glyphs=(glyph,)), include_text=True)
    (item,) = display.items
    assert isinstance(item, DisplayListItem)
    assert item.kind == "glyph"
    assert item.data["alpha_is_shape"] is True
    assert item.data["fill_opacity"] == 0.4
    assert item.data["soft_mask_alpha"] == 0.5
    assert item.data["blend_mode"] == "Multiply"


def test_pattern_initial_AIS_is_captured_and_included_in_cache_context() -> None:
    state = internal_state()
    stream = PdfStream(raw_data=b"0 0 1 1 re f", dictionary={})
    pattern = PdfTilingPattern(
        (0, 0, 1, 1), 1, 1, stream, {}, IDENTITY_MATRIX, 1, None, alpha_is_shape=True
    )
    first = state.capture_pattern(pattern)
    assert isinstance(first, TilingPattern)
    assert first.drawings[0].alpha_is_shape is True
    state.graphics.alpha_is_shape = True
    other = replace(pattern, alpha_is_shape=False)
    second = state.capture_pattern(other)
    assert isinstance(second, TilingPattern)
    assert second is not first
    assert second.drawings[0].alpha_is_shape is False
    assert state.capture_pattern(pattern) is first
    assert state.capture_pattern(other) is second


def test_reader_pattern_selection_keeps_initial_stream_AIS_instead_of_paint_time_flag() -> None:
    state = internal_state()
    state.graphics.alpha_is_shape = True
    pattern = PdfStream(
        raw_data=b"0 0 1 1 re f",
        dictionary={
            "PatternType": 1,
            "PaintType": 1,
            "TilingType": 1,
            "BBox": [0, 0, 1, 1],
            "XStep": 1,
            "YStep": 1,
            "Resources": {},
        },
    )
    state.stream_executor.consume(
        PdfStream(raw_data=b"/False gs /Pattern cs /P scn 0 0 1 1 re f"),
        {"Pattern": {"P": pattern}, "ExtGState": {"False": {"AIS": False}}},
        IDENTITY_MATRIX,
        0,
    )
    (paint,) = state.drawings
    assert paint.alpha_is_shape is False
    assert isinstance(paint.fill_pattern, TilingPattern)
    assert paint.fill_pattern.drawings[0].alpha_is_shape is True
    assert state.initial_alpha_is_shape is False

from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl._impl.model.runs import TextRun
from core_pdf.impl._impl.render.commands import append_captured_program
from core_pdf.impl._impl.render.display import DisplayList
from core_pdf.impl._impl.render.model import ImagePaintItem, PathPaintItem
from core_pdf.impl._impl.render.page import compose_page
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing, CapturedInlineImage
from core_pdf.impl.spec.s_07_content.page_program import CapturedProgram
from core_pdf.impl.spec.s_07_content.state import TextState
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from tests.helpers.pdf_bytes import one_page_pdf, open_pdf, stream_obj
from tests.helpers.resolvers import IdentityResolver


def test_xobject_capture_preserves_indirect_palette_and_soft_mask() -> None:
    data = one_page_pdf(
        b"2 0 0 3 4 5 cm /Im Do",
        resources=b"<< /XObject << /Im 6 0 R >> >>",
        extra_objects=[
            stream_obj(
                b"\x01",
                b"/Type /XObject /Subtype /Image /Width 1 /Height 1 "
                b"/BitsPerComponent 8 /ColorSpace 7 0 R /SMask 9 0 R",
            ),
            b"[/Indexed /DeviceRGB 1 8 0 R]",
            stream_obj(b"\xff\x00\x00\x00\xff\x00"),
            stream_obj(b"\x80", b"/Width 1 /Height 1 /BitsPerComponent 8 /ColorSpace /DeviceGray"),
        ],
    )
    with open_pdf(data) as document:
        drawing = document.pages[0].get_page_program().drawings[0]
        assert drawing.raw_data == b"\x01"
        assert drawing.soft_mask_alpha == 128 / 255
        assert drawing.image_source is not None
        raster = drawing.image_source.prepare()
        assert raster is not None
        assert raster.soft_mask is not None
        assert tuple(raster.raster.array[0, 0]) == (0, 255, 0, 128)
        assert drawing.rect is not None
        assert tuple(drawing.rect) == (4.0, 5.0, 6.0, 8.0)


def test_inline_image_has_one_capture_with_placement_and_paint_metadata() -> None:
    state = TextState(
        cast(Any, SimpleNamespace(resolver=IdentityResolver())),
        page_clip=(1.0, 2.0, 7.0, 9.0),
    )
    state.blend_mode = "Multiply"
    state.group_alpha = 0.4
    content = (
        b"0 0 1 1 re f 2 0 0 3 4 5 cm BI /W 1 /H 1 /BPC 8 /CS /RGB ID \xff\x00\x00 EI 0 0 1 1 re f"
    )
    state.consume_stream(
        PdfStream(raw_data=content), {}, IDENTITY_MATRIX, 0, clip_bbox=(1.0, 2.0, 7.0, 9.0)
    )
    program = CapturedProgram(
        drawings=tuple(state.drawings), inline_images=tuple(state.inline_images)
    )

    assert len(program.inline_images) == 1
    assert [drawing.kind for drawing in state.drawings] == ["fill", "fill"]
    image = program.inline_images[0]
    assert image.ctm == Matrix(2.0, 0.0, 0.0, 3.0, 4.0, 5.0)
    assert image.image_clip == (1.0, 2.0, 7.0, 9.0)
    assert image.stream_order == 0
    assert image.blend_mode == "Multiply"
    assert image.soft_mask_alpha == 0.4
    assert len(program.commands) == 3

    display = DisplayList(10, 10)
    append_captured_program(display, program, include_text=True)

    painted_image = next(item for item in display.items if isinstance(item, ImagePaintItem))
    assert painted_image.ctm == image.ctm
    assert painted_image.image_clip == image.image_clip
    assert painted_image.source is image.image_source
    assert painted_image.blend_mode == "Multiply"
    assert painted_image.soft_mask_alpha == 0.4


def test_appearance_inline_image_reaches_display_with_placement() -> None:
    appearance = b"BI /W 1 /H 1 /BPC 8 /CS /RGB ID \xff\x00\x00 EI 0 0 1 rg .5 0 .5 1 re f"
    data = one_page_pdf(
        b"",
        media_box=(0, 0, 10, 10),
        page_extra=b"/Annots [6 0 R]",
        extra_objects=[
            b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (image) "
            b"/Rect [4 5 6 8] /AP << /N 7 0 R >> >>",
            stream_obj(appearance, b"/Type /XObject /Subtype /Form /BBox [0 0 1 1]"),
        ],
    )
    with open_pdf(data) as document:
        page = document.pages[0]
        # The supplied program owns the already interpreted appearance.
        rendered = compose_page(page, page_program=page.get_page_program(), annotations=())
        image = next(
            item for item in rendered.display_list.items if isinstance(item, ImagePaintItem)
        )
        path = next(item for item in rendered.display_list.items if isinstance(item, PathPaintItem))
        assert image.ctm == Matrix(2.0, 0.0, 0.0, 3.0, 4.0, 5.0)
        assert image.source is not None
        assert image.source.raw == b"\xff\x00\x00"
        assert path.path is not None
        assert path.path.bbox() == (5.0, 5.0, 6.0, 8.0)


def test_inline_image_scope_does_not_clip_following_text() -> None:
    text = b"BT /F1 12 Tf 30 40 Td (outside image clip) Tj ET"
    image = b"q 0 0 10 10 re W n BI /W 1 /H 1 /BPC 8 /CS /RGB ID \xff\x00\x00 EI Q "
    with open_pdf(one_page_pdf(text, media_box=(0, 0, 150, 80))) as document:
        expected = document.pages[0].render().rasterize().array()
    with open_pdf(one_page_pdf(image + text, media_box=(0, 0, 150, 80))) as document:
        actual = document.pages[0].render().rasterize().array()

    # The image's clip scope must end before the following text is replayed.
    # Its Q marker therefore needs its own sequence number, just like BI.
    assert (expected[:, 20:] < 255).any()
    assert (actual[:, 20:] == expected[:, 20:]).all()


@pytest.mark.parametrize("inline", [False, True])
def test_images_shading_and_scope_markers_preserve_stream_order(inline: bool) -> None:
    image = b"BI /W 1 /H 1 /BPC 8 /CS /RGB ID \xff\x00\x00 EI" if inline else b"/Im Do"
    data = one_page_pdf(
        b"q 0 0 10 10 re W n " + image + b" Q /Sh sh BT /F1 12 Tf 30 40 Td (X) Tj ET",
        media_box=(0, 0, 150, 80),
        resources=b"<< /Font << /F1 5 0 R >> /XObject << /Im 6 0 R >> "
        b"/Shading << /Sh << /ShadingType 2 /ColorSpace /DeviceRGB "
        b"/Coords [0 0 10 0] /Function << /FunctionType 2 /Domain [0 1] "
        b"/C0 [0 0 0] /C1 [1 1 1] /N 1 >> >> >> >>",
        extra_objects=[
            stream_obj(
                bytes((255, 0, 0)),
                b"/Type /XObject /Subtype /Image /Width 1 /Height 1 "
                b"/ColorSpace /DeviceRGB /BitsPerComponent 8",
            ),
        ],
    )
    with open_pdf(data) as document:
        commands = document.pages[0].get_page_program().commands
    # Glyph paint and its diagnostic run deliberately share sequence numbers;
    # every other captured event must retain its original position around them.
    events = [
        command
        for command in commands
        if isinstance(command, (TextRun, CapturedDrawing, CapturedInlineImage))
    ]
    kinds = [
        "text"
        if isinstance(command, TextRun)
        else "inline-image"
        if isinstance(command, CapturedInlineImage)
        else command.kind
        for command in events
    ]
    assert kinds == [
        "state-push",
        "clip",
        "inline-image" if inline else "image",
        "state-pop",
        "shading",
        "text",
    ]
    sequence = [command.seqno for command in events]
    assert sequence == sorted(set(sequence))

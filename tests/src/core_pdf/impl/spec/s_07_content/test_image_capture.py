from types import SimpleNamespace
from typing import Any, cast

from core_pdf.impl.render.display import DisplayList
from core_pdf.impl.render.model import ImagePaintItem, PathPaintItem
from core_pdf.impl.render.page import compose_page, internal_append_page_program
from core_pdf.impl.spec.s_07_content.page_program import PageProgram
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
    program = PageProgram(drawings=tuple(state.drawings), inline_images=tuple(state.inline_images))

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
    internal_append_page_program(display, program, include_text=True)

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
        # Supply an empty page program to isolate the separately rendered form
        # appearance, which formerly read the duplicate drawing projection.
        rendered = compose_page(page, page_program=PageProgram(), annotations=())
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

    # Advancing BI alone moves preceding clip markers ahead of text, but Q
    # still ties with that text and sorts after it. Scope boundaries and paint
    # commands must be sequenced together before changing this legacy ordering.
    assert (expected[:, 20:] < 255).any()
    assert (actual[:, 20:] == expected[:, 20:]).all()

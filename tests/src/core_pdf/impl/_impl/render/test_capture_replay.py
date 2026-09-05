from types import SimpleNamespace
from typing import Any, cast

import numpy
import pytest

from core_pdf.impl._impl.render.clipping import internal_ClipState
from core_pdf.impl._impl.render.model import DisplayListItem, RenderOptions
from core_pdf.impl._impl.render.page import compose_page
from core_pdf.impl._impl.render.target import internal_RasterTarget
from core_pdf.impl.spec.s_07_content.page_program import CapturedProgram, PageProgram
from tests.helpers.pdf_bytes import one_page_pdf, open_pdf, stream_obj


@pytest.mark.parametrize(
    "matrix",
    [(4, 0, 0, 2, 2, 2), (-4, 0, 0, 2, 6, 2), (0, 4, -2, 0, 6, 2)],
)
def test_inline_image_uses_the_same_affine_placement_as_xobjects(matrix: tuple[int, ...]) -> None:
    content = " ".join(map(str, matrix)).encode() + (
        b" cm BI /W 2 /H 1 /BPC 8 /CS /RGB ID \xff\x00\x00\x00\xff\x00 EI"
    )
    with open_pdf(one_page_pdf(content, media_box=(0, 0, 8, 8))) as document:
        rendered = document.pages[0].render()
        pixels = rendered.rasterize().array()
        a, b, c, d, e, f = matrix
        for u, color in ((0.25, (255, 0, 0, 255)), (0.75, (0, 255, 0, 255))):
            x, y = u * a + 0.5 * c + e, u * b + 0.5 * d + f
            assert tuple(pixels[7 - int(y), int(x)]) == color
        cropped = rendered.rasterize(crop=(2, 2, 6, 6)).array()
        numpy.testing.assert_array_equal(cropped, pixels[2:6, 2:6])


def test_inline_stencil_keeps_its_current_fill_color() -> None:
    content = b"0 0 1 rg 2 0 0 2 1 1 cm BI /W 1 /H 1 /IM true ID \x00 EI"
    with open_pdf(one_page_pdf(content, media_box=(0, 0, 4, 4))) as document:
        pixels = document.pages[0].render().rasterize().array()
    assert numpy.all(pixels[1:3, 1:3] == (0, 0, 255, 255))


def annotation_pdf(*, widget: bool = False, flags: int = 0, duplicate: bool = False) -> bytes:
    subtype = b"/Subtype /Widget /FT /Tx /T (field)" if widget else b"/Subtype /Square"
    return one_page_pdf(
        b"",
        media_box=(0, 0, 8, 8),
        page_extra=b"/Annots [6 0 R 6 0 R]" if duplicate else b"/Annots [6 0 R]",
        extra_objects=[
            b"<< /Type /Annot "
            + subtype
            + b" /Rect [2 2 6 6] /F "
            + str(flags).encode()
            + b" /AP << /N 7 0 R >> >>",
            stream_obj(
                b"/GS gs 1 0 0 rg -1 -1 3 4 re f",
                b"/Type /XObject /Subtype /Form /BBox [0 0 1 2] /Matrix [0 1 -1 0 1 0] "
                b"/Resources << /ExtGState << /GS << /ca 0.5 >> >> >>",
            ),
        ],
    )


@pytest.mark.parametrize("widget", [False, True])
@pytest.mark.parametrize("include_annotations", [False, True])
@pytest.mark.parametrize("include_layers", [False, True])
def test_appearance_scope_is_painted_once_with_its_own_option(
    widget: bool, include_annotations: bool, include_layers: bool
) -> None:
    with open_pdf(annotation_pdf(widget=widget, duplicate=True)) as document:
        page = document.pages[0]
        program = page.get_page_program()
        assert len(program.appearances) == 1
        rendered = compose_page(
            page,
            RenderOptions(include_annotations=include_annotations, include_layers=include_layers),
            page_program=program,
        )
        pixels = rendered.rasterize().array()
    expected = numpy.zeros((8, 8), dtype=numpy.uint8)
    if include_layers if widget else include_annotations:
        expected[2:6, 2:6] = 128
    numpy.testing.assert_array_equal(pixels[:, :, 3], expected)


@pytest.mark.parametrize("flags", [2, 32])
def test_hidden_appearance_cannot_be_resurrected_by_rendering(flags: int) -> None:
    with open_pdf(annotation_pdf(flags=flags)) as document:
        page = document.pages[0]
        assert page.get_page_program().appearances == ()
        assert not page.render().rasterize().array()[:, :, 3].any()


def test_supplied_program_never_reinterprets_appearances(monkeypatch: pytest.MonkeyPatch) -> None:
    with open_pdf(annotation_pdf()) as document:
        page = document.pages[0]

        def unexpected_capture(**kwargs: Any) -> None:
            raise AssertionError("a supplied page program must be authoritative")

        monkeypatch.setattr(page, "get_page_program", unexpected_capture)
        rendered = compose_page(page, page_program=PageProgram())
        assert not rendered.rasterize().array()[:, :, 3].any()
        assert all(
            not item.data.get("appearance_rendered")
            for item in rendered.display_list.items
            if isinstance(item, DisplayListItem)
        )


def test_composition_forwards_supplied_record_iterators_once() -> None:
    field = SimpleNamespace(widget={}, dict={}, rect=None)
    annotation = SimpleNamespace(dict={}, rect=None, subtype="Square", contents="")
    calls = []

    def capture(**kwargs: Any) -> PageProgram:
        calls.append(kwargs)
        return PageProgram()

    page = SimpleNamespace(width=8, height=8, media_box=(0, 0, 8, 8), get_page_program=capture)
    compose_page(cast(Any, page), fields=iter((field,)), annotations=iter((annotation,)))
    assert calls == [{"fields": (field,), "annotations": (annotation,)}]


@pytest.mark.parametrize("inline", [False, True])
def test_pattern_cell_replays_typed_paths_and_inline_images(inline: bool) -> None:
    cell = (
        b"BI /W 1 /H 1 /BPC 8 /CS /RGB ID \xff\x00\x00 EI" if inline else b"1 0 0 rg 0 0 1 1 re f"
    )
    data = one_page_pdf(
        b"/Pattern cs /P scn 0 0 4 4 re f",
        media_box=(0, 0, 4, 4),
        resources=b"<< /Pattern << /P 6 0 R >> >>",
        extra_objects=[
            stream_obj(
                cell,
                b"/Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 "
                b"/BBox [0 0 2 2] /XStep 2 /YStep 2 /Resources << >>",
            )
        ],
    )
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize().array()
    assert numpy.all(pixels[1::2, ::2] == (255, 0, 0, 255))
    assert numpy.count_nonzero(pixels[:, :, 3]) == 4


def test_repeated_cell_cannot_pop_the_enclosing_transparency_group() -> None:
    pixels = bytearray(4 * 4 * 4)
    clip = internal_ClipState(crop_x0=0, crop_y1=4, scale=1, width=4, height=4)
    target = internal_RasterTarget(
        pixels,
        None,
        clip=clip,
        width=4,
        height=4,
        scale=1,
        crop_x0=0,
        crop_y0=0,
        crop_y1=4,
        page_view=numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(4, 4, 4),
    )
    parent = bytearray(4 * 4 * 4)
    target.push_group(parent, 0.5, None)
    target.paint_items([DisplayListItem("group-end", 0)], translation=(0, 0))
    assert len(target.buffer_stack) == 2
    assert target.pixels is parent


def test_body_only_program_aliases_its_flattened_products() -> None:
    body = CapturedProgram()
    program = PageProgram(body=body)
    assert program.drawings is body.drawings
    assert program.commands is body.commands


def test_uncolored_pattern_uses_the_base_color_without_erasing_colored_patterns() -> None:
    data = one_page_pdf(
        b"/Cs cs 0 0 1 /P scn 0 0 4 4 re f",
        media_box=(0, 0, 4, 4),
        resources=b"<< /ColorSpace << /Cs [/Pattern /DeviceRGB] >> /Pattern << /P 6 0 R >> >>",
        extra_objects=[
            stream_obj(
                b"0 0 1 1 re f",
                b"/Type /Pattern /PatternType 1 /PaintType 2 /TilingType 1 "
                b"/BBox [0 0 2 2] /XStep 2 /YStep 2 /Resources << >>",
            )
        ],
    )
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize().array()
    assert numpy.all(pixels[1::2, ::2] == (0, 0, 255, 255))
    assert numpy.count_nonzero(pixels[:, :, 3]) == 4


def test_inline_image_resolves_page_color_space_resources() -> None:
    data = one_page_pdf(
        b"2 0 0 2 1 1 cm BI /W 1 /H 1 /BPC 8 /CS /Colors ID \xff\x00\x00 EI",
        media_box=(0, 0, 4, 4),
        resources=b"<< /ColorSpace << /Colors /DeviceRGB >> >>",
    )
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize().array()
    assert numpy.all(pixels[1:3, 1:3] == (255, 0, 0, 255))


def test_malformed_appearance_retains_paint_captured_before_the_parse_error() -> None:
    data = one_page_pdf(
        b"",
        media_box=(0, 0, 4, 4),
        page_extra=b"/Annots [6 0 R]",
        extra_objects=[
            b"<< /Type /Annot /Subtype /Square /Rect [1 1 3 3] /AP << /N 7 0 R >> >>",
            stream_obj(
                b"1 0 0 rg 0 0 1 1 re f (",
                b"/Type /XObject /Subtype /Form /BBox [0 0 1 1]",
            ),
        ],
    )
    with open_pdf(data) as document:
        page = document.pages[0]
        assert len(page.get_page_program().appearances) == 1
        pixels = page.render().rasterize().array()
    assert numpy.all(pixels[1:3, 1:3] == (255, 0, 0, 255))


def test_recoverable_body_text_is_flushed_before_appearance_capture() -> None:
    data = one_page_pdf(
        b"BT /F1 12 Tf 1 20 Td (BODY) Tj (",
        media_box=(0, 0, 50, 40),
        page_extra=b"/Annots [6 0 R]",
        extra_objects=[
            b"<< /Type /Annot /Subtype /Square /Rect [1 1 20 10] /AP << /N 7 0 R >> >>",
            stream_obj(
                b"BT /F1 6 Tf 0 0 Td (AP) Tj ET",
                b"/Type /XObject /Subtype /Form /BBox [0 0 19 9]",
            ),
        ],
    )
    with open_pdf(data) as document:
        page = document.pages[0]
        # Contents arrays permit recovery past an individual damaged stream.
        page.contents = cast(Any, [page.contents])
        program = page.get_page_program()
    assert "".join(run.text for run in program.body.runs) == "BODY"
    assert "".join(run.text for run in program.appearances[0].program.runs) == "AP"
    assert "".join(run.text for run in program.runs) == "BODYAP"


def test_supplemental_field_parent_is_not_an_annotation_appearance() -> None:
    with open_pdf(annotation_pdf()) as document:
        page = document.pages[0]
        parent = dict(page.get_annotations()[0].dict)
        del parent["Subtype"]
        field = SimpleNamespace(widget=None, dict=parent)
        program = page.get_page_program(fields=(cast(Any, field),), annotations=())
    assert program.appearances == ()


def test_unbalanced_appearance_scope_cannot_clip_the_next_appearance() -> None:
    data = one_page_pdf(
        b"",
        media_box=(0, 0, 8, 8),
        page_extra=b"/Annots [6 0 R 8 0 R]",
        extra_objects=[
            b"<< /Type /Annot /Subtype /Square /Rect [1 1 3 3] /AP << /N 7 0 R >> >>",
            stream_obj(
                b"q 0 0 .5 .5 re W n 1 0 0 rg 0 0 1 1 re f",
                b"/Type /XObject /Subtype /Form /BBox [0 0 1 1]",
            ),
            b"<< /Type /Annot /Subtype /Square /Rect [5 5 7 7] /AP << /N 9 0 R >> >>",
            stream_obj(
                b"0 0 1 rg 0 0 1 1 re f",
                b"/Type /XObject /Subtype /Form /BBox [0 0 1 1]",
            ),
        ],
    )
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize().array()
    assert numpy.all(pixels[1:3, 5:7] == (0, 0, 255, 255))


@pytest.mark.parametrize("prefix", [b"", b"q "])
def test_page_body_clip_does_not_belong_to_appearances(prefix: bytes) -> None:
    data = one_page_pdf(
        prefix + b"0 0 1 1 re W n 1 0 0 rg 0 0 8 8 re f",
        media_box=(0, 0, 8, 8),
        page_extra=b"/Annots [6 0 R]",
        extra_objects=[
            b"<< /Type /Annot /Subtype /Square /Rect [5 5 7 7] /AP << /N 7 0 R >> >>",
            stream_obj(
                b"0 0 1 rg 0 0 1 1 re f",
                b"/Type /XObject /Subtype /Form /BBox [0 0 1 1]",
            ),
        ],
    )
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize().array()
    assert numpy.all(pixels[1:3, 5:7] == (0, 0, 255, 255))
    assert numpy.count_nonzero(numpy.all(pixels == (255, 0, 0, 255), axis=2)) == 1


def test_adjacent_appearance_text_cannot_sort_before_the_previous_restore() -> None:
    data = one_page_pdf(
        b"",
        media_box=(0, 0, 40, 10),
        page_extra=b"/Annots [6 0 R 8 0 R]",
        extra_objects=[
            b"<< /Type /Annot /Subtype /Square /Rect [0 0 10 10] /AP << /N 7 0 R >> >>",
            stream_obj(
                b"q 0 0 10 10 re W n BT /F1 8 Tf 1 2 Td (A) Tj ET Q",
                b"/Type /XObject /Subtype /Form /BBox [0 0 10 10]",
            ),
            b"<< /Type /Annot /Subtype /Square /Rect [20 0 30 10] /AP << /N 9 0 R >> >>",
            stream_obj(
                b"q 0 0 10 10 re W n BT /F1 8 Tf 1 2 Td (A) Tj ET Q",
                b"/Type /XObject /Subtype /Form /BBox [0 0 10 10]",
            ),
        ],
    )
    with open_pdf(data) as document:
        page = document.pages[0]
        program = page.get_page_program()
        previous_restore = program.appearances[0].program.drawings[-1]
        next_glyph = program.appearances[1].program.glyphs[0]
        assert previous_restore.kind == "state-pop"
        assert previous_restore.seqno < next_glyph.seqno
        pixels = compose_page(page, page_program=program).rasterize().array()
    assert numpy.count_nonzero(pixels[:, :10, 3]) > 0
    numpy.testing.assert_array_equal(pixels[:, :10], pixels[:, 20:30])

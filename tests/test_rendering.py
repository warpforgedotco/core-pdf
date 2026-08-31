from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from threading import Event

import imagecodecs
import numpy
import pytest

from core_pdf.impl.exceptions import PdfRasterTooLargeError
from core_pdf.impl.model.glyph_table import GlyphTable
from core_pdf.impl.model.glyphs import GlyphObservation
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.page import text_rotation_correction_for_runs
from core_pdf.impl.render.display import (
    CompiledRenderPlan,
    DisplayList,
    ImagePaintItem,
    PathPaintItem,
    RenderOptions,
)
from core_pdf.impl.render.kernels import (
    rasterize_packed_stroked_paths,
    rasterize_unclipped_line_normal,
)
from core_pdf.impl.render.page import (
    RenderedPage,
    compose_page,
    internal_append_glyph_paint,
)
from core_pdf.impl.runtime.image_cache import ImageCache
from core_pdf.impl.spec.s_07_content.capture import (
    CapturedDrawing,
    CapturedPath,
    CapturedSubpath,
)
from core_pdf.impl.spec.s_07_content.page_program import (
    LineTable,
    PageProducts,
    PageProgram,
)
from core_pdf.impl.spec.s_07_document.document import PdfDocument
from core_pdf.impl.spec.s_07_document.page import PdfPage
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_08_graphics.color import (
    ImageColorManager,
    internal_sampled_separation_rgb_lut,
)
from core_pdf.impl.spec.s_08_graphics.color_spec import ImageColorSpec
from core_pdf.impl.spec.s_08_graphics.image_decode import (
    ImageSource,
    decode_pdf_image,
)


class StaticDisplayPage(PdfPage):
    def __init__(
        self,
        chars: list[TextRun],
        *,
        rotation: int = 0,
        width: float = 0,
        height: float = 0,
    ) -> None:
        self.static_chars = chars
        self.static_rotation = rotation
        self.static_width = width
        self.static_height = height

    @property
    def chars(self) -> list[TextRun]:
        return self.static_chars

    @property
    def rotation(self) -> int:
        return self.static_rotation

    @property
    def width(self) -> float:
        return self.static_width

    @property
    def height(self) -> float:
        return self.static_height


def rendered_page(*, width: float = 5, height: float = 7, rotate: int = 0) -> RenderedPage:
    return RenderedPage(
        page_number=1,
        width=width,
        height=height,
        rotate=rotate,
        display_list=DisplayList(width=width, height=height),
    )


@pytest.mark.parametrize(("value", "normalized"), [(-90, 270), (360, 0), (450, 90)])
def test_render_options_normalizes_orthogonal_rotation(value: int, normalized: int) -> None:
    assert RenderOptions(rotate=value).rotate == normalized


def test_render_options_rejects_non_orthogonal_rotation() -> None:
    with pytest.raises(ValueError, match="multiple of 90"):
        RenderOptions(rotate=45)


def test_raster_size_accounts_for_rotation_crop_and_scale() -> None:
    page = rendered_page(width=100, height=200, rotate=90)
    page.metadata["crop"] = (10, 20, 70, 100)

    assert page.unrotated_raster_size(scale=2) == (120, 160)
    assert page.raster_size(scale=2) == (160, 120)
    assert page.rasterize(scale=2).nbytes == 160 * 120 * 4


def test_rasterize_accepts_a_crop_without_recomposing_the_display_list() -> None:
    page = rendered_page(width=100, height=80)

    full = page.rasterize(background=(255, 255, 255, 255), cache=False)
    cropped = page.rasterize(
        background=(255, 255, 255, 255),
        crop=(10.0, 20.0, 40.0, 60.0),
        cache=False,
    )

    assert full.nbytes == 100 * 80 * 4
    assert cropped.width == 30
    assert cropped.height == 40
    assert cropped.nbytes == 30 * 40 * 4
    assert page.raster_cache == {}


def test_compiled_render_plan_culls_distant_paint_but_preserves_state() -> None:
    display_list = DisplayList(width=2_000, height=200)
    display_list.append("state-push", 0)
    for index in range(10):
        path = CapturedPath()
        path.rect(index * 180.0, 10.0, 40.0, 40.0)
        display_list.append(
            "fill",
            index + 1,
            bbox=(index * 180.0, 10.0, index * 180.0 + 40.0, 50.0),
            path=path,
            fill=(0.0,),
        )
    display_list.append("state-pop", 11)

    plan = CompiledRenderPlan.compile(display_list)
    selected = plan.items_for_crop((0.0, 0.0, 100.0, 100.0))

    assert len(selected) == 3
    assert [item.kind for item in selected] == ["state-push", "fill", "state-pop"]


def test_unpatterned_path_uses_typed_display_record() -> None:
    page = rendered_page(width=4, height=4)
    path = CapturedPath()
    path.rect(1.0, 1.0, 2.0, 2.0)

    page.display_list.append(
        "fill",
        1,
        bbox=(1.0, 1.0, 3.0, 3.0),
        path=path,
        fill=(0.0, 0.0, 0.0),
        fill_opacity=1.0,
        fill_rule="nonzero",
    )

    assert type(page.display_list.items[0]) is PathPaintItem
    raster = page.rasterize(background=(255, 255, 255, 255))
    pixels = numpy.frombuffer(raster.pixels, dtype=numpy.uint8).reshape(4, 4, 4)
    assert numpy.all(pixels[1:3, 1:3, :3] == 0)


def test_packed_stroked_rasterizer_draws_antialiased_skeleton() -> None:
    display_list = DisplayList(width=10.0, height=6.0)
    path = CapturedPath()
    path.move_to(1.0, 1.0)
    path.line_to(9.0, 5.0)
    display_list.append(
        "stroke",
        1,
        path=path,
        stroke_color=(0.0, 0.0, 0.0),
        stroke_opacity=1.0,
        line_width=0.48,
        line_cap=1,
        line_join=1,
        dash_pattern=([], 0.0),
    )

    raster = rasterize_packed_stroked_paths(tuple(display_list.items), 10.0, 6.0, 6.0)
    pixels = raster.array()

    assert (raster.width, raster.height) == (60, 36)
    assert numpy.count_nonzero(pixels[:, :, 0] < 255) > 0
    assert numpy.count_nonzero(pixels[:, :, 0] < 128) > 0
    assert numpy.all(pixels[:, :, 3] == 255)


def test_consecutive_captured_strokes_coalesce_without_changing_pixels() -> None:
    first_path = CapturedPath()
    first_path.move_to(1.0, 1.0)
    first_path.line_to(4.0, 4.0)
    second_path = CapturedPath()
    second_path.move_to(4.0, 4.0)
    second_path.line_to(7.0, 2.0)
    drawings = tuple(
        CapturedDrawing(
            seqno=index,
            fill=None,
            fill_opacity=1.0,
            stroke_color=(0.1, 0.3, 0.8),
            stroke_opacity=0.75,
            line_width=0.8,
            line_cap=1,
            line_join=1,
            kind="stroke",
            path=path,
        )
        for index, path in enumerate((first_path, second_path), start=1)
    )
    coalesced = rendered_page(width=8, height=6)
    for drawing in drawings:
        coalesced.display_list.append_captured_drawing(drawing)
    separate = rendered_page(width=8, height=6)
    for drawing in drawings:
        separate.display_list.append(
            "stroke",
            drawing.seqno,
            bbox=drawing.rect,
            path=drawing.path,
            stroke_color=drawing.stroke_color,
            stroke_opacity=drawing.stroke_opacity,
            line_width=drawing.line_width,
            line_cap=drawing.line_cap,
            line_join=drawing.line_join,
        )

    assert len(coalesced.display_list.items) == 1
    item = coalesced.display_list.items[0]
    assert type(item) is PathPaintItem
    assert item.coalesced_path
    assert len(item.path.subpaths) == 2
    assert item.bbox == (1.0, 1.0, 7.0, 4.0)
    assert (
        coalesced.rasterize(scale=3, cache=False).pixels
        == separate.rasterize(
            scale=3,
            cache=False,
        ).pixels
    )
    assert (
        coalesced.rasterize(scale=3, crop=(0.0, 0.0, 4.0, 3.0), cache=False).pixels
        == separate.rasterize(
            scale=3,
            crop=(0.0, 0.0, 4.0, 3.0),
            cache=False,
        ).pixels
    )


def test_captured_stroke_coalescing_respects_control_items_and_batch_limit() -> None:
    display_list = DisplayList(width=300, height=2)

    def append_stroke(index: int) -> None:
        path = CapturedPath()
        path.move_to(float(index), 0.0)
        path.line_to(float(index), 1.0)
        display_list.append_captured_drawing(
            CapturedDrawing(
                seqno=index,
                fill=None,
                fill_opacity=1.0,
                stroke_color=(0.0,),
                stroke_opacity=1.0,
                line_width=0.5,
                kind="stroke",
                path=path,
            )
        )

    append_stroke(0)
    display_list.append("state-push", 1)
    for index in range(1, 258):
        append_stroke(index)

    assert [item.kind for item in display_list.items] == [
        "stroke",
        "state-push",
        "stroke",
        "stroke",
    ]
    first_batch = display_list.items[2]
    second_batch = display_list.items[3]
    assert type(first_batch) is PathPaintItem
    assert type(second_batch) is PathPaintItem
    assert len(first_batch.path.subpaths) == 256
    assert len(second_batch.path.subpaths) == 1


def test_clip_preserves_zero_area_stroke_bounds() -> None:
    page = rendered_page(width=4, height=4)
    clip_path = CapturedPath()
    clip_path.rect(1.0, 1.0, 2.0, 2.0)
    stroke_path = CapturedPath()
    stroke_path.move_to(2.0, 1.5)
    stroke_path.line_to(2.0, 2.5)
    page.display_list.append("state-push", 0)
    page.display_list.append("clip", 1, path=clip_path)
    page.display_list.append(
        "stroke",
        2,
        bbox=stroke_path.bbox(),
        path=stroke_path,
        stroke_color=(0.0,),
        stroke_opacity=1.0,
        line_width=0.5,
    )

    pixels = page.rasterize(scale=4, background=(255, 255, 255, 255), cache=False).array()

    assert numpy.any(numpy.all(pixels[:, :, :3] == 0, axis=2))


def test_vectorized_line_kernel_matches_scalar_antialias_samples() -> None:
    width = height = 5
    pixels = bytearray(width * height * 4)
    rgba = (19, 73, 211, 255)
    x0, y0, x1, y1 = 0.25, 0.5, 4.75, 4.25
    line_width = 0.8

    rasterize_unclipped_line_normal(
        pixels,
        width,
        0.0,
        float(height),
        1.0,
        x0,
        y0,
        x1,
        y1,
        line_width,
        rgba,
        0,
        (0, 0, width, height),
    )

    expected = bytearray(width * height * 4)
    x_delta = x1 - x0
    y_delta = y1 - y0
    segment_length_squared = x_delta * x_delta + y_delta * y_delta
    half_width = max(0.5, line_width * 0.5)
    cross_limit_squared = half_width * half_width * segment_length_squared
    for py in range(height):
        for px in range(width):
            covered = 0
            for sy in range(4):
                page_y = height - (py + (sy + 0.5) / 4.0)
                for sx in range(4):
                    page_x = px + (sx + 0.5) / 4.0
                    offset_x = page_x - x0
                    offset_y = page_y - y0
                    projection = (offset_x * x_delta + offset_y * y_delta) / (
                        segment_length_squared
                    )
                    cross = offset_x * y_delta - offset_y * x_delta
                    covered += 0.0 <= projection <= 1.0 and cross * cross <= cross_limit_squared
            if covered:
                index = (py * width + px) * 4
                expected[index : index + 4] = bytes((*rgba[:3], round(255 * covered / 16)))

    assert pixels == expected


def test_cmyk_conversion_handles_process_inks_and_black() -> None:
    # Solid inks come out as the colours a SWOP press actually prints, not as
    # the saturated (0, 255, 255) / (255, 0, 255) / (255, 255, 0) the naive
    # 255*(1-ink)*(1-black) formula produces. 100% K is the profile's black
    # after black point compensation, which is a near-neutral very dark grey.
    samples = bytes.fromhex("00000000 ff000000 00ff0000 0000ff00 0a141e28 000000ff")
    expected = bytes.fromhex("ffffff 00aef0 ec0a8d fff300 d6cdc6 292728")

    numpy.testing.assert_array_equal(
        ImageColorManager.convert_cmyk(samples),
        numpy.frombuffer(expected, dtype=numpy.uint8),
    )


def test_cmyk_conversion_is_monotonic_in_ink_and_black() -> None:
    """Properties any usable CMYK profile has, independent of which one ships."""
    no_ink = ImageColorManager.convert_cmyk(bytes(4))
    assert tuple(no_ink) == (255, 255, 255)

    for channel in range(4):
        ramp = bytearray()
        for level in (0, 64, 128, 192, 255):
            ink = [0, 0, 0, 0]
            ink[channel] = level
            ramp.extend(ink)
        luminance = ImageColorManager.convert_cmyk(bytes(ramp)).reshape(-1, 3).sum(axis=1)
        assert list(luminance) == sorted(luminance, reverse=True), (
            f"channel {channel} does not darken monotonically: {luminance}"
        )

    black = ImageColorManager.convert_cmyk(bytes.fromhex("000000ff")).reshape(3)
    assert black.max() < 64
    assert int(black.max()) - int(black.min()) <= 8


def test_sampled_separation_conversion_reuses_exact_rgb_lut() -> None:
    tint_function = PdfStream(
        {
            "FunctionType": 0,
            "BitsPerSample": 8,
            "Size": [256],
            "Domain": [0, 1],
            "Range": [0, 1],
        },
        bytes(range(256)),
    )
    spec = ImageColorSpec(
        kind="Separation",
        params={},
        bits_per_component=8,
        alt="DeviceGray",
        tint_fn=tint_function,
    )
    samples = bytes((0, 64, 128, 192, 255, 64, 0))

    internal_sampled_separation_rgb_lut.cache_clear()
    first = ImageColorManager.convert_separation(samples, spec)
    second = ImageColorManager.convert_separation(samples, spec)
    expected = numpy.repeat(numpy.frombuffer(samples, dtype=numpy.uint8), 3)

    numpy.testing.assert_array_equal(first, expected)
    numpy.testing.assert_array_equal(second, expected)
    assert internal_sampled_separation_rgb_lut.cache_info().misses == 1
    assert internal_sampled_separation_rgb_lut.cache_info().hits == 1
    assert not internal_sampled_separation_rgb_lut(tint_function, "DeviceGray").flags.writeable


def test_indexed_conversion_uses_rgb_lookup_entries() -> None:
    lookup = bytes(value for index in range(4) for value in (index, index + 10, index + 20))
    spec = ImageColorSpec("Indexed", {}, base="DeviceRGB", hival=3, lookup=lookup)

    numpy.testing.assert_array_equal(
        ImageColorManager.convert_indexed(bytes((3, 0, 2)), spec),
        numpy.asarray((3, 13, 23, 0, 10, 20, 2, 12, 22), dtype=numpy.uint8),
    )


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rasterize_rotation_preserves_rgba_pixel_order(rotation: int) -> None:
    page = rendered_page(width=2, height=3, rotate=rotation)
    page.display_list.append(
        "glyph",
        1,
        bitmap=(1, 2, 3, 4, 5, 6),
        bitmap_width=2,
        bitmap_height=3,
        bbox=(0, 0, 2, 3),
        fill_color=(0, 0, 0),
    )
    unrotated = rendered_page(width=2, height=3)
    unrotated.display_list.items = page.display_list.items

    source = unrotated.rasterize(background=(255, 255, 255, 255)).pixels
    actual = page.rasterize(background=(255, 255, 255, 255)).pixels
    width = 2
    height = 3
    expected_pixels = [bytes(source[index : index + 4]) for index in range(0, len(source), 4)]
    rotated_pixels = [b"\xff\xff\xff\xff"] * (width * height)
    for y in range(height):
        for x in range(width):
            if rotation == 90:
                dst_x, dst_y = height - 1 - y, x
                dst_width = height
            elif rotation == 180:
                dst_x, dst_y = width - 1 - x, height - 1 - y
                dst_width = width
            else:
                dst_x, dst_y = y, width - 1 - x
                dst_width = height
            rotated_pixels[dst_y * dst_width + dst_x] = expected_pixels[y * width + x]

    assert actual == b"".join(rotated_pixels)


def test_rasterize_rejects_oversized_canvas_before_allocation() -> None:
    page = rendered_page(width=100, height=200)

    with pytest.raises(PdfRasterTooLargeError, match="pixels=20000, maximum=19999"):
        page.rasterize(max_pixels=19_999)

    assert page.raster_cache == {}


def test_seekable_source_is_read_from_start_and_restored() -> None:
    source = BytesIO(b"complete PDF source")
    source.seek(9)
    loader = object.__new__(PdfDocument)

    data = loader.load_data(source)

    assert data == b"complete PDF source"
    assert source.tell() == 9


def test_display_chars_apply_page_rotation_to_text_geometry() -> None:
    run = TextRun(
        "text",
        10,
        20,
        30,
        40,
        10,
        20,
        12,
        4,
        0,
        0,
        0,
    )
    page = StaticDisplayPage([run], rotation=90, width=100, height=200)

    displayed = page.display_chars

    assert len(displayed) == 1
    assert (displayed[0].x0, displayed[0].y0, displayed[0].x1, displayed[0].y1) == (
        20,
        70,
        40,
        90,
    )
    assert displayed[0].rotation_angle == 270


def test_text_rotation_correction_uses_displayed_text_orientation() -> None:
    run = TextRun(
        "dominant text",
        0,
        0,
        10,
        10,
        0,
        0,
        10,
        3,
        0,
        0,
        0,
        rotation_angle=270,
    )
    correction = text_rotation_correction_for_runs([run], threshold=0.95)

    assert correction == 90


def test_text_display_items_do_not_fabricate_raster_pixels() -> None:
    page = rendered_page()
    page.display_list.append(
        "text",
        1,
        text="L",
        bbox=(0, 0, 5, 7),
        fill_color=(0, 0, 0),
    )

    raster = page.rasterize(background=(255, 255, 255, 255))

    assert raster.pixels == bytes((255, 255, 255, 255)) * 5 * 7


def test_text_free_composition_skips_glyph_paint_and_lazy_bitmap_resolution() -> None:
    class Decoder:
        def __init__(self) -> None:
            self.calls = 0

        def glyph_bitmap(self, code: int, *, width: int, height: int) -> tuple[int, ...]:
            self.calls += 1
            assert (code, width, height) == (65, 1, 1)
            return (255,)

    class Page:
        media_box = (0.0, 0.0, 10.0, 20.0)
        width = 10.0
        height = 20.0
        page_number = 1
        rotation = 0

        def get_fields(self) -> list[object]:
            return []

        def get_annotations(self) -> list[object]:
            return []

        def resolve_transparency_group_alpha(self) -> None:
            return None

    decoder = Decoder()
    glyph = GlyphObservation(
        "A",
        (1.0, 1.0, 2.0, 2.0),
        (1.0, 1.0, 2.0, 2.0),
        1,
        bitmap_width=1,
        bitmap_height=1,
        bitmap_code=65,
        font_decoder=decoder,
    )
    products = PageProducts((), GlyphTable.from_rows((glyph,)), (), (), LineTable.from_lines(()))
    page_program = PageProgram(products)

    text_free = compose_page(
        Page(),
        RenderOptions(include_text=False),
        page_program=page_program,
    )
    with_text = compose_page(Page(), page_program=page_program)

    assert text_free.display_list.items == []
    assert [item.kind for item in with_text.display_list.items] == ["glyph"]
    assert decoder.calls == 1

    assert text_free.cache_identity != with_text.cache_identity


@pytest.mark.parametrize(
    ("render_mode", "paint_kind", "clips"),
    [
        (0, "fill", False),
        (1, "stroke", False),
        (2, "fillstroke", False),
        (3, None, False),
        (4, "fill", True),
        (5, "stroke", True),
        (6, "fillstroke", True),
        (7, None, True),
    ],
)
def test_vector_glyph_honors_text_rendering_modes(
    render_mode: int, paint_kind: str | None, clips: bool
) -> None:
    class Decoder:
        def glyph_outline(
            self, code: int, gid: int | None, text: str
        ) -> tuple[tuple[tuple[float, float], ...], ...]:
            assert (code, gid, text) == (65, 7, "A")
            return (((0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (0.0, 1000.0)),)

    glyph = GlyphObservation(
        "A",
        (10.0, 20.0, 12.0, 23.0),
        (10.0, 20.0, 12.0, 23.0),
        1,
        gid=7,
        bitmap_code=65,
        font_decoder=Decoder(),
        glyph_transform=(0.002, 0.001, -0.001, 0.003, 10.0, 20.0),
        text_render_mode=render_mode,
    )
    display_list = DisplayList(100.0, 100.0)
    clipping_subpaths: list[CapturedSubpath] = []

    assert internal_append_glyph_paint(display_list, glyph, clipping_subpaths) is True
    expected_kinds = [] if paint_kind is None else [paint_kind]
    assert [item.kind for item in display_list.items] == expected_kinds
    assert bool(clipping_subpaths) is clips
    if paint_kind is not None:
        item = display_list.items[0]
        assert isinstance(item, PathPaintItem)
        assert item.path.subpaths[0].points == [
            (10.0, 20.0),
            (12.0, 21.0),
            (11.0, 24.0),
            (9.0, 23.0),
        ]


def test_vector_glyph_preserves_stroke_state_and_visibility() -> None:
    class Decoder:
        def glyph_outline(
            self, code: int, gid: int | None, text: str
        ) -> tuple[tuple[tuple[float, float], ...], ...]:
            return (((0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0)),)

    glyph = GlyphObservation(
        "A",
        (0.0, 0.0, 2.0, 3.0),
        (0.0, 0.0, 2.0, 3.0),
        1,
        bitmap_code=65,
        font_decoder=Decoder(),
        glyph_transform=(0.002, 0.0, 0.0, 0.003, 0.0, 0.0),
        text_render_mode=1,
        line_width=4.0,
        line_cap=2,
        line_join=1,
        dash_pattern=([6.0, 2.0], 1.0),
    )
    display_list = DisplayList(10.0, 10.0)

    assert internal_append_glyph_paint(display_list, glyph, []) is True
    item = display_list.items[0]
    assert isinstance(item, PathPaintItem)
    assert item.line_width == 4.0
    assert item.line_cap == 2
    assert item.line_join == 1
    assert item.dash_pattern == ([6.0, 2.0], 1.0)

    glyph.visible = False
    hidden_display_list = DisplayList(10.0, 10.0)
    assert internal_append_glyph_paint(hidden_display_list, glyph, []) is True
    assert hidden_display_list.items == []


def test_text_clip_is_committed_before_the_next_text_object() -> None:
    class Decoder:
        def glyph_outline(
            self, code: int, gid: int | None, text: str
        ) -> tuple[tuple[tuple[float, float], ...], ...]:
            return (((0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (0.0, 1000.0)),)

    class Page:
        media_box = (0.0, 0.0, 20.0, 20.0)
        width = 20.0
        height = 20.0
        page_number = 1
        rotation = 0

        def get_fields(self) -> list[object]:
            return []

        def get_annotations(self) -> list[object]:
            return []

        def resolve_transparency_group_alpha(self) -> None:
            return None

    decoder = Decoder()
    clipping = GlyphObservation(
        "A",
        (1.0, 1.0, 6.0, 6.0),
        (1.0, 1.0, 6.0, 6.0),
        1,
        text_render_mode=7,
        text_object_id=1,
        bitmap_code=65,
        font_decoder=decoder,
        glyph_transform=(0.005, 0.0, 0.0, 0.005, 1.0, 1.0),
    )
    painted = GlyphObservation(
        "A",
        (1.0, 1.0, 6.0, 6.0),
        (1.0, 1.0, 6.0, 6.0),
        2,
        text_render_mode=0,
        text_object_id=2,
        bitmap_code=65,
        font_decoder=decoder,
        glyph_transform=(0.005, 0.0, 0.0, 0.005, 1.0, 1.0),
    )
    products = PageProducts(
        (), GlyphTable.from_rows((clipping, painted)), (), (), LineTable.from_lines(())
    )

    rendered = compose_page(Page(), page_program=PageProgram(products))

    assert [item.kind for item in rendered.display_list.items] == ["clip", "fill"]


def test_shared_page_raster_cache_reuses_identical_crop() -> None:
    cache = ImageCache(max_bytes=1024 * 1024)
    first = rendered_page(width=20, height=20)
    second = rendered_page(width=20, height=20)
    first.image_cache = cache
    second.image_cache = cache
    first.cache_identity = second.cache_identity = ("page", 1, "program", False)

    first_raster = first.rasterize(crop=(0, 0, 10, 10))
    second_raster = second.rasterize(crop=(0, 0, 10, 10))

    assert second_raster is first_raster
    assert cache.stats().hits == 1


def test_axis_aligned_image_rasterizes_native_array_samples() -> None:
    samples = numpy.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=numpy.uint8,
    )
    encoded = bytes(imagecodecs.jpeg_encode(samples, level=100))
    page = rendered_page(width=2, height=2)
    page.display_list.append(
        "image",
        1,
        raw_data=encoded,
        dictionary={
            "Filter": "DCTDecode",
            "Width": 2,
            "Height": 2,
            "ColorSpace": "DeviceRGB",
            "BitsPerComponent": 8,
        },
        items=[("quad", ((0, 0), (2, 0), (0, 2), (2, 2)))],
        bbox=(0, 0, 2, 2),
    )

    raster = page.rasterize(background=(255, 255, 255, 255))

    decoded = numpy.asarray(imagecodecs.jpeg_decode(encoded), dtype=numpy.uint8)
    expected = numpy.full((2, 2, 4), 255, dtype=numpy.uint8)
    expected[:, :, :3] = decoded
    numpy.testing.assert_array_equal(raster.array(), expected)


def test_image_paint_boundary_prepares_without_mutating_source_dictionary() -> None:
    dictionary = {
        "Width": 2,
        "Height": 1,
        "ColorSpace": "DeviceRGB",
        "BitsPerComponent": 8,
        "__soft_mask_raw_data__": bytes((0, 255)),
        "__soft_mask_dictionary__": {
            "Width": 2,
            "Height": 1,
            "ColorSpace": "DeviceGray",
            "BitsPerComponent": 8,
        },
    }
    original = deepcopy(dictionary)
    page = rendered_page(width=2, height=1)
    page.display_list.append(
        "image",
        1,
        raw_data=bytes((10, 20, 30, 40, 50, 60)),
        dictionary=dictionary,
        bbox=(0, 0, 2, 1),
    )

    item = page.display_list.items[0]
    assert isinstance(item, ImagePaintItem)
    assert item.source is not None
    assert item.source_metadata["width"] == 2
    assert item.to_data()["source_metadata"] is item.source_metadata
    assert "image_metadata" not in item.to_data()
    page.rasterize(background=(255, 255, 255, 255), cache=False)
    page.rasterize(background=(255, 255, 255, 255), cache=False)

    assert dictionary == original
    assert "__core_pdf_render_converted_image_data__" not in dictionary


def test_image_decode_recovers_unambiguous_samples_from_malformed_icc() -> None:
    samples = bytes((10, 20, 30, 40, 50, 60))

    decoded = decode_pdf_image(
        samples,
        {
            "Width": 2,
            "Height": 1,
            "BitsPerComponent": 8,
            "ColorSpace": ["ICCBased", object()],
        },
    )

    assert decoded is not None
    assert decoded.channels == 3
    assert decoded.data == samples


def test_image_decode_converts_cmyk_jpeg_samples_to_rgb() -> None:
    samples = numpy.array([[[10, 20, 30, 40]]], dtype=numpy.uint8)
    encoded = bytes(imagecodecs.jpeg_encode(samples, level=100))

    decoded = decode_pdf_image(
        encoded,
        {
            "Filter": "DCTDecode",
            "Width": 1,
            "Height": 1,
            "BitsPerComponent": 8,
            "ColorSpace": "DeviceCMYK",
        },
    )

    assert decoded is not None
    assert decoded.channels == 3
    jpeg_samples = numpy.asarray(imagecodecs.jpeg_decode(encoded), dtype=numpy.uint8)
    expected = ImageColorManager.convert_cmyk(jpeg_samples.reshape(-1))
    numpy.testing.assert_array_equal(decoded.data, expected)


def test_image_source_decodes_once_into_read_only_ndarray() -> None:
    source = ImageSource(
        bytes((10, 20, 30, 40, 50, 60)),
        {
            "Width": 2,
            "Height": 1,
            "ColorSpace": "DeviceRGB",
            "BitsPerComponent": 8,
        },
    )

    first = source.decode()
    second = source.decode()

    assert first is second
    assert first is not None
    assert first.array.shape == (1, 2, 3)
    assert first.array.strides[0] == first.stride
    assert not first.array.flags.writeable


def test_image_source_applies_soft_mask_once() -> None:
    source = ImageSource(
        bytes((10, 20, 30, 40, 50, 60)),
        {
            "Width": 2,
            "Height": 1,
            "ColorSpace": "DeviceRGB",
            "BitsPerComponent": 8,
            "__soft_mask_raw_data__": bytes((0, 255)),
            "__soft_mask_dictionary__": {
                "Width": 2,
                "Height": 1,
                "ColorSpace": "DeviceGray",
                "BitsPerComponent": 8,
            },
        },
    )

    raster = source.decode()

    assert raster is not None
    assert raster.has_alpha
    numpy.testing.assert_array_equal(raster.array[0, :, 3], (0, 255))


def test_image_source_prepares_native_soft_mask_and_reports_all_bytes() -> None:
    source = ImageSource(
        bytes((10, 20, 30)),
        {
            "Width": 1,
            "Height": 1,
            "ColorSpace": "DeviceRGB",
            "BitsPerComponent": 8,
            "__soft_mask_raw_data__": bytes((0, 64, 128, 255)),
            "__soft_mask_dictionary__": {
                "Width": 2,
                "Height": 2,
                "ColorSpace": "DeviceGray",
                "BitsPerComponent": 8,
            },
        },
    )

    prepared = source.prepare()

    assert prepared is source.prepare()
    assert prepared is not None
    assert prepared.soft_mask is not None
    assert prepared.soft_mask.array.shape == (2, 2, 1)
    assert prepared.nbytes == prepared.raster.nbytes + prepared.soft_mask.nbytes
    numpy.testing.assert_array_equal(prepared.soft_mask.array[:, :, 0], ((0, 64), (128, 255)))


def test_image_source_cache_single_flights_preparation() -> None:
    cache = ImageCache(max_bytes=1024)
    entered = Event()
    release = Event()
    second_started = Event()
    preparations: list[ImageSource] = []

    class BlockingImageSource(ImageSource):
        def internal_prepare(self):
            preparations.append(self)
            entered.set()
            assert release.wait(5)
            return super().internal_prepare()

    dictionary = {
        "Width": 1,
        "Height": 1,
        "ColorSpace": "DeviceRGB",
        "BitsPerComponent": 8,
    }
    first_source = BlockingImageSource(
        bytes((10, 20, 30)), dictionary, cache=cache, cache_key=("shared", 1)
    )
    second_source = BlockingImageSource(
        bytes((10, 20, 30)), dictionary, cache=cache, cache_key=("shared", 1)
    )

    def prepare_second():
        second_started.set()
        return second_source.prepare()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(first_source.prepare)
        assert entered.wait(5)
        second_future = executor.submit(prepare_second)
        assert second_started.wait(5)
        assert not second_future.done()
        release.set()
        first = first_future.result(timeout=5)
        second = second_future.result(timeout=5)

    assert first is second
    assert len(preparations) == 1


def test_image_mask_source_exposes_alpha_channel() -> None:
    source = ImageSource(
        bytes((0b10000000,)),
        {
            "Width": 2,
            "Height": 1,
            "ImageMask": True,
            "BitsPerComponent": 1,
        },
    )

    raster = source.decode()

    assert raster is not None
    assert raster.color_model == "gray"
    numpy.testing.assert_array_equal(raster.array[0, :, 1], (255, 0))


def test_image_mask_decode_is_applied_once_at_preparation_boundary() -> None:
    page = rendered_page(width=2, height=1)
    page.display_list.append(
        "image",
        1,
        raw_data=bytes((0b10000000,)),
        dictionary={
            "Width": 2,
            "Height": 1,
            "ImageMask": True,
            "BitsPerComponent": 1,
            "Decode": [1, 0],
        },
        bbox=(0, 0, 2, 1),
    )

    actual = page.rasterize(background=(255, 255, 255, 255)).array()

    numpy.testing.assert_array_equal(actual[0, :, 0], (255, 0))


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_orthogonal_image_blit_preserves_rotation_pixel_order(rotation: int) -> None:
    samples = numpy.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
            [[255, 255, 0], [0, 255, 255]],
        ],
        dtype=numpy.uint8,
    )
    encoded = bytes(imagecodecs.jpeg_encode(samples, level=100))
    page = rendered_page(width=2, height=3, rotate=rotation)
    page.display_list.append(
        "image",
        1,
        raw_data=encoded,
        dictionary={
            "Filter": "DCTDecode",
            "Width": 2,
            "Height": 3,
            "ColorSpace": "DeviceRGB",
            "BitsPerComponent": 8,
        },
        items=[("quad", ((0, 0), (2, 0), (0, 3), (2, 3)))],
        bbox=(0, 0, 2, 3),
    )
    unrotated = rendered_page(width=2, height=3)
    unrotated.display_list.items = page.display_list.items

    source = numpy.frombuffer(
        unrotated.rasterize(background=(255, 255, 255, 255)).pixels,
        dtype=numpy.uint8,
    ).reshape(3, 2, 4)
    actual = numpy.frombuffer(
        page.rasterize(background=(255, 255, 255, 255)).pixels,
        dtype=numpy.uint8,
    ).reshape(page.raster_size()[1], page.raster_size()[0], 4)
    if rotation == 90:
        expected = numpy.rot90(source, k=3)
    elif rotation == 180:
        expected = numpy.rot90(source, k=2)
    else:
        expected = numpy.rot90(source, k=1)

    numpy.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(("empty_first", "expected_tiled"), [(True, 1), (False, 0)])
def test_axis_aligned_image_blit_preserves_empty_clip_subpath_order(
    empty_first: bool,
    expected_tiled: int,
) -> None:
    samples = numpy.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=numpy.uint8,
    )
    clip = CapturedPath()
    if empty_first:
        clip.move_to(0.0, 0.0)
        clip.rect(1.0, 0.0, 1.0, 2.0)
    else:
        clip.rect(1.0, 0.0, 1.0, 2.0)
        clip.move_to(0.0, 0.0)
    page = rendered_page(width=2, height=2)
    page.display_list.append("state-push", 0)
    page.display_list.append("clip", 1, path=clip)
    page.display_list.append(
        "image",
        2,
        raw_data=samples.tobytes(),
        dictionary={
            "Width": 2,
            "Height": 2,
            "ColorSpace": "DeviceRGB",
            "BitsPerComponent": 8,
        },
        items=[("quad", ((0, 0), (2, 0), (0, 2), (2, 2)))],
        bbox=(0, 0, 2, 2),
    )
    page.display_list.append("state-pop", 3)

    actual = page.rasterize(background=(255, 255, 255, 255), cache=False).array()

    expected = numpy.full((2, 2, 4), 255, dtype=numpy.uint8)
    expected[:, 1, :3] = samples[:, 1]
    numpy.testing.assert_array_equal(actual, expected)
    timings = page.metadata["__core_pdf_raster_image_timings__"]
    assert timings["tiled_affine_blit_count"] == expected_tiled
    assert 0 <= timings["tiled_affine_peak_scratch_bytes"] <= 1 << 20


def test_orthogonal_image_quad_rasterizes_rotated_samples() -> None:
    samples = numpy.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
            [[255, 255, 0], [0, 255, 255]],
        ],
        dtype=numpy.uint8,
    )
    page = rendered_page(width=3, height=2)
    page.display_list.append(
        "image",
        1,
        raw_data=samples.tobytes(),
        dictionary={
            "Width": 2,
            "Height": 3,
            "ColorSpace": "DeviceRGB",
            "BitsPerComponent": 8,
        },
        items=[("quad", ((0, 2), (0, 0), (3, 2), (3, 0)))],
        bbox=(0, 0, 3, 2),
    )

    actual = numpy.frombuffer(
        page.rasterize(background=(255, 255, 255, 255)).pixels,
        dtype=numpy.uint8,
    ).reshape(2, 3, 4)

    expected = numpy.full((2, 3, 4), 255, dtype=numpy.uint8)
    expected[:, :, :3] = numpy.rot90(samples, k=3)
    numpy.testing.assert_array_equal(actual, expected)


def test_image_mask_opaque_region_uses_masked_page_view() -> None:
    page = rendered_page(width=2, height=2)
    page.display_list.append(
        "image",
        1,
        raw_data=b"\x80\x40",
        dictionary={
            "ImageMask": True,
            "Width": 2,
            "Height": 2,
            "BitsPerComponent": 1,
        },
        bbox=(0, 0, 2, 2),
    )

    actual = numpy.frombuffer(
        page.rasterize(background=(255, 255, 255, 255)).pixels,
        dtype=numpy.uint8,
    ).reshape(2, 2, 4)
    expected = numpy.full((2, 2, 4), 255, dtype=numpy.uint8)
    expected[0, 0, :3] = 0
    expected[1, 1, :3] = 0

    numpy.testing.assert_array_equal(actual, expected)


def test_image_mask_paints_the_current_fill_colour() -> None:
    # PDF 8.9.6.2: a stencil mask carries no colour samples of its own, so its set
    # bits take the fill colour that was current when it was drawn. Painting them
    # black regardless is only correct by accident, because black is the default
    # fill -- which is why every stencil in the corpus hid this.
    page = rendered_page(width=2, height=2)
    page.display_list.append(
        "image",
        1,
        raw_data=b"\x80\x40",
        dictionary={
            "ImageMask": True,
            "Width": 2,
            "Height": 2,
            "BitsPerComponent": 1,
        },
        bbox=(0, 0, 2, 2),
        fill=(1.0, 0.0, 0.0),
    )

    actual = numpy.frombuffer(
        page.rasterize(background=(255, 255, 255, 255)).pixels,
        dtype=numpy.uint8,
    ).reshape(2, 2, 4)
    expected = numpy.full((2, 2, 4), 255, dtype=numpy.uint8)
    expected[0, 0, :3] = (255, 0, 0)
    expected[1, 1, :3] = (255, 0, 0)

    numpy.testing.assert_array_equal(actual, expected)


def test_image_mask_without_a_recorded_fill_stays_black() -> None:
    # No fill recorded means the drawing never set one, so the PDF default applies.
    page = rendered_page(width=2, height=2)
    page.display_list.append(
        "image",
        1,
        raw_data=b"\x80\x40",
        dictionary={
            "ImageMask": True,
            "Width": 2,
            "Height": 2,
            "BitsPerComponent": 1,
        },
        bbox=(0, 0, 2, 2),
    )

    actual = numpy.frombuffer(
        page.rasterize(background=(255, 255, 255, 255)).pixels,
        dtype=numpy.uint8,
    ).reshape(2, 2, 4)
    expected = numpy.full((2, 2, 4), 255, dtype=numpy.uint8)
    expected[0, 0, :3] = 0
    expected[1, 1, :3] = 0

    numpy.testing.assert_array_equal(actual, expected)


def test_aligned_opaque_glyph_bitmap_expands_through_page_view() -> None:
    page = rendered_page(width=4, height=4)
    page.display_list.append(
        "glyph",
        1,
        bitmap=[1, 2],
        bitmap_width=2,
        bitmap_height=2,
        bbox=(0, 0, 4, 4),
        fill_color=(0, 0, 0),
    )

    actual = numpy.frombuffer(
        page.rasterize(background=(255, 255, 255, 255)).pixels,
        dtype=numpy.uint8,
    ).reshape(4, 4, 4)
    expected = numpy.full((4, 4, 4), 255, dtype=numpy.uint8)
    expected[:2, :2, :3] = 0
    expected[2:, 2:, :3] = 0

    numpy.testing.assert_array_equal(actual, expected)


def test_soft_masked_image_blends_vectorized_samples() -> None:
    samples = numpy.zeros((2, 2, 3), dtype=numpy.uint8)
    mask = numpy.array([[0, 255], [128, 255]], dtype=numpy.uint8)
    page = rendered_page(width=2, height=2)
    page.display_list.append(
        "image",
        1,
        raw_data=samples.tobytes(),
        dictionary={
            "Width": 2,
            "Height": 2,
            "ColorSpace": "DeviceRGB",
            "BitsPerComponent": 8,
            "__soft_mask_raw_data__": mask.tobytes(),
            "__soft_mask_dictionary__": {
                "Width": 2,
                "Height": 2,
                "ColorSpace": "DeviceGray",
                "BitsPerComponent": 8,
            },
        },
        items=[("quad", ((0, 0), (2, 0), (0, 2), (2, 2)))],
        bbox=(0, 0, 2, 2),
    )

    actual = numpy.frombuffer(
        page.rasterize(background=(255, 255, 255, 255)).pixels,
        dtype=numpy.uint8,
    ).reshape(2, 2, 4)

    assert actual[0, 0, 0] == 255
    assert actual[0, 1, 0] == 0
    assert actual[1, 0, 0] == 127
    assert actual[1, 1, 0] == 0


def test_shared_image_preserves_higher_resolution_soft_mask() -> None:
    dictionary = {
        "Width": 1,
        "Height": 1,
        "ColorSpace": "DeviceRGB",
        "BitsPerComponent": 8,
        "__soft_mask_raw_data__": bytes((0, 255, 0, 255)),
        "__soft_mask_dictionary__": {
            "Width": 2,
            "Height": 2,
            "ColorSpace": "DeviceGray",
            "BitsPerComponent": 8,
        },
    }
    source = ImageSource(bytes((0, 0, 0)), dictionary)
    page = rendered_page(width=2, height=2)
    page.display_list.append(
        "image",
        1,
        raw_data=source.raw,
        dictionary=dictionary,
        image_source=source,
        items=[("quad", ((0, 0), (2, 0), (0, 2), (2, 2)))],
        bbox=(0, 0, 2, 2),
    )

    actual = numpy.frombuffer(
        page.rasterize(background=(255, 255, 255, 255)).pixels,
        dtype=numpy.uint8,
    ).reshape(2, 2, 4)

    numpy.testing.assert_array_equal(actual[:, :, 0], ((255, 0), (255, 0)))


def test_affine_image_blit_preserves_sheared_source_samples() -> None:
    samples = numpy.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=numpy.uint8,
    )
    encoded = bytes(imagecodecs.jpeg_encode(samples, level=100))
    page = rendered_page(width=3, height=2)
    page.display_list.append(
        "image",
        1,
        raw_data=encoded,
        dictionary={
            "Filter": "DCTDecode",
            "Width": 2,
            "Height": 2,
            "ColorSpace": "DeviceRGB",
            "BitsPerComponent": 8,
        },
        items=[("quad", ((0, 0), (2, 0), (1, 2), (3, 2)))],
        bbox=(0, 0, 3, 2),
    )

    decoded = numpy.asarray(imagecodecs.jpeg_decode(encoded), dtype=numpy.uint8)
    expected = numpy.full((2, 3, 4), 255, dtype=numpy.uint8)
    expected[0, 1, :3] = decoded[0, 0]
    expected[0, 2, :3] = decoded[0, 1]
    expected[1, 0, :3] = decoded[1, 0]
    expected[1, 1, :3] = decoded[1, 1]

    actual = numpy.frombuffer(
        page.rasterize(background=(255, 255, 255, 255)).pixels,
        dtype=numpy.uint8,
    ).reshape(2, 3, 4)

    numpy.testing.assert_array_equal(actual, expected)


def test_captured_glyph_bitmap_rows_use_top_to_bottom_order() -> None:
    page = rendered_page()
    page.display_list.append(
        "glyph",
        1,
        bitmap=(1, 1, 1, 1, 1, 1, 31),
        bitmap_width=5,
        bitmap_height=7,
        bbox=(0, 0, 5, 7),
        fill_color=(0, 0, 0),
    )

    raster = page.rasterize(background=(255, 255, 255, 255))

    top_row = [raster.pixels[x * 4] for x in range(5)]
    bottom_row_start = 6 * 5 * 4
    bottom_row = [raster.pixels[bottom_row_start + x * 4] for x in range(5)]
    assert top_row == [0, 255, 255, 255, 255]
    assert bottom_row == [0, 0, 0, 0, 0]

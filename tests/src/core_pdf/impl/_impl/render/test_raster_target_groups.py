# SPDX-License-Identifier: AGPL-3.0-only
"""Transparency groups must own every fast-path paint operation."""

import numpy
import pytest

from core_pdf.impl._impl.render.clipping import internal_ClipState
from core_pdf.impl._impl.render.model import DisplayListItem
from core_pdf.impl._impl.render.target import internal_RasterTarget
from tests.helpers.pdf_bytes import one_page_pdf, open_pdf, stream_obj


def internal_target(size: int = 32) -> internal_RasterTarget:
    pixels = bytearray(size * size * 4)
    return internal_RasterTarget(
        pixels,
        None,
        clip=internal_ClipState(crop_x0=0, crop_y1=size, scale=1, width=size, height=size),
        width=size,
        height=size,
        scale=1,
        crop_x0=0,
        crop_y0=0,
        crop_y1=size,
        page_view=numpy.frombuffer(pixels, dtype=numpy.uint8).reshape(size, size, 4),
    )


@pytest.mark.parametrize("shape", ["image", "line", "circle"])
def test_fast_paint_follows_nested_group_buffers(shape: str) -> None:
    def paint(target: internal_RasterTarget) -> None:
        if shape == "image":
            assert target.blit_affine_image(
                ((0.0, 0.0), (32.0, 0.0), (0.0, 32.0), (32.0, 32.0)),
                bytes((255, 0, 0)),
                1,
                1,
                3,
                None,
                None,
            )
        elif shape == "line":
            target.fill_line(3, 3, 28, 28, 3, (255, 0, 0, 255))
        else:
            target.fill_circle(16, 16, 8, (255, 0, 0, 255))

    reference = internal_target()
    paint(reference)
    assert any(reference.pixels)

    target = internal_target()
    root = target.pixels
    outer = bytearray(len(root))
    inner = bytearray(len(root))
    target.push_group(outer, None, None)
    target.push_group(inner, None, None)
    paint(target)
    assert inner == reference.pixels
    assert not any(outer)
    assert not any(root)

    target.pop_group()
    paint(target)
    assert outer == reference.pixels
    assert not any(root)

    target.pop_group()
    paint(target)
    assert root == reference.pixels


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize(
    "matrix",
    [(4, 0, 0, 4, 2, 2), (-4, 0, 0, 4, 6, 2), (0, 4, -4, 0, 6, 2), (4, 1, 1, 4, 1, 1)],
)
def test_zero_alpha_group_image_leaves_backdrop_unchanged(
    inline: bool, matrix: tuple[int, ...]
) -> None:
    image = b"BI /W 1 /H 1 /BPC 8 /CS /RGB ID \xff\x00\x00 EI" if inline else b"/Im Do"
    data = one_page_pdf(
        b"/Zero gs /Fm Do",
        media_box=(0, 0, 8, 8),
        resources=b"<< /ExtGState << /Zero << /ca 0 >> >> /XObject << /Fm 6 0 R >> >>",
        extra_objects=[
            stream_obj(
                " ".join(map(str, matrix)).encode() + b" cm " + image,
                b"/Type /XObject /Subtype /Form /BBox [0 0 8 8] "
                b"/Group << /S /Transparency /I true >> "
                b"/Resources << /XObject << /Im 7 0 R >> >>",
            ),
            stream_obj(
                bytes((255, 0, 0)),
                b"/Type /XObject /Subtype /Image /Width 1 /Height 1 "
                b"/ColorSpace /DeviceRGB /BitsPerComponent 8",
            ),
        ],
    )
    background = (10, 20, 30, 255)
    with open_pdf(data) as document:
        rendered = document.pages[0].render()
        markers = [
            item
            for item in rendered.display_list.items
            if isinstance(item, DisplayListItem) and item.kind == "group-begin"
        ]
        assert len(markers) == 1
        assert markers[0].data["fill_opacity"] == 0
        pixels = rendered.rasterize(background=background).array()
    assert numpy.all(pixels == background)

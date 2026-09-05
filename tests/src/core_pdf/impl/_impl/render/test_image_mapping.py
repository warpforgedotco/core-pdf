# SPDX-License-Identifier: AGPL-3.0-only
"""Image placement regressions verified with qpdf 12.3.2 and Poppler 26.07.0."""

import numpy
import pytest

from core_pdf.impl._impl.render.model import ImagePaintItem
from tests.helpers.pdf_bytes import one_page_pdf, open_pdf, stream_obj

MATRICES = (
    (8, 0, 0, 8, 2, 2),
    (-8, 0, 0, 8, 10, 2),
    (8, 0, 0, -8, 2, 10),
    (0, 8, -8, 0, 10, 2),
    (0, -8, 8, 0, 2, 10),
    (6, 2, 2, 6, 2, 2),
)
COLORS = bytes((255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0))
ALPHAS = bytes((255, 128, 64, 0))


def internal_image_pdf(
    matrix: tuple[int, ...],
    *,
    mask: bytes | None = ALPHAS,
    mask_size: tuple[int, int] = (2, 2),
    stencil: bool = False,
    gray: bool = False,
    prefix: bytes = b"",
    image_size: tuple[int, int] = (2, 2),
    colors: bytes = COLORS,
) -> bytes:
    if stencil:
        image = stream_obj(
            b"\x40\x80", b"/Type /XObject /Subtype /Image /Width 2 /Height 2 /ImageMask true"
        )
    else:
        width, height = image_size
        image = stream_obj(
            bytes((20, 80, 160, 240)) if gray else colors,
            f"/Type /XObject /Subtype /Image /Width {width} /Height {height} ".encode()
            + (b"/ColorSpace /DeviceGray " if gray else b"/ColorSpace /DeviceRGB ")
            + b"/BitsPerComponent 8"
            + (b" /SMask 7 0 R" if mask is not None else b""),
        )
    width, height = mask_size
    return one_page_pdf(
        prefix + b" 1 0 0 rg " + " ".join(map(str, matrix)).encode() + b" cm /Im Do",
        media_box=(0, 0, 12, 12),
        resources=b"<< /XObject << /Im 6 0 R >> >>",
        extra_objects=[
            image,
            stream_obj(
                mask if mask is not None else bytes(width * height),
                f"/Type /XObject /Subtype /Image /Width {width} /Height {height} ".encode()
                + b"/ColorSpace /DeviceGray /BitsPerComponent 8",
            ),
        ],
    )


@pytest.mark.parametrize("matrix", MATRICES)
@pytest.mark.parametrize("stencil", [False, True])
def test_color_and_mask_follow_the_original_image_transform(
    matrix: tuple[int, ...], stencil: bool
) -> None:
    # qpdf validated all twelve PDFs; Poppler confirmed these interior samples
    # after rotation, reflection and shear, without relying on edge antialiasing.
    data = internal_image_pdf(matrix, stencil=stencil)
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize(background=(255, 255, 255, 255)).array()
    expected = (
        ((255, 0, 0), (255, 255, 255), (255, 255, 255), (255, 0, 0))
        if stencil
        else ((255, 0, 0), (127, 255, 127), (191, 191, 255), (255, 255, 255))
    )
    a, b, c, d, e, f = matrix
    for (u, v), color in zip(((0.25, 0.75), (0.75, 0.75), (0.25, 0.25), (0.75, 0.25)), expected):
        x, y = int(u * a + v * c + e), 11 - int(u * b + v * d + f)
        assert tuple(pixels[y, x, :3]) == color


@pytest.mark.parametrize("matrix", MATRICES)
@pytest.mark.parametrize("stencil", [False, True])
def test_cropping_restricts_destination_without_rescaling_the_source(
    matrix: tuple[int, ...], stencil: bool
) -> None:
    with open_pdf(internal_image_pdf(matrix, stencil=stencil)) as document:
        page = document.pages[0].render()
        full = page.rasterize().array()
        crop = page.rasterize(crop=(4, 4, 10, 10)).array()
    numpy.testing.assert_array_equal(crop, full[2:8, 4:10])


@pytest.mark.parametrize("matrix", MATRICES)
@pytest.mark.parametrize("gray", [False, True])
@pytest.mark.parametrize("clip", [b"", b"2 2 m 10 2 l 2 10 l h W n"])
def test_opaque_soft_mask_has_the_same_placement_as_an_unmasked_image(
    matrix: tuple[int, ...], gray: bool, clip: bytes
) -> None:
    rasters = []
    for mask in (None, bytes((255,)) * 16):
        data = internal_image_pdf(matrix, mask=mask, mask_size=(4, 4), gray=gray, prefix=clip)
        with open_pdf(data) as document:
            rasters.append(document.pages[0].render().rasterize().array())
    numpy.testing.assert_array_equal(*rasters)


def test_native_soft_mask_keeps_its_own_sample_resolution() -> None:
    data = internal_image_pdf(
        MATRICES[0],
        mask=bytes((255, 0, 128, 64, 32, 64, 0, 255)),
        mask_size=(4, 2),
        image_size=(2, 1),
        colors=bytes((255, 0, 0, 0, 255, 0)),
    )
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize(background=(255, 255, 255, 255)).array()
    # Poppler confirms all eight alpha samples, including the ones that would
    # disappear if this 4x2 mask were first reduced to the 2x1 color dimensions.
    assert pixels[3, [3, 5, 7, 9], :3].tolist() == [
        [255, 0, 0],
        [255, 255, 255],
        [127, 255, 127],
        [191, 255, 191],
    ]
    assert pixels[7, [3, 5, 7, 9], :3].tolist() == [
        [255, 223, 223],
        [255, 191, 191],
        [255, 255, 255],
        [0, 255, 0],
    ]


def test_nonrectangular_clip_keeps_color_and_partial_alpha_in_the_same_coordinates() -> None:
    data = internal_image_pdf(MATRICES[0], prefix=b"2 2 m 10 2 l 2 10 l h W n")
    with open_pdf(data) as document:
        page = document.pages[0].render()
        pixels = page.rasterize(background=(255, 255, 255, 255)).array()
        crop = page.rasterize(background=(255, 255, 255, 255), crop=(4, 4, 10, 10)).array()
    assert tuple(pixels[7, 3, :3]) == (191, 191, 255)
    assert tuple(pixels[3, 7, :3]) == (255, 255, 255)
    numpy.testing.assert_array_equal(crop, pixels[2:8, 4:10])


@pytest.mark.parametrize("stencil", [False, True])
def test_form_bbox_clips_image_destination_without_changing_placement(stencil: bool) -> None:
    image = stream_obj(
        b"\x00" if stencil else bytes((255, 0, 0)),
        b"/Type /XObject /Subtype /Image /Width 1 /Height 1 "
        + (b"/ImageMask true" if stencil else b"/ColorSpace /DeviceRGB /BitsPerComponent 8"),
    )
    data = one_page_pdf(
        b"/Fm Do",
        media_box=(0, 0, 100, 100),
        resources=b"<< /XObject << /Fm 6 0 R >> >>",
        extra_objects=[
            stream_obj(
                b"1 0 0 rg 100 0 0 100 0 0 cm /Im Do",
                b"/Type /XObject /Subtype /Form /BBox [10 10 30 30] "
                b"/Resources << /XObject << /Im 7 0 R >> >>",
            ),
            image,
        ],
    )
    with open_pdf(data) as document:
        pixels = document.pages[0].render().rasterize().array()
    # Both RGB and stencil PDFs paint exactly this 20x20 area in Poppler.
    assert numpy.count_nonzero(pixels[:, :, 3]) == 400
    assert numpy.all(pixels[70:90, 10:30] == (255, 0, 0, 255))


def test_reduction_uses_quad_extent_instead_of_a_restricted_display_bbox() -> None:
    colors = bytes(value for sample in range(64) for value in (sample * 4,) * 3)
    data = internal_image_pdf(MATRICES[0], mask=None, image_size=(64, 1), colors=colors)
    with open_pdf(data) as document:
        page = document.pages[0].render()
        full = page.rasterize().array()
        for item in page.display_list.items:
            if isinstance(item, ImagePaintItem):
                item.bbox = (4, 4, 6, 6)
        restricted = page.rasterize().array()
    numpy.testing.assert_array_equal(restricted, full)

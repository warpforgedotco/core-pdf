# SPDX-License-Identifier: AGPL-3.0-only


import gzip
import math
import pickle
from pathlib import Path

import pytest

from core_pdf_cythonized import horizontal_glyph_geometry

GOLDEN = pickle.loads(
    gzip.decompress((Path(__file__).parent / "glyph_geometry_golden.pkl.gz").read_bytes())
)


def deep_repr(value: object) -> object:
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return [deep_repr(item) for item in value]
    return value


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 713
    assert sum(len(kwargs["offsets"]) for kwargs, _ in GOLDEN) > 3000
    bases = {kwargs["basis"][3:5] for kwargs, _ in GOLDEN}
    assert (0.0, 0.0) in bases, "no axis-aligned case"
    assert any(shear != (0.0, 0.0) for shear in bases), "no sheared case"
    assert any(
        kwargs["clip_primary"] is not None or kwargs["clip_page"] is not None
        for kwargs, _ in GOLDEN
    )


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_golden_vector(index: int) -> None:
    kwargs, expected = GOLDEN[index]
    offsets = kwargs.pop("offsets")
    advances = kwargs.pop("advances")
    glyph_boxes = kwargs.pop("glyph_boxes")
    produced = horizontal_glyph_geometry(offsets, advances, glyph_boxes, **kwargs)
    kwargs["offsets"], kwargs["advances"], kwargs["glyph_boxes"] = offsets, advances, glyph_boxes
    assert deep_repr(produced[:6]) == deep_repr(expected)


@pytest.mark.parametrize("index", range(0, len(GOLDEN), 3))
def test_the_run_unions_are_run_geometry_adds(index: int) -> None:
    pytest.importorskip("core_pdf")
    from core_pdf.impl.capture_glyphs import RunGeometry

    kwargs, _ = GOLDEN[index]
    arguments = dict(kwargs)
    produced = horizontal_glyph_geometry(
        arguments.pop("offsets"),
        arguments.pop("advances"),
        arguments.pop("glyph_boxes"),
        **arguments,
    )
    geometry = RunGeometry()
    for advance, ink in zip(produced[0], produced[3], strict=True):
        geometry.add(advance, ink, None)
    if not produced[0]:
        assert produced[6] is None
        assert produced[7] is None
        return
    assert repr(produced[6]) == repr(geometry.advance)
    assert repr(produced[7]) == repr(geometry.ink)


def test_kernel_rejects_a_short_glyph_box_array():
    with pytest.raises((IndexError, ValueError)):
        horizontal_glyph_geometry(
            [0.0, 6.0],
            [6.0, 6.0],
            [0.0, 0.0, 1.0, 1.0],
            basis=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
            font_ascent=0.75,
            font_descent=-0.22,
            rise=0.0,
            font_scale=0.012,
            advance_scale=0.012,
            font_size=12.0,
            clip_primary=None,
            clip_page=None,
            visible=True,
            want_bitmap=[1, 1],
        )


@pytest.mark.parametrize("font_size", [1e9, 1e10, 1e300])
def test_a_huge_font_size_clamps_rather_than_overflowing(font_size: float) -> None:
    plane = horizontal_glyph_geometry(
        [0.0],
        [6.0],
        [0.0, 0.0, 0.5, 0.7],
        basis=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        font_ascent=0.8,
        font_descent=-0.2,
        rise=0.0,
        font_scale=0.012,
        advance_scale=0.012,
        font_size=font_size,
        clip_primary=None,
        clip_page=None,
        visible=True,
        want_bitmap=[1],
    )
    height = max(16, min(64, math.ceil(max(font_size, 1.0) * 2.5)))
    width = max(1, min(96, math.ceil(height * 0.5 / 0.7)))
    assert plane[5] == [width, height]


def test_a_degenerate_glyph_box_clamps_the_bitmap_width() -> None:
    plane = horizontal_glyph_geometry(
        [0.0],
        [6.0],
        [0.0, 0.0, 1.0, 1e-300],
        basis=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        font_ascent=0.8,
        font_descent=-0.2,
        rise=0.0,
        font_scale=0.012,
        advance_scale=0.012,
        font_size=12.0,
        clip_primary=None,
        clip_page=None,
        visible=True,
        want_bitmap=[1],
    )
    assert plane[5] == [96, 30]

from typing import Any, cast

import pytest

from core_pdf._vendor.fontTools.misc.psCharStrings import T1CharString
from core_pdf.impl.engine.spec.s_09_fonts.font_program_opentype import OpenTypeFontProgram
from core_pdf.impl.engine.spec.s_09_fonts.font_program_type1 import Type1FontProgram

internal_CURVE_PROGRAM = [
    0,
    500,
    "hsbw",
    0,
    0,
    "rmoveto",
    0,
    100,
    100,
    -100,
    0,
    0,
    "rrcurveto",
    "closepath",
    "endchar",
]


def internal_type1_curve_program() -> Type1FontProgram:
    charstring = T1CharString(program=internal_CURVE_PROGRAM, subrs=[])
    program = cast(Any, object.__new__(Type1FontProgram))
    program.charstrings = {"curve": charstring}
    program.font_matrix = (0.0, 0.002, -0.001, 0.0, 0.01, -0.02)
    program.glyph_names = ("curve",)
    program.glyph_name_to_id = {"curve": 0}
    program.internal_contour_cache = {}
    program.subrs = []
    return cast(Type1FontProgram, program)


class internal_CubicGlyph:
    def draw(self, pen: Any) -> None:
        pen.moveTo((0, 0))
        pen.curveTo((0, 100), (100, 0), (100, 0))
        pen.closePath()


class internal_FakeOpenTypeFont:
    def getGlyphName(self, glyph_id: int) -> str:
        if glyph_id != 0:
            raise IndexError(glyph_id)
        return "curve"


def internal_opentype_curve_program() -> OpenTypeFontProgram:
    program = cast(Any, object.__new__(OpenTypeFontProgram))
    program.font = internal_FakeOpenTypeFont()
    program.glyph_count = 1
    program.internal_contour_cache = {}
    program.internal_glyph_set = {"curve": internal_CubicGlyph()}
    program.reverse_glyph_map = {"curve": 0}
    program.units_per_em = 2000.0
    return cast(OpenTypeFontProgram, program)


def test_type1_bbox_uses_exact_transformed_bezier_extrema() -> None:
    program = internal_type1_curve_program()

    contours = program.normalized_glyph_contours(0)
    sampled_x_min = min(x for contour in contours for x, ignored_y in contour)
    bbox = program.glyph_bbox_for_gid(0)

    # The curve's native y maximum is 400/9 at t=1/3. The Type 1 FontMatrix
    # rotates and translates it to x = 10-y, whose exact minimum does not lie
    # on the contour flattener's eight-point sampling grid.
    assert bbox == pytest.approx((-310 / 9, -20, 10, 180))
    assert sampled_x_min > bbox[0]


def test_opentype_bbox_uses_exact_normalized_bezier_extrema() -> None:
    program = internal_opentype_curve_program()

    contours = program.normalized_glyph_contours(0)
    sampled_y_max = max(y for contour in contours for ignored_x, y in contour)
    bbox = program.glyph_bbox_for_gid(0)

    # Normalizing a 2000-unit em scales the exact 400/9 native maximum by 1/2.
    assert bbox == pytest.approx((0, 0, 50, 200 / 9))
    assert sampled_y_max < bbox[3]

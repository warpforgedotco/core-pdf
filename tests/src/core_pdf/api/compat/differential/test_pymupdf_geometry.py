from pathlib import Path
from typing import Any

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .support import FIXTURES_ROOT

real_pymupdf = pytest.importorskip("pymupdf")
pytestmark = pytest.mark.compat_differential
internal_PDFS = tuple(
    FIXTURES_ROOT / "PyMuPDF/tests/resources" / name
    for name in ("small-table.pdf", "test-3143.pdf", "test_4043.pdf")
)


def internal_geometry_expected(value: Any) -> Any:
    if isinstance(value, float):
        return pytest.approx(value, rel=1e-6, abs=1e-5)
    if isinstance(value, tuple):
        return tuple(internal_geometry_expected(item) for item in value)
    if isinstance(value, list):
        return [internal_geometry_expected(item) for item in value]
    if isinstance(value, dict):
        return {key: internal_geometry_expected(item) for key, item in value.items()}
    return value


@pytest.mark.parametrize("pdf_path", internal_PDFS, ids=lambda path: path.name)
def test_points_and_affine_page_coordinates(pdf_path: Path) -> None:
    def snapshot(module: Any) -> list[Any]:
        with module.open(pdf_path) as document:
            rect = module.Rect(document[0].rect)
            output = []
            for coordinates in (
                (0, 0),
                (rect.width, rect.height),
                (-rect.width / 3, rect.height / 7),
            ):
                point = module.Point(coordinates)
                matrix = module.Matrix(1.25, 0.25, -0.5, 0.75, 10, -20)
                output.append(
                    {
                        "repr": repr(point),
                        "sequence": tuple(point),
                        "norm": point.norm(),
                        "unit": tuple(point.unit),
                        "abs_unit": tuple(point.abs_unit),
                        "distance": [
                            point.distance_to(rect.br, unit) for unit in ("px", "in", "cm", "mm")
                        ],
                        "rect_distance": point.distance_to(rect),
                        "transform": tuple(point * matrix),
                        "inverse": tuple(point / matrix),
                        "sum": tuple(point + (3, 4)),
                        "difference": tuple(point - 2),
                        "scale": tuple(point * 2),
                        "divide": tuple(point / 2),
                        "positive": tuple(+point),
                        "negative": tuple(-point),
                        "absolute": abs(point),
                        "bool": bool(point),
                    }
                )
                transformed = module.Point(point)
                assert transformed.transform(matrix) is transformed
                output.append(tuple(transformed))
            return output

    assert snapshot(compat_pymupdf) == internal_geometry_expected(snapshot(real_pymupdf))


@pytest.mark.parametrize("pdf_path", internal_PDFS, ids=lambda path: path.name)
def test_rectangles_and_page_regions(pdf_path: Path) -> None:
    def snapshot(module: Any) -> list[Any]:
        with module.open(pdf_path) as document:
            page = document[0]
            assert isinstance(page.rect, module.Rect)
            bounds = module.Rect(page.rect)
            output = []
            regions = (
                tuple(bounds),
                (1.25, 2.5, bounds.width / 2, bounds.height / 2),
                (4, 3, 2, 1),
                (0, 0, 0, 2),
                (0, 0, 0, 0),
            )
            for values in regions:
                rect = module.Rect(values)
                matrix = module.Matrix(30)
                output.append(
                    {
                        "repr": repr(rect),
                        "size": (rect.width, rect.height),
                        "state": (rect.is_empty, rect.is_valid, rect.is_infinite),
                        "corners": [tuple(p) for p in (rect.tl, rect.tr, rect.bl, rect.br)],
                        "aliases": [
                            tuple(p)
                            for p in (
                                rect.top_left,
                                rect.top_right,
                                rect.bottom_left,
                                rect.bottom_right,
                            )
                        ],
                        "area": [rect.get_area(unit) for unit in ("px", "in", "cm", "mm")],
                        "norm": rect.norm(),
                        "absolute": abs(rect),
                        "contains": [
                            rect.contains(p)
                            for p in (rect.tl, rect.br, bounds, (2, 2), (2, 2, 2, 2))
                        ],
                        "intersects": rect.intersects(bounds),
                        "intersection": tuple(rect & bounds),
                        "union": tuple(rect | bounds),
                        "point_union": tuple(module.Rect(rect).include_point(bounds.br)),
                        "normalized": tuple(module.Rect(rect).normalize()),
                        "transformed": tuple(rect * matrix),
                        "inverse": tuple(rect / matrix),
                        "rounded": tuple(rect.round()),
                        "irect": tuple(rect.irect),
                        "copy_points": tuple(module.Rect(rect.tl, rect.br)),
                        "mixed_points": tuple(module.Rect(rect.x0, rect.y0, rect.br)),
                        "keywords": tuple(module.Rect(rect, x0=10)),
                        "arithmetic": [
                            tuple(rect + 1),
                            tuple(rect - (1, 2, 3, 4)),
                            tuple(rect * 2),
                            tuple(rect / 2),
                        ],
                    }
                )
                if not rect.is_empty:
                    output.append(tuple(rect.torect(bounds)))
            return output

    assert snapshot(compat_pymupdf) == internal_geometry_expected(snapshot(real_pymupdf))


@pytest.mark.parametrize("pdf_path", internal_PDFS, ids=lambda path: path.name)
def test_integer_pixel_bounds(pdf_path: Path) -> None:
    def snapshot(module: Any) -> list[Any]:
        with module.open(pdf_path) as document:
            width, height = module.Rect(document[0].rect).br
            output = []
            for coords in (
                (0, 0, width, height),
                (-1.8, 2.8, 3.3, 4.4),
                (0.0009, 0.0009, 1.0009, 1.0009),
                (4, 3, 2, 1),
            ):
                integer = module.IRect(coords)
                output.append(
                    (
                        repr(integer),
                        tuple(integer),
                        tuple(integer.rect),
                        tuple(module.Rect(coords).round()),
                        tuple(integer * 2),
                        tuple(integer / 2),
                        tuple(integer * module.Matrix(30)),
                        tuple(integer * module.Matrix(0.9999, 0.9999)),
                        tuple(integer / module.Matrix(1.0001, 1.0001)),
                    )
                )
            return output

    assert snapshot(compat_pymupdf) == snapshot(real_pymupdf)


@pytest.mark.parametrize("pdf_path", internal_PDFS, ids=lambda path: path.name)
def test_quadrilaterals_and_fixed_point_transforms(pdf_path: Path) -> None:
    def snapshot(module: Any) -> list[Any]:
        with module.open(pdf_path) as document:
            bounds = module.Rect(document[0].rect)
            cases = (
                bounds.quad,
                module.Quad(),
                module.Quad((0, 0), (4, 0), (1, 2), (3, 2)),
                module.Quad((0, 0), (2, 2), (0, 2), (2, 0)),
                module.Quad((0, 0), (1, 1), (2, 2), (3, 3)),
            )
            output = []
            matrix = module.Matrix(30)
            pivot = bounds.br / 2
            for quad in cases:
                output.append(
                    {
                        "points": [tuple(p) for p in quad],
                        "repr": repr(quad),
                        "size": (quad.width, quad.height),
                        "absolute": abs(quad),
                        "bounds": tuple(quad.rect),
                        "state": (
                            quad.is_empty,
                            quad.is_infinite,
                            quad.is_convex,
                            quad.is_rectangular,
                        ),
                        "transformed": [tuple(p) for p in quad * matrix],
                        "morphed": [tuple(p) for p in quad.morph(pivot, matrix)],
                        "divide": [tuple(p) for p in quad / 2],
                        "sum": [tuple(p) for p in quad + 1],
                        "difference": [tuple(p) for p in quad - 1],
                    }
                )
                copy = module.Quad(quad)
                assert copy.transform(matrix) is copy
                output.append([tuple(p) for p in copy])
            output.append([tuple(p) for p in bounds.morph(pivot, matrix)])
            return output

    assert snapshot(compat_pymupdf) == internal_geometry_expected(snapshot(real_pymupdf))

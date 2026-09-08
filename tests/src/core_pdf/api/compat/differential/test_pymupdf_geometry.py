import sys
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


@pytest.mark.parametrize("pdf_path", internal_PDFS, ids=lambda path: path.name)
def test_integer_rectangle_mutation_and_errors(pdf_path: Path) -> None:
    def snapshot(module: Any) -> list[Any]:
        with module.open(pdf_path) as document:
            width, height = module.Rect(document[0].rect).br
            output = []
            for coordinates in ((1, 2, width, height), (4, 3, 2, 1), (0, 0, 0, 0)):
                for operation in (
                    lambda r: r.include_rect((0.0009, 0.0009, width + 0.0009, height + 0.0009)),
                    lambda r: r.include_point((width + 0.0009, height + 0.0009)),
                    lambda r: r.intersect((1.5, 2.5, 3.5, 4.5)),
                    lambda r: r.transform(module.Matrix(30)),
                    lambda r: r & (1.5, 2.5, 3.5, 4.5),
                    lambda r: r | (0.0009, 0.0009, width + 0.0009, height + 0.0009),
                    lambda r: r.normalize(),
                    lambda r: abs(r),
                ):
                    rectangle = module.IRect(coordinates)
                    try:
                        result = operation(rectangle)
                        outcome = ("returned", repr(result), result is rectangle)
                    except Exception as error:
                        outcome = ("raised", type(error).__name__, str(error))
                    output.append(
                        (outcome, repr(rectangle), tuple(type(v).__name__ for v in rectangle))
                    )
            rectangle = module.IRect(1, 2, 3, 4)
            output.append(
                (
                    isinstance(rectangle, module.Rect),
                    hasattr(rectangle, "round"),
                    hasattr(rectangle, "irect"),
                )
            )
            return output

    assert snapshot(compat_pymupdf) == snapshot(real_pymupdf)


def internal_page_geometry_snapshot(module: Any, source: bytes) -> dict[str, Any]:
    with module.open(stream=source, filetype="pdf") as document:
        page = document[0]
        names = ("mediabox", "cropbox", "bleedbox", "trimbox", "artbox", "rect")
        for name in names:
            assert isinstance(getattr(page, name), module.Rect)
        return {
            "boxes": {name: tuple(getattr(page, name)) for name in names},
            "bound": tuple(page.bound()),
            "position": tuple(page.cropbox_position),
            "rotation": page.rotation,
            "transformation": tuple(page.transformation_matrix),
            "rotation_matrix": tuple(page.rotation_matrix),
            "derotation_matrix": tuple(page.derotation_matrix),
        }


@pytest.mark.parametrize("pdf_path", internal_PDFS, ids=lambda path: path.name)
def test_page_box_types_and_coordinate_matrices(pdf_path: Path) -> None:
    source = pdf_path.read_bytes()
    expected = internal_page_geometry_snapshot(real_pymupdf, source)
    actual = internal_page_geometry_snapshot(compat_pymupdf, source)
    assert actual == internal_geometry_expected(expected)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("unit", [0, 1, 2, -2])
def test_offset_page_boxes_and_user_units(rotation: int, unit: int) -> None:
    pdf_path = internal_PDFS[0]
    # Both readers receive the same in-memory variant; the upstream fixture is untouched.
    with real_pymupdf.open(pdf_path) as fixture:
        page = fixture[0]
        page.set_mediabox(real_pymupdf.Rect(10, 20, 610, 820))
        page.set_cropbox(real_pymupdf.Rect(30, 40, 530, 740))
        page.set_bleedbox(real_pymupdf.Rect(35, 45, 525, 735))
        page.set_trimbox(real_pymupdf.Rect(40, 50, 520, 730))
        page.set_artbox(real_pymupdf.Rect(45, 55, 515, 725))
        page.set_rotation(rotation)
        fixture.xref_set_key(page.xref, "UserUnit", str(unit))
        source = fixture.tobytes()
    expected = internal_page_geometry_snapshot(real_pymupdf, source)
    actual = internal_page_geometry_snapshot(compat_pymupdf, source)
    assert actual == internal_geometry_expected(expected)


@pytest.mark.parametrize("pdf_path", internal_PDFS, ids=lambda path: path.name)
def test_page_geometry_copy_and_closed_handle_behavior(pdf_path: Path) -> None:
    def snapshot(module: Any) -> list[Any]:
        output = []
        with module.open(pdf_path) as document:
            page = document[0]
            for name in ("mediabox", "cropbox", "bleedbox", "trimbox", "artbox", "rect"):
                box = getattr(page, name)
                box.x0 += 100
                output.append((name, tuple(getattr(page, name))))
        for name in (
            "mediabox",
            "cropbox",
            "bleedbox",
            "trimbox",
            "artbox",
            "rect",
            "cropbox_position",
            "transformation_matrix",
            "rotation_matrix",
            "derotation_matrix",
        ):
            try:
                getattr(page, name)
            except Exception as error:
                output.append((name, type(error).__name__, str(error)))
            else:
                pytest.fail(f"closed page accepted {name}")
        return output

    assert snapshot(compat_pymupdf) == snapshot(real_pymupdf)


def test_rotation_matrices_keep_unnormalized_angles() -> None:
    """PyMuPDF feeds the raw angle to `math.radians`, so signed zeros must survive."""

    def snapshot(module: Any) -> list[Any]:
        return [
            (angle, repr(module.Matrix(angle)), tuple(module.Matrix(angle)))
            for angle in (
                -450,
                -360,
                -270,
                -180,
                -90,
                -45,
                -0.0,
                0,
                37,
                90,
                180,
                270,
                360,
                450,
                720,
                1e10,
            )
        ]

    assert snapshot(compat_pymupdf) == snapshot(real_pymupdf)


@pytest.mark.parametrize(
    "determinant",
    [
        0.0,
        1e-30,
        1e-20,
        sys.float_info.epsilon / 2,
        sys.float_info.epsilon,
        sys.float_info.epsilon * 2,
        1e-12,
        1e-8,
        1.0,
        -sys.float_info.epsilon / 2,
        -sys.float_info.epsilon,
        -sys.float_info.epsilon * 2,
        -1.0,
    ],
)
def test_matrix_inversion_rejects_the_same_determinants(determinant: float) -> None:
    """`util_invert_matrix` refuses anything within `DBL_EPSILON` and leaves the matrix alone."""
    scale = abs(determinant) ** 0.5 or 1.0
    coefficients = (scale, 0.0, 0.0, determinant / scale, 3.0, -4.0)

    def snapshot(module: Any) -> tuple[Any, ...]:
        matrix = module.Matrix(coefficients)
        status = matrix.invert()
        return (status, tuple(matrix), tuple(~module.Matrix(coefficients)))

    assert snapshot(compat_pymupdf) == snapshot(real_pymupdf)


@pytest.mark.parametrize("shear", [0.0, 1e-9, 1e-8, 1e-6, 9.9e-6, 1e-5, 1.1e-5, 1e-4, 1.0])
def test_is_rectilinear_shares_the_pymupdf_epsilon(shear: float) -> None:
    """PyMuPDF compares against `EPSILON = 1e-5`, on both the shear and the scale pair."""

    def snapshot(module: Any) -> tuple[bool, ...]:
        return (
            module.Matrix(1.0, shear, shear, 1.0, 0.0, 0.0).is_rectilinear,
            module.Matrix(shear, 1.0, 1.0, shear, 0.0, 0.0).is_rectilinear,
            module.Matrix(shear, shear, shear, shear, 0.0, 0.0).is_rectilinear,
        )

    assert snapshot(compat_pymupdf) == snapshot(real_pymupdf)

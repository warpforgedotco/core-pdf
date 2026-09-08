from pathlib import Path
from typing import Any

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .support import FIXTURES_ROOT

real_pymupdf = pytest.importorskip("pymupdf")
pytestmark = pytest.mark.compat_differential

internal_PDFS = (
    FIXTURES_ROOT / "PyMuPDF/tests/resources/small-table.pdf",
    FIXTURES_ROOT / "PyMuPDF/tests/resources/test_2957_1.pdf",
)


@pytest.mark.parametrize("pdf_path", internal_PDFS, ids=lambda path: path.name)
def test_page_geometry_matrix_operations(pdf_path: Path) -> None:
    """Compare transforms derived from the same PDF page in both engines."""
    with real_pymupdf.open(pdf_path) as expected, compat_pymupdf.open(pdf_path) as actual:
        assert actual.page_count == expected.page_count
        expected_box = tuple(expected[0].rect)
        actual_box = tuple(actual[0].rect)
        assert actual_box == expected_box
        width, height = expected_box[2:]
        constructors = (
            (),
            (expected[0].rotation,),
            (37,),
            (width / height, height / width),
            (2, 3, 0),
            (2, 3, 1),
            ((1, 2, 3, 4, width, height),),
            (1, 2, 3, 4, width, height),
        )
        for args in constructors:
            reference = real_pymupdf.Matrix(*args)
            facade = compat_pymupdf.Matrix(*args)
            assert tuple(facade) == tuple(reference)
            assert repr(facade) == repr(reference)
            assert bool(facade) == bool(reference)
            assert abs(facade) == abs(reference)
            assert facade.is_rectilinear == reference.is_rectilinear
            assert tuple(~facade) == pytest.approx(tuple(~reference), rel=1e-6, abs=1e-6)
            for method, arguments in (
                ("prescale", (width / height, 2)),
                ("preshear", (0.25, 0.5)),
                ("pretranslate", (width, -height)),
                ("prerotate", (90,)),
                ("prerotate", (37,)),
            ):
                left = compat_pymupdf.Matrix(facade)
                right = real_pymupdf.Matrix(reference)
                assert getattr(left, method)(*arguments) is left
                assert getattr(right, method)(*arguments) is right
                assert tuple(left) == pytest.approx(tuple(right), rel=1e-12, abs=1e-12)
            for scalar in (2, -0.5):
                assert tuple(facade * scalar) == tuple(reference * scalar)
                assert tuple(facade / scalar) == tuple(reference / scalar)
                assert tuple(facade + scalar) == tuple(reference + scalar)
                assert tuple(facade - scalar) == tuple(reference - scalar)
            other = (0.5, 0.25, -0.5, 2, width, -height)
            assert tuple(facade * other) == tuple(reference * other)
            assert tuple(facade / other) == pytest.approx(
                tuple(reference / other), rel=1e-6, abs=1e-6
            )
            assert facade.invert() == reference.invert()
            assert tuple(facade) == pytest.approx(tuple(reference), rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("pdf_path", internal_PDFS, ids=lambda path: path.name)
def test_page_matrix_sequence_and_singular_inversion(pdf_path: Path) -> None:
    def snapshot(module: Any) -> tuple[Any, ...]:
        with module.open(pdf_path) as document:
            _, _, width, height = document[0].rect
            matrix = module.Matrix(1, 2, 2, 4, width, height)
            status = matrix.invert()
            singular = tuple(matrix)
            inverse = tuple(~matrix)
            with pytest.raises(IndexError):
                matrix[-1] = width
            matrix[5] = width
            matrix[0] = height
            return (
                status,
                singular,
                inverse,
                tuple(matrix),
                matrix[1:4],
                matrix == tuple(matrix),
                tuple(+matrix),
                tuple(-matrix),
                len(matrix),
            )

    assert snapshot(compat_pymupdf) == snapshot(real_pymupdf)


@pytest.mark.parametrize("pdf_path", internal_PDFS, ids=lambda path: path.name)
def test_page_pixmap_scale_matrices(pdf_path: Path) -> None:
    """Uniform scales must match PyMuPDF; the facade honours nothing else, so it refuses."""
    supported = (None, (1, 1), (2, 2), (0.5, 0.5))
    # `matrix.a` is the only coefficient the facade can honour, so anything carrying rotation,
    # shear, translation, a non-uniform scale or a non-positive scale has to raise rather than
    # silently render an unrotated, unshifted, minimum-scale page.
    unsupported = ((90,), (-2, -2), (0, 0), (1, 2), ((1, 1, 0, 1, 0, 0),), ((1, 0, 0, 1, 5, 7),))

    def snapshot(module: Any) -> list[Any]:
        output: list[Any] = []
        with module.open(pdf_path) as document:
            page = document[0]
            for args in supported:
                matrix = module.Matrix(*args) if args is not None else None
                pixmap = page.get_pixmap(matrix=matrix)
                output.append((args, pixmap.width, pixmap.height))
            for args in unsupported:
                try:
                    page.get_pixmap(matrix=module.Matrix(*args))
                except ValueError:
                    output.append((args, "ValueError"))
                else:
                    output.append((args, "rendered"))
        return output

    actual = snapshot(compat_pymupdf)
    assert [entry for entry in actual if entry[0] in supported] == [
        entry for entry in snapshot(real_pymupdf) if entry[0] in supported
    ]
    assert [entry for entry in actual if entry[0] in unsupported] == [
        (args, "ValueError") for args in unsupported
    ]

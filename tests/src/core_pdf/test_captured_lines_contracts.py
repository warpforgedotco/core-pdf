# SPDX-License-Identifier: AGPL-3.0-only

"""Stroke lines as arrays, and flattened paths whose points wait.

A painted path's stroke lines are rows of a CapturedLines table rather than
CapturedLine objects, and its point lists are built only when something reads
them. These pin what the rest of the code relies on: the table still reads as
a sequence of CapturedLine, programs cut and merge it correctly, and a deferred
path answers exactly as the path built eagerly would.
"""

import math
import pickle

import pytest

from core_pdf.impl.capture.program import CapturedProgram, PageProgram
from core_pdf.impl.capture.recording import StrokeLineRows, flatten_path
from core_pdf.impl.capture.records import (
    EMPTY_LINES,
    CapturedLine,
    CapturedLines,
    CapturedPath,
    CapturedSubpath,
)
from core_pdf_spec.s_07_content.model import PdfPath
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix


def path_of(*commands: tuple[str, tuple[float, ...]]) -> PdfPath:
    path = PdfPath()
    for operator, values in commands:
        match operator:
            case "m":
                path.move_to(*values)
            case "l":
                path.line_to(*values)
            case "h":
                path.close()
            case "re":
                path.rect(*values)
            case "c":
                path.cubic_to(values, IDENTITY_MATRIX, 0.0)
            case _:
                raise ValueError(operator)
    return path


def eager(path: CapturedPath) -> CapturedPath:
    """The same path with its subpaths built, and the deferral gone."""
    return CapturedPath([CapturedSubpath(list(s.points), closed=s.closed) for s in path.subpaths])


def test_lines_read_as_a_sequence_of_captured_lines() -> None:
    lines = CapturedLines([CapturedLine(0, 1, 2, 3, 0.5), CapturedLine(4, 5, 6, 7)])
    assert len(lines) == 2
    first, second = lines
    assert (first.x0, first.y0, first.x1, first.y1, first.line_width) == (0, 1, 2, 3, 0.5)
    assert second.line_width == 1.0
    assert lines[1].x1 == 6.0
    assert isinstance(lines[1:], CapturedLines)
    assert len(lines[1:]) == 1
    x0, y0, x1, y1 = lines.columns()
    assert x0.tolist() == [0.0, 4.0]
    assert y1.tolist() == [3.0, 7.0]
    assert not EMPTY_LINES
    assert EMPTY_LINES.columns()[0].shape == (0,)


def test_lines_compare_and_hash_by_value_and_survive_pickling() -> None:
    left = CapturedLines([CapturedLine(0, 0, 1, 1)])
    right = CapturedLines([CapturedLine(0, 0, 1, 1)])
    assert left == right
    assert hash(left) == hash(right)
    assert left != CapturedLines([CapturedLine(0, 0, 1, 2)])
    assert pickle.loads(pickle.dumps(left)) == left
    with pytest.raises(ValueError, match="read-only"):
        left.array[0, 0] = 9.0


def test_programs_accept_lines_built_by_hand_and_merge_them() -> None:
    body = CapturedProgram(lines=CapturedLines([CapturedLine(0, 0, 1, 1)]))
    assert type(body.lines) is CapturedLines
    # A plain list, as a hand-built program may pass; it is converted.
    by_hand = CapturedProgram(
        lines=[CapturedLine(2, 2, 3, 3)]  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    )
    assert type(by_hand.lines) is CapturedLines
    appearance = type("Appearance", (), {"program": by_hand})()
    page = PageProgram(
        body=body,
        appearances=(appearance,),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    )
    assert [line.x0 for line in page.lines] == [0.0, 2.0]


@pytest.mark.parametrize("mark", [0, 1, 2, 3, 5, 6, 7])
def test_line_rows_cut_at_any_mark(mark: int) -> None:
    rows = StrokeLineRows()
    flatten_path(path_of(("m", (0.0, 0.0)), ("l", (1.0, 1.0)), ("l", (2.0, 2.0))), None, rows, 0.5)
    flatten_path(path_of(("m", (9.0, 9.0))), None, rows, 9.0)
    flatten_path(
        path_of(("m", (2.0, 2.0)), ("l", (3.0, 3.0)), ("l", (4.0, 4.0)), ("l", (5.0, 5.0))),
        None,
        rows,
        2.0,
    )
    flatten_path(path_of(("m", (5.0, 5.0)), ("l", (6.0, 6.0))), None, rows, 3.0)
    assert rows.count == 6
    cut = rows.since(mark)
    assert [line.x0 for line in cut] == [float(x) for x in range(mark, 6)]
    assert [line.line_width for line in cut] == [0.5, 0.5, 2.0, 2.0, 2.0, 3.0][mark:]


def test_flattened_path_answers_box_and_segments_without_building_points() -> None:
    source = path_of(("m", (0.0, 0.0)), ("l", (10.0, 5.0)), ("h", ()), ("m", (20.0, -3.0)))
    rows = StrokeLineRows()
    path = flatten_path(source, None, rows)
    lines = rows.since(0).array[:, :4]
    assert path.bbox() == (0.0, -3.0, 20.0, 5.0)
    assert path.has_segments()
    # Neither answer built the subpaths: the slot is still unset.
    assert path._deferred is not None
    assert lines.tolist() == [[0.0, 0.0, 10.0, 5.0]]
    assert [(s.points, s.closed) for s in path.subpaths] == [
        ([(0.0, 0.0), (10.0, 5.0)], True),
        ([(20.0, -3.0)], False),
    ]
    assert path._deferred is None
    assert path.bbox() == eager(path).bbox()


def test_a_rectangle_with_a_closing_point_is_still_a_rectangle() -> None:
    # Glyph outlines arrive with the closing duplicate dropped, and their
    # deferred check relies on that; a flattened path does not, so the check
    # must not apply to it.
    source = path_of(
        ("m", (0.0, 0.0)),
        ("l", (10.0, 0.0)),
        ("l", (10.0, 4.0)),
        ("l", (0.0, 4.0)),
        ("l", (0.0, 0.0)),
        ("h", ()),
    )
    path = flatten_path(source, None)
    assert path.axis_aligned_rect() == (0.0, 0.0, 10.0, 4.0)


def test_a_deferred_path_can_still_be_extended() -> None:
    path = flatten_path(path_of(("m", (0.0, 0.0)), ("l", (1.0, 1.0))), None)
    path.line_to(2.0, 0.0)
    assert path.bbox() == (0.0, 0.0, 2.0, 1.0)
    assert path.fill_edges()[-1] == (2.0, 0.0, 0.0, 0.0)


def test_transform_matches_the_eager_transform() -> None:
    source = path_of(
        ("m", (1.0, 2.0)),
        ("c", (1.0, 2.0, 5.0, 9.0, 9.0, 9.0, 12.0, 2.0)),
        ("re", (3.0, 3.0, 4.0, 5.0)),
    )
    matrix = Matrix(0.8, 0.6, -0.6, 0.8, 30.0, 700.0)
    untransformed = flatten_path(source, None)
    rows = StrokeLineRows()
    transformed = flatten_path(source, matrix, rows)
    lines = rows.since(0).array
    expected = eager(untransformed).transformed(matrix)
    assert [s.points for s in transformed.subpaths] == [s.points for s in expected.subpaths]
    assert transformed.bbox() == expected.bbox()
    assert math.isclose(lines[0, 0], expected.subpaths[0].points[0][0])

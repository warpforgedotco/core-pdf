# SPDX-License-Identifier: AGPL-3.0-only
"""Validated capture records keep their geometry and normalization ownership."""

import pytest

from core_pdf.impl.extract.capture import internal_normalized_tokens
from core_pdf.impl.extract.contracts import ParsedBlock, ParsedLine
from core_pdf.impl.extract.emit import internal_line_decoration_bbox, internal_normalized_blocks
from core_pdf.impl.model.geometry import RectBox
from core_pdf.impl.records import TextWord
from core_pdf.impl.spec.s_07_content.capture import CapturedDrawing, CapturedPath
from tests.helpers.extract_fakes import text_run


class CountingPath(CapturedPath):
    def __init__(self, bounds: tuple[float, float, float, float] | None) -> None:
        super().__init__()
        self.bounds = bounds
        self.calls = 0

    def bbox(self) -> tuple[float, float, float, float] | None:
        self.calls += 1
        return self.bounds


@pytest.mark.parametrize("bounds", [None, (1.0, 2.0, 3.0, 4.0)])
def test_decoration_bounds_evaluate_optional_path_once(
    bounds: tuple[float, float, float, float] | None,
) -> None:
    path = CountingPath(bounds)
    drawing = CapturedDrawing(0, None, None, path=path)

    assert internal_line_decoration_bbox(drawing) == bounds
    assert path.calls == 1


def test_decoration_bounds_preserve_explicit_inverted_box_precedence() -> None:
    path = CountingPath((1.0, 2.0, 3.0, 4.0))
    drawing = CapturedDrawing(0, None, None, bbox=RectBox(9.0, 8.0, 2.0, 1.0), path=path)

    assert internal_line_decoration_bbox(drawing) == (9.0, 8.0, 2.0, 1.0)
    assert path.calls == 0


def test_decoration_without_geometry_is_absent() -> None:
    assert internal_line_decoration_bbox(CapturedDrawing(0, None, None)) is None


def test_token_normalization_accepts_an_iterable_of_validated_runs() -> None:
    runs = (text_run("Hello, STRAẞE!"), text_run("123"))

    assert internal_normalized_tokens(iter(runs)) == ("hello", "strasse", "123")


def test_decoration_projects_directly_without_rebuilding_parsed_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bbox = (0.0, 0.0, 20.0, 10.0)
    word = TextWord("Hello", bbox=bbox)
    parsed = ParsedBlock((ParsedLine("Hello", bbox, "native", words=(word,)),), bbox)
    underline = CapturedDrawing(0, None, None, bbox=RectBox(0, -1, 20, 0), kind="stroke")

    def unexpected_reconciliation(*args: object) -> tuple[TextWord, ...]:
        raise AssertionError("parsed lines must not be rebuilt during output projection")

    monkeypatch.setattr(
        "core_pdf.impl.extract.contracts.internal_reconcile_text_words", unexpected_reconciliation
    )
    result = internal_normalized_blocks((parsed,), (underline,))

    assert result[0].lines[0].text == "Hello"
    assert result[0].lines[0].underline
    assert result[0].lines[0].words == (word,)
    assert result[0].lines[0].words[0] is word

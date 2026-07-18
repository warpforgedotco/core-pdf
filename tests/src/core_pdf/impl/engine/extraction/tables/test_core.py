from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
from core_layout.impl.layout.models import TextRun

from core_pdf.impl.engine.extraction.tables.api import PageTableMixin
from core_pdf.impl.engine.extraction.tables.core import PageTableCoreMixin


class _TablePage(PageTableMixin):
    rotation = 0
    width = 200.0
    height = 200.0
    grid_lines = None

    def __init__(self) -> None:
        self.tables = {}
        self._chars = [
            _text_run("A", 10.0, 140.0, 30.0, 150.0, 0),
            _text_run("B", 80.0, 140.0, 100.0, 150.0, 1),
            _text_run("C", 10.0, 100.0, 30.0, 110.0, 2),
            _text_run("D", 80.0, 100.0, 100.0, 110.0, 3),
        ]

    @property
    def chars(self) -> list[TextRun]:
        return self._chars

    def get_grid_lines(self) -> list[Any]:
        return []

    def get_text_spans(self) -> list[Any]:
        return []

    def crop(self, bbox: tuple[float, float, float, float]) -> _TablePage:
        del bbox
        return self


def test_canonical_table_extraction_runs_each_strategy_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = PageTableCoreMixin.extract_table_strategy
    calls: Counter[tuple[str, bool]] = Counter()

    def counted_extract_table_strategy(cls: Any, context: Any, options: Any) -> Any:
        calls[(options.flavor, options.canonicalize)] += 1
        return original(context, options)

    monkeypatch.setattr(
        PageTableCoreMixin,
        "extract_table_strategy",
        classmethod(counted_extract_table_strategy),
    )

    _TablePage().table_extraction_payload(flavor="hybrid")

    assert calls[("hybrid", True)] == 0
    assert calls[("hybrid", False)] == 1
    assert calls[("lattice", False)] == 1
    assert calls[("stream", False)] == 1
    assert calls[("network", False)] == 1
    assert calls[("auto", False)] == 1
    assert sum(calls.values()) == 5


def _text_run(
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    order: int,
) -> TextRun:
    return TextRun(
        text,
        x0,
        y0,
        x1,
        y1,
        x0,
        y0,
        10.0,
        4.0,
        order,
        order,
        0,
        seqno=order,
    )

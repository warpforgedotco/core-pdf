from io import BytesIO
from typing import Any

import pytest

from core_pdf.api.compat import pdfplumber as compat

reference = pytest.importorskip("pdfplumber")
pytestmark = pytest.mark.compat_differential


@pytest.fixture
def blank_page_pdf(text_pdf_bytes: bytes) -> bytes:
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    page = writer.add_page(PdfReader(BytesIO(text_pdf_bytes)).pages[0])
    page.replace_contents(None)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def table_snapshot(table: Any) -> dict[str, Any]:
    return {
        "bbox": tuple(table.bbox),
        "cells": sorted(table.cells),
        "rows": [row.cells for row in table.rows],
        "columns": [column.cells for column in table.columns],
        "text": table.extract(),
    }


@pytest.mark.parametrize("content", ["blank", "prose"])
def test_explicit_grid_has_one_row_per_horizontal_interval(
    blank_page_pdf: bytes,
    text_pdf_bytes: bytes,
    content: str,
) -> None:
    settings = {
        "vertical_strategy": "explicit",
        "horizontal_strategy": "explicit",
        "explicit_vertical_lines": [20, 100, 180],
        "explicit_horizontal_lines": [20, 100, 180],
    }
    source = blank_page_pdf if content == "blank" else text_pdf_bytes
    snapshots = []
    for library in (reference, compat):
        with library.open(BytesIO(source)) as pdf:
            page = pdf.pages[0]
            tables = page.find_tables(settings)
            assert len(tables) == 1
            snapshots.append(table_snapshot(tables[0]))
            assert page.extract_tables(settings) == [tables[0].extract()]
            assert page.extract_table(settings) == tables[0].extract()
            finder = page.debug_tablefinder(settings)
            assert len(finder.edges) == 6
            assert len(finder.intersections) == 9
            assert sorted(finder.cells) == sorted(tables[0].cells)
    assert snapshots[1] == snapshots[0]


@pytest.mark.parametrize("content", ["blank", "prose"])
def test_default_line_strategy_does_not_invent_tables_from_prose(
    text_pdf_bytes: bytes,
    blank_page_pdf: bytes,
    content: str,
) -> None:
    source = blank_page_pdf if content == "blank" else text_pdf_bytes
    for library in (reference, compat):
        with library.open(BytesIO(source)) as pdf:
            page = pdf.pages[0]
            assert page.find_tables() == []
            assert page.find_table() is None
            assert page.extract_table() is None
            assert page.extract_tables() == []


@pytest.mark.parametrize(
    "settings",
    [
        None,
        {},
        {"snap_tolerance": 7, "join_tolerance": 2},
        {"intersection_tolerance": 5, "snap_x_tolerance": 1},
        {"text_tolerance": 4},
        {"text_x_tolerance": 1},
        {"vertical_strategy": "text", "horizontal_strategy": "text"},
    ],
)
def test_table_settings_resolve_defaults_and_axis_fallbacks(settings: Any) -> None:
    from pdfplumber.table import TableSettings as ReferenceSettings

    from core_pdf.api.compat.pdfplumber.table import TableSettings

    actual = TableSettings.resolve(settings)
    expected = ReferenceSettings.resolve(settings)
    assert actual.__dict__ == expected.__dict__
    assert TableSettings.resolve(actual) is actual


@pytest.mark.parametrize(
    "name",
    [
        "snap_tolerance",
        "join_y_tolerance",
        "intersection_x_tolerance",
        "edge_min_length",
        "edge_min_length_prefilter",
        "min_words_vertical",
        "min_words_horizontal",
    ],
)
def test_table_settings_reject_negative_measurements(name: str) -> None:
    from pdfplumber.table import TableSettings as ReferenceSettings

    for settings_class in (ReferenceSettings, compat.TableSettings):
        with pytest.raises(ValueError):
            settings_class.resolve({name: -1})


@pytest.mark.parametrize("settings", ["lines", 1, []])
def test_table_settings_reject_invalid_shapes(settings: Any) -> None:
    from pdfplumber.table import TableSettings as ReferenceSettings

    for settings_class in (ReferenceSettings, compat.TableSettings):
        with pytest.raises(ValueError):
            settings_class.resolve(settings)


@pytest.mark.parametrize(
    "settings",
    [
        {"vertical_strategy": "explicit"},
        {"vertical_strategy": "explicit", "explicit_vertical_lines": [20]},
        {
            "vertical_strategy": "explicit",
            "horizontal_strategy": "explicit",
            "explicit_vertical_lines": [20, 180],
        },
    ],
)
def test_explicit_table_strategy_validates_both_axes_at_finding_time(
    blank_page_pdf: bytes,
    settings: dict[str, Any],
) -> None:
    from pdfplumber.table import TableSettings as ReferenceSettings

    assert ReferenceSettings.resolve(settings).vertical_strategy == "explicit"
    assert compat.TableSettings.resolve(settings).vertical_strategy == "explicit"
    for library in (reference, compat):
        with library.open(BytesIO(blank_page_pdf)) as pdf:
            missing_lines = any(
                settings.get(f"{axis}_strategy") == "explicit"
                and settings.get(f"explicit_{axis}_lines") is None
                for axis in ("vertical", "horizontal")
            )
            error_type = TypeError if library is reference and missing_lines else ValueError
            for method in (pdf.pages[0].find_tables, pdf.pages[0].debug_tablefinder):
                with pytest.raises(error_type):
                    method(settings)


@pytest.mark.parametrize("settings", [{"strategy": "lines"}, {"unknown": True}])
def test_unknown_table_settings_are_rejected_as_arguments(settings: dict[str, Any]) -> None:
    from pdfplumber.table import TableSettings as ReferenceSettings

    for settings_class in (ReferenceSettings, compat.TableSettings):
        with pytest.raises(TypeError):
            settings_class.resolve(settings)


@pytest.mark.parametrize("join", [0, 1, 3])
def test_tablefinder_merges_fragments_before_minimum_length_filter(blank_page_pdf, join):
    from copy import deepcopy

    lines = [
        {
            "object_type": "line",
            "x0": start,
            "x1": end,
            "top": 10,
            "bottom": 10,
            "width": end - start,
            "height": 0,
            "doctop": 10,
            "y0": 190,
            "y1": 190,
        }
        for start, end in [(10, 13), (14, 17), (18, 21)]
    ]
    snapshots = []
    for library in (reference, compat):
        with library.open(BytesIO(blank_page_pdf)) as pdf:
            page = pdf.pages[0]
            page._objects = {"line": deepcopy(lines)}
            finder = page.debug_tablefinder(
                {
                    "snap_tolerance": 0,
                    "join_tolerance": join,
                    "edge_min_length": 6,
                    "edge_min_length_prefilter": 1,
                }
            )
            snapshots.append(finder.edges)
            assert page.lines == lines
    assert snapshots[0] == snapshots[1]

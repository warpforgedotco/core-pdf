"""CSV output preserves scalar values and never mutates cached page objects."""

import csv
from copy import deepcopy
from io import BytesIO, StringIO
from typing import Any

import pytest

from core_pdf.api.compat import pdfplumber as compat


@pytest.mark.parametrize("precision", [None, 0, 2])
@pytest.mark.parametrize("object_types", [None, ["char"], []])
def test_csv_quotes_text_and_selects_objects(text_pdf_bytes, precision, object_types):
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        objects: dict[str, list[dict[str, Any]]] = {
            "char": [{"object_type": "char", "text": 'a,"b"\nc', "x0": 1.2345}],
            "rect": [{"object_type": "rect", "x0": 9.8765, "metadata": {"nested": 1}}],
        }
        pdf.pages[0]._objects = objects
        original = deepcopy(objects)
        reader = csv.DictReader(StringIO(pdf.to_csv(object_types, precision=precision)))
        actual = list(reader)
        expected = []
        for kind, rows in objects.items():
            if object_types is not None and kind not in object_types:
                continue
            for row in rows:
                number = row["x0"] if precision is None else round(row["x0"], precision)
                expected.append(
                    {"object_type": kind, "text": row.get("text", ""), "x0": str(number)}
                )
        assert actual == expected
        assert objects == original
        assert not reader.fieldnames or "metadata" not in reader.fieldnames


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"include_attrs": ["text"]}, {"object_type": "char", "text": "Hello"}),
        ({"exclude_attrs": ["x0"]}, {"object_type": "char", "text": "Hello"}),
        ({"include_attrs": [], "exclude_attrs": []}, {"object_type": "char"}),
        (
            {"include_attrs": ["text", "x0"], "exclude_attrs": ["x0"]},
            {"object_type": "char", "text": "Hello"},
        ),
    ],
)
def test_csv_attribute_selection_preserves_source(text_pdf_bytes, options, expected):
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        obj = {"object_type": "char", "text": "Hello", "x0": 1.25}
        original = obj.copy()
        pdf.pages[0]._objects = {"char": [obj]}
        assert list(csv.DictReader(StringIO(pdf.to_csv(**options)))) == [expected]
        assert obj == original

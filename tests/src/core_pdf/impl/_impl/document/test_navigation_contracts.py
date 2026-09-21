from typing import Any

import pytest

from core_pdf.impl._impl.document.document import PdfDocument
from core_pdf.impl.types import PdfName


@pytest.mark.parametrize(
    ("style", "start", "expected"),
    [
        ("D", 3, ["P3", "P4", "P5"]),
        ("r", 4, ["Piv", "Pv", "Pvi"]),
        ("R", 9, ["PIX", "PX", "PXI"]),
        ("a", 26, ["Pz", "Paa", "Pbb"]),
        ("A", 26, ["PZ", "PAA", "PBB"]),
        (None, 1, ["P", "P", "P"]),
        ("unknown", 1, ["P", "P", "P"]),
        ("D", 0, ["P1", "P2", "P3"]),
        ("D", True, ["P1", "P2", "P3"]),
    ],
)
def test_page_label_styles_and_reader_defaults(text_pdf_bytes, style, start, expected):
    with PdfDocument(text_pdf_bytes) as document:
        spec: dict[str, Any] = {"P": b"P", "St": start}
        if style is not None:
            spec["S"] = PdfName(style.encode())
        document.catalog()["PageLabels"] = {"Nums": [0, spec]}
        assert document.build_page_labels(page_count=3) == expected
        assert document.page_label(-1) is None
        assert document.page_label(1) is None
        assert document.page_label(0) == expected[0]


@pytest.mark.parametrize("recover", [False, True])
def test_page_labels_missing_initial_range(text_pdf_bytes, recover):
    with PdfDocument(text_pdf_bytes) as document:
        document.catalog()["PageLabels"] = {"Nums": [2, {"S": PdfName(b"D")}]}
        document.xref_was_recovered = recover
        if recover:
            assert document.build_page_labels(page_count=4) == ["", "", "1", "2"]
        else:
            with pytest.raises(ValueError, match="missing page index 0"):
                document.build_page_labels(page_count=4)


def test_page_labels_switch_ranges_and_ignore_non_dictionary_specs(text_pdf_bytes):
    with PdfDocument(text_pdf_bytes) as document:
        document.catalog()["PageLabels"] = {
            "Nums": [0, {"S": PdfName(b"D")}, 1, None, 2, {"P": b"Annex"}]
        }
        assert document.build_page_labels(page_count=4) == ["1", "2", "Annex", "Annex"]


@pytest.mark.parametrize("labels", [None, {}, {"Nums": []}, {"Nums": [0, None]}])
def test_absent_or_empty_page_labels_have_no_projection(text_pdf_bytes, labels):
    with PdfDocument(text_pdf_bytes) as document:
        document.catalog()["PageLabels"] = labels
        assert document.page_labels is None
        assert document.page_label(0) is None


@pytest.mark.parametrize("value", [1, b"invalid", []])
def test_invalid_page_label_roots_are_rejected(text_pdf_bytes, value):
    with PdfDocument(text_pdf_bytes) as document:
        document.catalog()["PageLabels"] = value
        with pytest.raises(ValueError, match="number tree"):
            _ = document.page_labels


@pytest.mark.parametrize("wrapper_count", [0, 1, 4, 1500])
def test_destination_wrappers_preserve_the_original_array(text_pdf_bytes, wrapper_count):
    with PdfDocument(text_pdf_bytes) as document:
        destination = [document.build_page_dicts()[0], PdfName(b"XYZ"), 10, None, 2]
        value: Any = destination
        for _ in range(wrapper_count):
            value = {"D": value}
        result = document.normalize_destination_value(value)
        assert result.page_index == 0
        assert result.type == "XYZ"
        assert result.args == [10, None, 2]
        assert result.raw is destination


@pytest.mark.parametrize("cycle_size", [1, 2, 5])
def test_cyclic_destination_wrappers_are_rejected(text_pdf_bytes, cycle_size):
    with PdfDocument(text_pdf_bytes) as document:
        root: dict[str, Any] = {}
        tail = root
        for _ in range(cycle_size - 1):
            child: dict[str, Any] = {}
            tail["D"] = child
            tail = child
        tail["D"] = root
        with pytest.raises(ValueError, match="cyclic destination"):
            document.normalize_destination_value(root)


def test_named_destinations_share_aliases_and_isolate_bad_entries(text_pdf_bytes):
    with PdfDocument(text_pdf_bytes) as document:
        target = [document.build_page_dicts()[0], PdfName(b"Fit")]
        document.catalog()["Names"] = {
            "Dests": {"Names": [b"alias", b"target", b"broken", [], b"target", target]}
        }
        results = document.named_destinations()
        assert results["alias"] is results["target"]
        assert results["target"].raw is target
        assert results["target"].page_index == 0
        assert results["broken"].page_index is None
        assert results["broken"].raw == "broken"


@pytest.mark.parametrize("value", [[], {}, {"D": None}, [None], [0, 1], 1])
def test_invalid_destination_values_raise_a_consistent_error(text_pdf_bytes, value):
    with PdfDocument(text_pdf_bytes) as document:
        with pytest.raises(ValueError, match="invalid destination"):
            document.normalize_destination_value(value)


@pytest.mark.parametrize("tuple_input", [False, True])
@pytest.mark.parametrize("kind", [None, PdfName(b"Fit"), b"XYZ"])
def test_destination_arrays_accept_optional_type_and_tuple_input(text_pdf_bytes, tuple_input, kind):
    with PdfDocument(text_pdf_bytes) as document:
        values = [document.build_page_dicts()[0]]
        if kind is not None:
            values.append(kind)
        result = document.normalize_destination_value(tuple(values) if tuple_input else values)
        assert result.page_index == 0
        assert result.type == (
            None if kind is None else "Fit" if isinstance(kind, PdfName) else "XYZ"
        )
        assert result.args == []


@pytest.mark.parametrize("target", [b"missing", b"alias"])
def test_dangling_or_cyclic_named_aliases_remain_unresolved(text_pdf_bytes, target):
    with PdfDocument(text_pdf_bytes) as document:
        document.catalog()["Names"] = {"Dests": {"Names": [b"alias", target]}}
        results = document.named_destinations()
        assert results["alias"].page_index is None
        assert document.resolve_named_destination("absent") is None

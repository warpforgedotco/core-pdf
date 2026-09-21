from copy import deepcopy
from io import BytesIO
from typing import Any

import pytest

from core_pdf.api.compat import pdfplumber as compat

reference = pytest.importorskip("pdfplumber")
pytestmark = pytest.mark.compat_differential


def character_snapshot(page: Any) -> tuple[str, list[tuple[float, ...]]]:
    return (
        "".join(char["text"] for char in page.chars),
        [
            tuple(char[key] for key in ("x0", "top", "x1", "bottom", "width", "height"))
            for char in page.chars
        ],
    )


@pytest.mark.parametrize("method", ["crop", "within_bbox", "outside_bbox"])
@pytest.mark.parametrize("bbox", [(0, 0, 200, 200), (22, 88, 70, 106), (0, 0, 10, 10)])
def test_page_bbox_operations_match_reference(
    text_pdf_bytes: bytes,
    method: str,
    bbox: tuple[int, int, int, int],
) -> None:
    with (
        compat.open(BytesIO(text_pdf_bytes)) as actual,
        reference.open(BytesIO(text_pdf_bytes)) as expected,
    ):
        native_chars = deepcopy(actual.pages[0].chars)
        actual_page = getattr(actual.pages[0], method)(bbox)
        expected_page = getattr(expected.pages[0], method)(bbox)
        actual_text, actual_boxes = character_snapshot(actual_page)
        expected_text, expected_boxes = character_snapshot(expected_page)
        assert actual_text == expected_text
        for a, e in zip(actual_boxes, expected_boxes, strict=True):
            assert a == pytest.approx(e)
        assert actual_page.bbox == expected_page.bbox
        assert actual_page.is_original == expected_page.is_original
        assert actual_page.objects is actual_page.objects
        assert actual.pages[0].chars == native_chars


@pytest.mark.parametrize("tolerance", [0, 0.5, 1])
@pytest.mark.parametrize("extra_attrs", [(), ("fontname",), ("size",)])
def test_deduplication_keeps_distinct_attributes_and_nontext_objects(
    text_pdf_bytes: bytes,
    tolerance: float,
    extra_attrs: tuple[str, ...],
) -> None:
    snapshots = []
    for library in (reference, compat):
        with library.open(BytesIO(text_pdf_bytes)) as pdf:
            page = pdf.pages[0]
            first = page.chars[0]
            chars = [
                first,
                dict(first),
                {**first, "x0": first["x0"] + 0.75},
                {**first, "fontname": "Different"},
                {**first, "size": 40},
                {**first, "top": first["top"] + 0.75, "doctop": first["doctop"] + 0.75},
            ]
            page._objects = {"char": chars, "rect": [{"object_type": "rect"}]}
            result = page.dedupe_chars(tolerance=tolerance, extra_attrs=extra_attrs)
            assert result.rects == [{"object_type": "rect"}]
            assert result.objects is result.objects
            assert len(page.chars) == 6
            snapshots.append(
                [(c["text"], c["x0"], c["top"], c["fontname"], c["size"]) for c in result.chars]
            )
    assert snapshots[1] == snapshots[0]


@pytest.mark.parametrize("document_level", [False, True])
@pytest.mark.parametrize("filters", [{"include_attrs": ["text"]}, {"exclude_attrs": ["fontname"]}])
def test_serializing_filtered_attributes_does_not_mutate_cached_page_objects(
    text_pdf_bytes: bytes,
    document_level: bool,
    filters: dict[str, list[str]],
) -> None:
    for library in (reference, compat):
        with library.open(BytesIO(text_pdf_bytes)) as pdf:
            page = pdf.pages[0]
            before = deepcopy(page.chars)
            target = pdf if document_level else page
            serialized = target.to_json(**filters)
            assert "Hello" not in serialized
            assert page.chars == before
            assert page.extract_text() == "Hello maintenance"


def test_nested_relative_crop_respects_parent_selection(text_pdf_bytes: bytes) -> None:
    snapshots = []
    for library in (reference, compat):
        with library.open(BytesIO(text_pdf_bytes)) as pdf:
            page = pdf.pages[0].crop((20, 80, 100, 120))
            cropped = page.crop((2, 5, 30, 25), relative=True)
            snapshots.append(character_snapshot(cropped))
            assert cropped.bbox == (22, 85, 50, 105)
    assert snapshots[0][0] == snapshots[1][0]
    for a, e in zip(snapshots[1][1], snapshots[0][1], strict=True):
        assert a == pytest.approx(e)


@pytest.mark.parametrize("variant", ["chain", "font", "size", "upright", "same-object"])
def test_deduplication_defaults_and_transitive_position_clusters(
    text_pdf_bytes: bytes,
    variant: str,
) -> None:
    snapshots = []
    for library in (reference, compat):
        with library.open(BytesIO(text_pdf_bytes)) as pdf:
            page = pdf.pages[0]
            char = page.chars[0]
            if variant == "chain":
                chars = [
                    {**char, "x0": char["x0"] + offset, "x1": char["x1"] + offset}
                    for offset in (0, 0.75, 1.5)
                ]
            elif variant == "same-object":
                chars = [char, char]
            else:
                key, value = {
                    "font": ("fontname", "Different"),
                    "size": ("size", 40),
                    "upright": ("upright", False),
                }[variant]
                chars = [char, {**char, key: value}]
            page._objects = {"char": chars}
            snapshots.append(character_snapshot(page.dedupe_chars()))
    assert snapshots[0] == snapshots[1]

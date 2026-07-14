# SPDX-License-Identifier: AGPL-3.0-only
"""Native unstructured layout records backed by core-pdf."""

from __future__ import annotations

from collections.abc import Container, Iterator
from dataclasses import dataclass
from os import PathLike, fspath
from typing import BinaryIO, TypeAlias, cast

from core_pdf.impl.engine.spec.s_07_content.models import LayoutBox, LayoutLine, TextRun
from core_pdf.impl.engine.spec.s_07_content.ordering import LayoutAnalyzer
from core_pdf.impl.engine.spec.s_07_content.rendering import strip_private_use_chars
from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument
from core_pdf.impl.engine.spec.s_07_document.models import FieldRecord
from core_pdf.impl.engine.spec.s_07_objects.resolver import ObjectResolver
from core_pdf.impl.engine.spec.s_07_syntax.primitives import PdfDictLike, PdfObject, PdfSource

FileOrName: TypeAlias = str | PathLike[str] | BinaryIO
PdfInput: TypeAlias = FileOrName | bytes | bytearray | memoryview
BBox: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class UnstructuredWord:
    text: str
    bbox: BBox
    start_index: int


@dataclass(frozen=True, slots=True)
class UnstructuredRegion:
    text: str
    bbox: BBox
    words: tuple[UnstructuredWord, ...]
    visible: bool = True


@dataclass(frozen=True, slots=True)
class UnstructuredLink:
    bbox: BBox
    url: str
    link_type: str | None = "Link"


@dataclass(frozen=True, slots=True)
class UnstructuredPageLayout:
    width: float
    height: float
    regions: tuple[UnstructuredRegion, ...]
    links: tuple[UnstructuredLink, ...]


def iter_unstructured_region_layouts(
    pdf_file: PdfInput,
    password: str = "",
    page_numbers: Container[int] | None = None,
    maxpages: int = 0,
) -> Iterator[UnstructuredPageLayout]:
    """Yield page layouts for unstructured without requiring pdfminer.six objects."""

    source = _normalize_pdf_input(pdf_file)
    document = PdfDocument.open(source, password=password)
    selected_pages = page_numbers

    for page_index, page in enumerate(document.pages):
        if selected_pages and page_index not in selected_pages:
            continue

        yield UnstructuredPageLayout(
            width=page.width,
            height=page.height,
            regions=tuple(
                _page_regions(
                    page.chars,
                    page.get_fields(),
                    document.resolver,
                    page.width,
                    page.height,
                )
            ),
            links=tuple(_page_links(document, page.get_annotations(), page.width, page.height)),
        )
        page.state = None
        page.graphics = None
        page.grid_lines = None
        page.texttrace = None
        page.tables = {}

        if maxpages and maxpages <= page_index + 1:
            break


def _normalize_pdf_input(pdf_file: PdfInput) -> PdfSource:
    if isinstance(pdf_file, PathLike):
        path = fspath(pdf_file)
        if not isinstance(path, str):
            raise TypeError(f"Unsupported input type: {type(pdf_file).__name__}")
        return path
    return cast(PdfSource, pdf_file)


def _page_regions(
    runs: list[TextRun],
    fields: list[FieldRecord],
    resolver: ObjectResolver,
    page_width: float,
    page_height: float,
) -> Iterator[UnstructuredRegion]:
    lines = LayoutAnalyzer.cluster_into_lines(runs)
    boxes = LayoutAnalyzer.order_boxes(LayoutAnalyzer.cluster_into_boxes(lines))
    for box in boxes:
        yield from _regions_from_box(box, page_width, page_height)
    yield from _field_regions(fields, resolver, page_width, page_height)


def _field_regions(
    fields: list[FieldRecord],
    resolver: ObjectResolver,
    page_width: float,
    page_height: float,
) -> Iterator[UnstructuredRegion]:
    for field in fields:
        widget = field.widget if isinstance(field.widget, dict) else field.dict
        rect = resolver.resolve_box(widget.get("Rect"))
        if rect is None:
            continue
        text = resolver.resolve_str(field.value) or resolver.resolve_name_like_value(field.value)
        if not text or not text.strip():
            continue
        bbox = _clamp_bbox(rect, page_width, page_height)
        yield UnstructuredRegion(
            text=text,
            bbox=bbox,
            words=(UnstructuredWord(text, bbox, 0),),
        )


def _regions_from_box(
    box: LayoutBox, page_width: float, page_height: float
) -> Iterator[UnstructuredRegion]:
    current: list[LayoutLine] = []
    for line in sorted(box.lines, key=lambda ln: (-ln.y1, ln.x0)):
        if current and _line_starts_new_region(current[-1], line):
            region = _region_from_lines(current, page_width, page_height)
            if region.text.strip():
                yield region
            current = []
        current.append(line)
    if current:
        region = _region_from_lines(current, page_width, page_height)
        if region.text.strip():
            yield region


def _line_starts_new_region(prev: LayoutLine, line: LayoutLine) -> bool:
    vertical_gap = prev.y0 - line.y1
    if vertical_gap > max(8.0, min(prev.height, line.height) * 0.8):
        return True
    if prev.height > line.height * 1.8 or line.height > prev.height * 1.8:
        return True
    x_overlap = min(prev.x1, line.x1) - max(prev.x0, line.x0)
    min_width = min(prev.x1 - prev.x0, line.x1 - line.x0)
    return min_width > 0 and x_overlap / min_width < 0.25


def _region_from_lines(
    lines: list[LayoutLine], page_width: float, page_height: float
) -> UnstructuredRegion:
    text_parts: list[str] = []
    words: list[UnstructuredWord] = []
    visible = True
    for line in lines:
        if text_parts:
            text_parts.append("\n")
        line_text, line_words, line_visible = _render_line_with_words(
            line, len("".join(text_parts))
        )
        text_parts.append(line_text)
        words.extend(line_words)
        visible = visible and line_visible

    return UnstructuredRegion(
        text="".join(text_parts),
        bbox=_clamp_bbox(
            (
                min(line.x0 for line in lines),
                min(line.y0 for line in lines),
                max(line.x1 for line in lines),
                max(line.y1 for line in lines),
            ),
            page_width,
            page_height,
        ),
        words=tuple(
            UnstructuredWord(
                word.text,
                _clamp_bbox(word.bbox, page_width, page_height),
                word.start_index,
            )
            for word in words
        ),
        visible=visible,
    )


def _render_line_with_words(
    line: LayoutLine, start_offset: int
) -> tuple[str, list[UnstructuredWord], bool]:
    parts: list[str] = []
    words: list[UnstructuredWord] = []
    word_text: list[str] = []
    word_bbox: BBox | None = None
    word_start = start_offset
    cursor = start_offset
    visible = True

    prev_run: TextRun | None = None
    for run in _sorted_line_runs(line):
        text = strip_private_use_chars(run.text)
        if not text:
            continue
        visible = visible and run.visible

        separator = _separator_between(prev_run, run)
        if separator and _should_emit_separator(parts, separator, text):
            _flush_word(words, word_text, word_bbox, word_start)
            word_bbox = None
            parts.append(separator)
            cursor += len(separator)

        for char in text:
            if char.isspace():
                _flush_word(words, word_text, word_bbox, word_start)
                word_bbox = None
                parts.append(char)
                cursor += len(char)
                continue
            if not word_text:
                word_start = cursor
                word_bbox = (run.x0, run.y0, run.x1, run.y1)
            else:
                word_bbox = _union_bbox(word_bbox, (run.x0, run.y0, run.x1, run.y1))
            word_text.append(char)
            parts.append(char)
            cursor += len(char)
        prev_run = run

    _flush_word(words, word_text, word_bbox, word_start)
    return "".join(parts).rstrip(), words, visible


def _should_emit_separator(parts: list[str], separator: str, next_text: str) -> bool:
    if separator == " ":
        if parts and parts[-1].isspace():
            return False
        return not next_text[:1].isspace()
    return True


def _sorted_line_runs(line: LayoutLine) -> list[TextRun]:
    if line.rotation_angle == 90:
        return sorted(line.runs, key=lambda run: (run.y0, run.order))
    if line.rotation_angle == 270:
        return sorted(line.runs, key=lambda run: (-run.y1, run.order))
    return sorted(line.runs, key=lambda run: (run.x0, run.order))


def _separator_between(prev_run: TextRun | None, run: TextRun) -> str:
    if prev_run is None:
        return ""
    angle = run.rotation_angle
    if angle == 90:
        axis_gap = run.y0 - prev_run.y1
        thickness = run.x1 - run.x0
        threshold = max(run.space_width * 0.25, thickness * 0.12, 1.0)
        return " " if axis_gap > threshold else ""
    if angle == 270:
        axis_gap = prev_run.y0 - run.y1
        thickness = run.x1 - run.x0
        threshold = max(run.space_width * 0.25, thickness * 0.12, 1.0)
        return " " if axis_gap > threshold else ""

    x_gap = run.x0 - prev_run.x1
    height = run.y1 - run.y0
    threshold = max(run.space_width * 0.25, height * 0.12, 1.0)
    if x_gap > threshold:
        return " "
    if (prev_run.x1 - run.x0) > max(threshold * 4.0, height * 2.0, 24.0):
        return "\n"
    return ""


def _flush_word(
    words: list[UnstructuredWord],
    word_text: list[str],
    word_bbox: BBox | None,
    word_start: int,
) -> None:
    if word_text and word_bbox is not None:
        words.append(UnstructuredWord("".join(word_text), word_bbox, word_start))
        word_text.clear()


def _union_bbox(left: BBox | None, right: BBox) -> BBox:
    if left is None:
        return right
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def _page_links(
    document: PdfDocument, annotations: object, page_width: float, page_height: float
) -> Iterator[UnstructuredLink]:
    if not isinstance(annotations, list):
        return
    for annotation in annotations:
        rect = getattr(annotation, "rect", None)
        if rect is None:
            continue
        annot_dict = getattr(annotation, "dict", None)
        if not isinstance(annot_dict, dict):
            continue
        url = _annotation_url(document, annot_dict)
        if url:
            yield UnstructuredLink(
                _clamp_bbox(rect, page_width, page_height),
                url,
                getattr(annotation, "subtype", None),
            )


def _annotation_url(document: PdfDocument, annotation: PdfDictLike) -> str | None:
    action = document.resolver.resolve_dict(annotation.get("A"))
    if action is not None:
        uri = document.resolver.resolve_str(action.get("URI"))
        if uri:
            return uri
    uri_obj: PdfObject = annotation.get("URI")
    return document.resolver.resolve_str(uri_obj)


def _clamp_bbox(bbox: BBox, page_width: float, page_height: float) -> BBox:
    x0, y0, x1, y1 = bbox
    left, right = sorted((max(0.0, min(page_width, x0)), max(0.0, min(page_width, x1))))
    bottom, top = sorted((max(0.0, min(page_height, y0)), max(0.0, min(page_height, y1))))
    return (left, bottom, right, top)


__all__ = [
    "UnstructuredLink",
    "UnstructuredPageLayout",
    "UnstructuredRegion",
    "UnstructuredWord",
    "iter_unstructured_region_layouts",
]


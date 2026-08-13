"""Local Unstructured-style partition and element conversion APIs."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from typing import Any, TypeAlias

import numpy

from core_pdf import PdfDocument
from core_pdf.api.compat.pdfminer import LAParams, LTChar, LTFigure, LTTextBox, extract_pages

PdfInput: TypeAlias = Any


class ElementMetadata(dict[str, Any]):
    """Dictionary-compatible Unstructured metadata with attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self)


@dataclass(frozen=True, slots=True)
class Element:
    text: str
    metadata: ElementMetadata = field(default_factory=ElementMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, ElementMetadata):
            object.__setattr__(self, "metadata", ElementMetadata(self.metadata))

    @property
    def category(self) -> str:
        return type(self).__name__

    @property
    def element_id(self) -> str:
        return str(self.metadata.get("element_id", ""))

    @property
    def id(self) -> str:
        return self.element_id

    @property
    def page_number(self) -> int | None:
        value = self.metadata.get("page_number")
        return value if isinstance(value, int) else None

    @property
    def coordinates(self) -> tuple[float, float, float, float] | None:
        value = self.metadata.get("coordinates", self.metadata.get("bbox"))
        return tuple(value) if isinstance(value, (tuple, list)) and len(value) == 4 else None

    @property
    def text_as_html(self) -> str | None:
        value = self.metadata.get("text_as_html")
        return value if isinstance(value, str) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.category,
            "text": self.text,
            "element_id": self.element_id,
            "metadata": dict(self.metadata),
        }


class Title(Element):
    pass


class NarrativeText(Element):
    pass


class UncategorizedText(Element):
    pass


class Table(Element):
    pass


class Image(Element):
    pass


class PageBreak(Element):
    pass


class Header(Element):
    pass


class Footer(Element):
    pass


class ListItem(Element):
    pass


class EmailAddress(Element):
    pass


class Address(Element):
    pass


internal_BULLET_CHARS = "\x95•‣⁃ㅤ⁌⁍∙○●◘◦☙❥❧⦾⦿-–\uf0b7*·"
internal_BULLET = re.compile(
    rf"^\s*[{re.escape(internal_BULLET_CHARS)}](?![{re.escape(internal_BULLET_CHARS)}])"
)
internal_NUMBERED = re.compile(r"^\s*\d+(?:\.|\))\s+.+")
internal_EMAIL = re.compile(r"[a-z0-9.\-+_]+@[a-z0-9.\-+_]+\.[a-z]+", re.IGNORECASE)
internal_ADDRESS = re.compile(
    r"^(?:[A-Z][a-z.\-]{1,15} ?){1,5},\s?"
    r"(?:AL|AK|AS|AZ|AR|CA|CO|CT|DE|DC|FM|FL|GA|GU|HI|ID|IL|IN|IA|KS|KY|LA|ME|MH|"
    r"MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|MP|OH|OK|OR|PW|PA|PR|RI|SC|SD|"
    r"TN|TX|UT|VT|VI|VA|WA|WV|WI|WY)(?:, |\s)?\d{5}(?:-\d{4})?\b",
    re.IGNORECASE,
)
internal_VERB = re.compile(
    r"\b(?:am|are|be|been|being|can|could|did|do|does|had|has|have|is|may|might|must|"
    r"shall|should|was|were|will|would|suggest|suggests|include|includes|included|"
    r"provide|provides|provided|show|shows|shown|use|uses|used|observations?|"
    r"accepts?|agrees?|applies|built|clears?|closes?|consider|constructs?|contains?|"
    r"convert|create|creates|declares?|define|describe|describes|determine|display|"
    r"escape|explain|extract|gives?|inspect|install|live|looks|mail|make|makes|means|"
    r"merge|needs?|note|observe|offer|perform|points|reads?|reference|refund|regain|"
    r"represents?|run|save|saves|see|sign|supports?|take|transform|watch|wins?)\b",
    re.IGNORECASE,
)
internal_GRAPHICS_OPS = re.compile(
    rb"(?:^|(?<=\s))(?:m|l|c|v|y|h|re|S|s|f|F|f\*|B|B\*|b|b\*|n|W|W\*|cm|q|Q|"
    rb"Do|g|G|rg|RG|k|K|cs|CS|w|J|j|M|d|i|gs)(?=\s|$)"
)
internal_TEXT_OPS = re.compile(rb"(?:^|(?<=\s))(?:Tj|TJ|'|\"|Tf|Td|TD|Tm|T\*|BT|ET)(?=\s|$)")
internal_POS_VERB_TAGS = frozenset({"VB", "VBG", "VBD", "VBN", "VBP", "VBZ"})
internal_NLP: Any | None = None
internal_NLP_UNAVAILABLE = False


def internal_nlp() -> Any | None:
    """Return an optional English POS pipeline without depending on Unstructured.

    The compatibility package remains usable in core-pdf's minimal installation.
    When the standard ``en_core_web_sm`` model is present, use its tokenizer,
    sentence boundaries, and POS tagger to reproduce Unstructured's semantic
    element classification instead of maintaining an ever-growing lexical
    approximation here.
    """
    global internal_NLP, internal_NLP_UNAVAILABLE
    if internal_NLP is not None:
        return internal_NLP
    if internal_NLP_UNAVAILABLE:
        return None
    try:
        model = import_module("en_core_web_sm")
        internal_NLP = model.load()
    except (ImportError, OSError):
        internal_NLP_UNAVAILABLE = True
        return None
    return internal_NLP


@lru_cache(maxsize=4096)
def internal_nlp_features(
    text: str,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]] | None:
    nlp = internal_nlp()
    if nlp is None:
        return None
    document = nlp(text[: nlp.max_length])
    tokens = tuple((token.text, token.tag_) for token in document)
    sentences = tuple(sentence.text for sentence in document.sents)
    return tokens, sentences


def internal_pdf_too_complex(filename: object, password: str) -> bool:
    with PdfDocument.open(filename, password=password) as document:
        if len(document.raw_data) < 1_048_576:
            return False
        for page in document.pages:
            raw_data = b"".join(stream.data for stream in page.content_streams)
            if len(raw_data) < 100_000:
                continue
            graphics = len(internal_GRAPHICS_OPS.findall(raw_data))
            if graphics <= 10_000:
                continue
            text = len(internal_TEXT_OPS.findall(raw_data))
            if graphics / max(text, 1) > 20.0:
                return True
    return False


def internal_clean_text(text: str) -> str:
    # Match Unstructured's ``clean_extra_whitespace_with_index_run``: line
    # endings and non-breaking spaces become ordinary spaces, repeated spaces
    # collapse, but tabs remain meaningful (notably in table-like text boxes).
    cleaned = text.translate(
        {ord("\n"): ord(" "), ord("\xa0"): ord(" "), ord("\u2009"): ord(" ")}
    )
    return re.sub(r" {2,}", " ", cleaned).strip()


def internal_layout_regions(
    items: list[LTTextBox],
    page_height: float,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Project each canonical pdfminer text box to one Unstructured region."""
    del page_height
    return [
        (text, item.bbox)
        for item in items
        if (text := internal_clean_text(internal_deduplicated_box_text(item)))
    ]


def internal_duplicate_character(first: LTChar, second: LTChar, threshold: float = 2.0) -> bool:
    if first.get_text() != second.get_text():
        return False
    if abs(first.x0 - second.x0) >= threshold or abs(first.y0 - second.y0) >= threshold:
        return False
    first_width = first.x1 - first.x0
    second_width = second.x1 - second.x0
    average_width = (first_width + second_width) / 2.0
    if average_width <= 0:
        return False
    overlap = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    return overlap / average_width > 0.5


def internal_deduplicated_box_text(box: LTTextBox) -> str:
    parts: list[str] = []
    for line in box:
        previous: LTChar | None = None
        for item in line:
            if isinstance(item, LTChar):
                if previous is not None and internal_duplicate_character(previous, item):
                    continue
                previous = item
            parts.append(item.get_text())
    return "".join(parts)


def internal_sentence_count(sentences: tuple[str, ...], minimum_words: int) -> int:
    punctuation = str.maketrans("", "", string.punctuation)
    return sum(
        len(sentence.translate(punctuation).split()) >= minimum_words for sentence in sentences
    )


def internal_element_class(
    text: str,
    bbox: tuple[float, float, float, float],
    page_height: float,
) -> type[Element]:
    height_percentage = 1.0 - (bbox[1] + bbox[3]) / (2.0 * page_height) if page_height else 0.5
    if height_percentage < 0.07:
        return Header
    if height_percentage > 0.93:
        return Footer
    if internal_BULLET.match(text) or internal_NUMBERED.match(text):
        return ListItem
    if internal_EMAIL.match(text.strip()):
        return EmailAddress
    if internal_ADDRESS.search(text):
        return Address
    alphabetic = sum(character.isalpha() for character in text)
    non_space = sum(not character.isspace() for character in text)
    alpha_ratio = alphabetic / max(non_space, 1)
    nlp_features = internal_nlp_features(text)
    if nlp_features is None:
        tagged_tokens: tuple[tuple[str, str], ...] = ()
        sentences = tuple(part for part in re.split(r"(?<=[.!?])\s+", text) if part)
        word_tokens = re.findall(r"[^\W\d_]+", text, re.UNICODE)
    else:
        tagged_tokens, sentences = nlp_features
        word_tokens = [token for token, _tag in tagged_tokens if token.isalpha()]
    capitalized = sum(word.istitle() or word.isupper() for word in word_tokens)
    long_sentence_count = internal_sentence_count(sentences, 3)
    exceeds_cap_ratio = long_sentence_count <= 1 and (
        text.isupper() or not word_tokens or capitalized / len(word_tokens) > 0.5
    )
    has_verb = (
        any(tag in internal_POS_VERB_TAGS for _token, tag in tagged_tokens)
        if nlp_features is not None
        else internal_VERB.search(text) is not None
    )
    if (
        alpha_ratio >= 0.5
        and not text.isnumeric()
        and (long_sentence_count > 1 or not exceeds_cap_ratio)
        and (has_verb or long_sentence_count >= 2)
    ):
        return NarrativeText
    if (
        len(text.split(" ")) <= 12
        and alpha_ratio >= 0.5
        and not text.isnumeric()
        and not text.endswith(",")
        and not (text.isupper() and re.search(r"[^\w\s]$", text))
        and internal_sentence_count(sentences, 5) <= 1
    ):
        return Title
    return UncategorizedText


def internal_projection_segments(
    boxes: numpy.ndarray[Any, Any], axis: int
) -> list[tuple[int, int]]:
    if not len(boxes):
        return []
    length = int(numpy.max(boxes[:, axis::2]))
    projection = numpy.zeros(max(0, length), dtype=numpy.int64)
    for box in boxes:
        projection[int(box[axis]) : int(box[axis + 2])] += 1
    occupied = numpy.where(projection > 0)[0]
    if not len(occupied):
        return []
    gaps = numpy.where(occupied[1:] - occupied[:-1] > 1)[0]
    starts = [int(occupied[0]), *(int(occupied[index + 1]) for index in gaps)]
    ends = [*(int(occupied[index]) + 1 for index in gaps), int(occupied[-1]) + 1]
    return list(zip(starts, ends))


def internal_recursive_xy_cut(
    boxes: numpy.ndarray[Any, Any],
    indices: numpy.ndarray[Any, Any],
    result: list[int],
) -> None:
    x_order = boxes[:, 0].argsort()
    x_boxes = boxes[x_order]
    x_indices = indices[x_order]
    for x0, x1 in internal_projection_segments(x_boxes, 0):
        x_mask = (x0 <= x_boxes[:, 0]) & (x_boxes[:, 0] < x1)
        chunk = x_boxes[x_mask]
        chunk_indices = x_indices[x_mask]
        y_order = chunk[:, 1].argsort()
        y_boxes = chunk[y_order]
        y_indices = chunk_indices[y_order]
        y_segments = internal_projection_segments(y_boxes, 1)
        if len(y_segments) == 1:
            result.extend(int(index) for index in y_indices)
            continue
        for y0, y1 in y_segments:
            y_mask = (y0 <= y_boxes[:, 1]) & (y_boxes[:, 1] < y1)
            internal_recursive_xy_cut(y_boxes[y_mask], y_indices[y_mask], result)


def internal_region_order(
    regions: list[tuple[str, tuple[float, float, float, float]]],
    page_height: float,
) -> list[int]:
    basic_order = sorted(
        range(len(regions)),
        key=lambda index: (page_height - regions[index][1][3], regions[index][1][0]),
    )
    boxes = []
    for index in basic_order:
        x0, y0, x1, y1 = regions[index][1]
        left = int(x0)
        top = int(page_height - y1)
        right = int(x1)
        bottom = int(page_height - y0)
        boxes.append(
            (
                left,
                top,
                int(right - (right - left) * 0.1),
                int(bottom - (bottom - top) * 0.1),
            )
        )
    if not boxes:
        return []
    # Unstructured requires every coordinate to be non-negative before it
    # applies XY-cut. A single box extending beyond the media box makes it keep
    # the deterministic basic (top, left) order for the entire page.
    if any(coordinate < 0 for box in boxes for coordinate in box):
        return basic_order
    result: list[int] = []
    internal_recursive_xy_cut(
        numpy.asarray(boxes, dtype=numpy.int64),
        numpy.asarray(basic_order, dtype=numpy.int64),
        result,
    )
    return result


def internal_combine_list_regions(
    regions: list[tuple[str, tuple[float, float, float, float]]],
    page_height: float,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Apply Unstructured's pre-sort continuation merge for list elements."""
    combined: list[tuple[str, tuple[float, float, float, float]]] = []
    active_index: int | None = None
    active_bbox: tuple[float, float, float, float] | None = None
    for text, bbox in regions:
        element_class = internal_element_class(text, bbox, page_height)
        if element_class is ListItem:
            active_index = len(combined)
            active_bbox = bbox
            combined.append((text, bbox))
            continue
        if active_index is not None and active_bbox is not None:
            left, bottom, right, top = active_bbox
            width = right - left
            height = top - bottom
            current_left, current_bottom, current_right, current_top = bbox
            within_x = (
                current_left > left - 0.2 * width
                and current_right < right + 0.2 * width
                and current_left >= left
            )
            within_y = (
                current_top > bottom - 0.3 * height and current_top < top + 0.3 * height
            )
            if within_x and within_y:
                active_text, _ = combined[active_index]
                merged_bbox = (
                    min(left, current_left),
                    min(bottom, current_bottom),
                    max(right, current_right),
                    max(top, current_top),
                )
                merged_region = (f"{active_text} {text}", merged_bbox)
                combined[active_index] = merged_region
                # Mirror the reference continuation loop exactly: it removes
                # the most recently emitted element, which need not be the
                # active list item when intervening columns were encountered,
                # then appends a merged copy of that list item.
                if combined:
                    combined.pop()
                combined.append(merged_region)
                active_index = len(combined) - 1
                active_bbox = merged_bbox
                continue
        combined.append((text, bbox))
    return combined


def partition_pdf(filename: object, **kwargs: object) -> list[Element]:
    include_page_breaks = bool(kwargs.pop("include_page_breaks", False))
    include_metadata = bool(kwargs.pop("include_metadata", True))
    word_margin = float(kwargs.pop("pdfminer_word_margin", 0.185) or 0.185)
    password = str(kwargs.pop("password", "") or "")
    if internal_pdf_too_complex(filename, password):
        return []
    result: list[Element] = []
    pages = extract_pages(filename, password=password, laparams=LAParams(word_margin=word_margin))
    for page in pages:
        text_boxes = [item for item in page if isinstance(item, LTTextBox)]
        regions = internal_layout_regions(text_boxes, page.height)
        for figure in (item for item in page if isinstance(item, LTFigure)):
            regions.extend(
                (text, figure.bbox)
                for snippet in figure.text_snippets
                if (text := internal_clean_text(snippet))
            )
        regions = internal_combine_list_regions(regions, page.height)
        for element_index in internal_region_order(regions, page.height):
            text, bbox = regions[element_index]
            element_class = internal_element_class(text, bbox, page.height)
            if element_class is ListItem:
                text = internal_BULLET.sub("", text, count=1).strip()
            metadata = (
                ElementMetadata(
                    {
                        "element_id": f"p{page.pageid}-e{element_index}",
                        "page_number": page.pageid,
                        "bbox": bbox,
                    }
                )
                if include_metadata
                else ElementMetadata()
            )
            result.append(element_class(text, metadata))
        if include_page_breaks:
            result.append(PageBreak("", ElementMetadata(page_number=page.pageid)))
    return result


__all__ = (
    "Element",
    "ElementMetadata",
    "EmailAddress",
    "Footer",
    "Header",
    "Image",
    "ListItem",
    "NarrativeText",
    "PageBreak",
    "Table",
    "Title",
    "UncategorizedText",
    "partition_pdf",
)

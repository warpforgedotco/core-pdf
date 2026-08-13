"""Local Unstructured-style partition and element conversion APIs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from core_pdf.api.compat.pdfminer import LAParams, LTTextBox, extract_pages

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


internal_BULLET = re.compile(r"^\s*(?:[•◦▪‣⁃●○■□◆◇‒–—]|[-*+]\s)")
internal_NUMBERED = re.compile(r"^\s*(?:\(?\d+[.)]|[A-Za-z][.)])\s+")
internal_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
internal_ADDRESS = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?$")
internal_VERB = re.compile(
    r"\b(?:am|are|be|been|being|can|could|did|do|does|had|has|have|is|may|might|must|"
    r"shall|should|was|were|will|would|suggest|suggests|include|includes|included|"
    r"provide|provides|provided|show|shows|shown|use|uses|used|observations?)\b",
    re.IGNORECASE,
)


def internal_clean_text(text: str) -> str:
    return " ".join(text.replace("\x00", "(cid:0)").split())


def internal_layout_regions(
    items: list[LTTextBox],
    page_height: float,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Coalesce fragments that pdfminer places in the same text box."""
    regions: list[tuple[list[tuple[float, str]], tuple[float, float, float, float]]] = []
    for item in items:
        text = internal_clean_text(item.get_text())
        if not text:
            continue
        bbox = item.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        match: int | None = None
        for index in range(len(regions) - 1, -1, -1):
            previous_parts, previous = regions[index]
            previous_height = previous[3] - previous[1]
            previous_width = previous[2] - previous[0]
            vertical_overlap = max(0.0, min(previous[3], bbox[3]) - max(previous[1], bbox[1]))
            horizontal_gap = max(0.0, max(previous[0], bbox[0]) - min(previous[2], bbox[2]))
            vertical_gap = max(0.0, max(previous[1], bbox[1]) - min(previous[3], bbox[3]))
            same_line = (
                vertical_overlap >= min(previous_height, height) * 0.5
                and horizontal_gap <= max(previous_height, height) * 0.6
                and (
                    min(previous_width, width) <= max(previous_height, height) * 1.5
                    or max(previous[3], bbox[3]) / max(page_height, 1.0) < 0.07
                )
            )
            tiny_adjunct = (
                height <= 2.0
                and vertical_gap <= 4.0
                and bbox[0] <= previous[2] + 2.0
                and bbox[2] >= previous[0] - 2.0
            )
            continuation = (
                vertical_gap <= max(previous_height, height) * 0.65
                and abs(previous[2] - bbox[2]) <= max(previous_height, height) * 0.5
                and bbox[0] >= previous[0]
                and text.startswith(("»", "(", "["))
            )
            if same_line or tiny_adjunct or continuation:
                match = index
                break
            if previous[1] - bbox[3] > max(previous_height, height) * 2.0:
                break
        if match is None:
            regions.append(([(bbox[0], text)], bbox))
            continue
        previous_parts, previous = regions[match]
        regions[match] = (
            [*previous_parts, (bbox[0], text)],
            (
                min(previous[0], bbox[0]),
                min(previous[1], bbox[1]),
                max(previous[2], bbox[2]),
                max(previous[3], bbox[3]),
            ),
        )
    projected = [(" ".join(text for _, text in sorted(parts)), bbox) for parts, bbox in regions]
    return projected


def internal_element_class(
    text: str,
    bbox: tuple[float, float, float, float],
    page_height: float,
) -> type[Element]:
    top = page_height - bbox[3]
    bottom = page_height - bbox[1]
    header_threshold = 0.07 if "(cid:" in text else 0.075
    if page_height > 0 and top / page_height < header_threshold:
        return Header
    if page_height > 0 and bottom / page_height > 0.93:
        return Footer
    if internal_BULLET.match(text) or internal_NUMBERED.match(text):
        return ListItem
    if internal_EMAIL.fullmatch(text):
        return EmailAddress
    if internal_ADDRESS.search(text):
        return Address
    words = text.split()
    alphabetic = sum(character.isalpha() for character in text)
    alpha_ratio = alphabetic / max(len(text), 1)
    has_verb = internal_VERB.search(text) is not None or bool(
        re.search(r"\b(?:given|using|located|detected|identified|checked|attached)\b", text, re.I)
    )
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", text))
    if (
        alpha_ratio >= 0.5
        and not text.isnumeric()
        and not text.isupper()
        and (
            has_verb
            or sentence_count >= 2
            or (len(words) >= 5 and text.rstrip().endswith((".", "?", "!")))
        )
    ):
        return NarrativeText
    if len(words) <= 12 and alpha_ratio >= 0.4 and not text.isnumeric() and not text.endswith(","):
        return Title
    return UncategorizedText


def partition_pdf(filename: object, **kwargs: object) -> list[Element]:
    include_page_breaks = bool(kwargs.pop("include_page_breaks", False))
    include_metadata = bool(kwargs.pop("include_metadata", True))
    word_margin = float(kwargs.pop("pdfminer_word_margin", 0.185) or 0.185)
    password = str(kwargs.pop("password", "") or "")
    result: list[Element] = []
    pages = extract_pages(filename, password=password, laparams=LAParams(word_margin=word_margin))
    for page in pages:
        items = [item for item in page if isinstance(item, LTTextBox)]
        regions = internal_layout_regions(items, page.height)
        # Core's pdfminer box geometry is not yet pixel-identical enough for recursive
        # XY-cut partitions to be stable. Preserve its deterministic page-space order
        # until those box boundaries converge.
        order = sorted(
            range(len(regions)),
            key=lambda index: (
                -(regions[index][1][1] + regions[index][1][3]) / 2.0,
                regions[index][1][0],
            ),
        )
        for element_index in order:
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

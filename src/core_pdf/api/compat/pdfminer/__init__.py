from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from html import escape
from io import BytesIO
from typing import Any, BinaryIO, TextIO, TypeAlias, cast

from core_pdf import PdfDocument
from core_pdf.impl.engine.layout.geometry import bbox_union

PdfInput: TypeAlias = Any


@dataclass(frozen=True, slots=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True, slots=True)
class TextCharacter:
    text: str
    bbox: Rect
    font_name: str | None = None
    font_size: float | None = None
    sequence: int | None = None


@dataclass(frozen=True, slots=True)
class TextSpan:
    text: str
    bbox: Rect
    characters: tuple[TextCharacter, ...] = ()
    font_name: str | None = None
    font_size: float | None = None
    sequence: int | None = None


def synthesize_characters(
    text: str, box: tuple[float, float, float, float]
) -> Iterator[tuple[str, tuple[float, float, float, float]]]:
    x0, y0, x1, y1 = box
    width = (x1 - x0) / max(1, len(text))
    for index, character in enumerate(text):
        yield character, (x0 + index * width, y0, x0 + (index + 1) * width, y1)


@dataclass(slots=True)
class LAParams:
    """pdfminer.six layout parameters accepted by the compatibility facade."""

    line_overlap: float = 0.5
    char_margin: float = 2.0
    line_margin: float = 0.5
    word_margin: float = 0.1
    boxes_flow: float | None = 0.5
    detect_vertical: bool = False
    all_texts: bool = False


class LTItem:
    def analyze(self, laparams: LAParams) -> None:
        del laparams


@dataclass(slots=True)
class LTComponent(LTItem):
    bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        self.x0, self.y0, self.x1, self.y1 = self.bbox
        self.width = self.x1 - self.x0
        self.height = self.y1 - self.y0


class LTText(LTItem):
    def get_text(self) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class LTAnno(LTText):
    text: str

    def get_text(self) -> str:
        return self.text


@dataclass(slots=True)
class LTChar(LTComponent, LTText):
    text: str = ""
    fontname: str | None = None
    size: float = 0.0

    def get_text(self) -> str:
        return self.text

    @property
    def font_name(self) -> str | None:
        return self.fontname

    @property
    def adv(self) -> float:
        return self.width

    @property
    def upright(self) -> bool:
        return self.width >= self.height

    @property
    def matrix(self) -> tuple[float, float, float, float, float, float]:
        return (1.0, 0.0, 0.0, 1.0, self.x0, self.y0)


@dataclass(slots=True)
class LTTextLine(LTComponent, LTText):
    _objs: list[LTText | LTChar] = field(default_factory=list)
    fragment_group: object | None = None

    def __iter__(self) -> Iterator[LTText | LTChar]:
        return iter(self._objs)

    def get_text(self) -> str:
        return "".join(item.get_text() for item in self._objs)


class LTTextLineHorizontal(LTTextLine):
    pass


class LTTextLineVertical(LTTextLine):
    pass


@dataclass(slots=True)
class LTTextBox(LTComponent, LTText):
    _objs: list[LTTextLine] = field(default_factory=list)

    def __iter__(self) -> Iterator[LTTextLine]:
        return iter(self._objs)

    def get_text(self) -> str:
        return "".join(line.get_text() for line in self._objs)


class LTTextBoxHorizontal(LTTextBox):
    pass


class LTTextBoxVertical(LTTextBox):
    pass


@dataclass(slots=True)
class LTImage(LTComponent):
    name: str = ""
    stream: object | None = None


@dataclass(slots=True)
class LTPage(LTComponent):
    pageid: int
    rotate: float = 0
    _objs: list[LTItem] = field(default_factory=list)

    def __iter__(self) -> Iterator[LTItem]:
        return iter(self._objs)


def _bbox(rect: Rect) -> tuple[float, float, float, float]:
    return (rect.x0, rect.y0, rect.x1, rect.y1)


def _characters(span: TextSpan) -> Iterable[TextCharacter]:
    if span.characters:
        return span.characters
    box = (span.bbox.x0, span.bbox.y0, span.bbox.x1, span.bbox.y1)
    return tuple(
        TextCharacter(
            text=character,
            bbox=Rect(*sub_box),
            font_name=span.font_name,
            font_size=span.font_size,
            sequence=span.sequence,
        )
        for character, sub_box in synthesize_characters(span.text, box)
    )


def _span_fragments(span: TextSpan, char_margin: float = 2.0) -> Iterable[TextSpan]:
    characters = tuple(_characters(span))
    if not characters:
        return (span,)
    fragments: list[TextSpan] = []
    current: list[TextCharacter] = []
    current_sequence: int | None = None
    sequence_run_length = 0
    for character in characters:
        vertical_fragment = False
        if current:
            previous = current[-1].bbox
            current_box = character.bbox
            x_overlap = min(previous.x1, current_box.x1) - max(previous.x0, current_box.x0)
            vertical_fragment = x_overlap > 0 and abs(previous.y0 - current_box.y0) > (
                0.5 * min(previous.height, current_box.height)
            )
            horizontal_gap = current_box.x0 - previous.x1
            horizontal_fragment = horizontal_gap > char_margin * max(
                previous.height, current_box.height
            )
        else:
            horizontal_fragment = False
        sequence_fragment = (
            character.sequence != current_sequence and sequence_run_length >= 2
            if current
            else False
        )
        if current and (sequence_fragment or vertical_fragment or horizontal_fragment):
            fragments.append(_character_span(current, span))
            current = []
        if character.sequence == current_sequence:
            sequence_run_length += 1
        else:
            current_sequence = character.sequence
            sequence_run_length = 1
        current.append(character)
    if current:
        fragments.append(_character_span(current, span))
    return tuple(fragments)


def _character_span(characters: list[TextCharacter], source: TextSpan) -> TextSpan:
    box = bbox_union(
        (character.bbox.x0, character.bbox.y0, character.bbox.x1, character.bbox.y1)
        for character in characters
    )
    if box is None:
        raise ValueError("cannot build a span from zero characters")
    return TextSpan(
        text="".join(character.text for character in characters),
        bbox=Rect(*box),
        characters=tuple(characters),
        font_name=source.font_name,
        font_size=source.font_size,
        sequence=characters[0].sequence,
    )


def _is_vertical_span(span: TextSpan) -> bool:
    characters = tuple(_characters(span))
    if len(characters) < 2:
        return False
    x_values = [character.bbox.x0 for character in characters]
    y_values = [character.bbox.y0 for character in characters]
    return max(x_values) - min(x_values) <= 1.0 and max(y_values) - min(y_values) > 1.0


def _group_lines(lines: list[LTTextLineHorizontal], margin: float) -> list[LTTextBoxHorizontal]:
    boxes: list[LTTextBoxHorizontal] = []
    for line in sorted(lines, key=lambda item: (-item.y1, item.x0)):
        if not boxes:
            boxes.append(LTTextBoxHorizontal(line.bbox, [line]))
            continue
        previous = boxes[-1]
        overlap = min(line.y1, previous.y1) - max(line.y0, previous.y0)
        gap = max(0.0, line.y0 - previous.y1, previous.y0 - line.y1)
        x_overlap = min(line.x1, previous.x1) - max(line.x0, previous.x0)
        if (overlap <= 0 and gap <= margin * max(line.height, previous.height)) or (
            x_overlap > 0
            and max(line.height, previous.height) < 2
            and gap <= 3 * margin * max(line.height, previous.height)
        ):
            previous._objs.append(line)
            previous.bbox = bbox_union((previous.bbox, line.bbox)) or previous.bbox
            previous.__post_init__()
        else:
            boxes.append(LTTextBoxHorizontal(line.bbox, [line]))
    return boxes


def _merge_line_fragments(
    lines: list[LTTextLineHorizontal], char_margin: float, word_margin: float
) -> list[LTTextLineHorizontal]:
    grouped: dict[int, list[LTTextLineHorizontal]] = {}
    ungrouped: list[LTTextLineHorizontal] = []
    for line in lines:
        if line.fragment_group is None:
            ungrouped.append(line)
        else:
            grouped.setdefault(id(line.fragment_group), []).append(line)

    ordered_lines: list[LTTextLineHorizontal] = []
    for group in grouped.values():
        fragments = sorted(group, key=lambda item: item.x0)
        if len(fragments) == 1:
            ordered_lines.extend(fragments)
            continue
        ordered_lines.append(
            LTTextLineHorizontal(
                bbox_union(item.bbox for item in fragments) or fragments[0].bbox,
                [
                    child
                    for fragment in fragments
                    for child in fragment._objs
                    if not isinstance(child, LTAnno)
                ]
                + [LTAnno("\n")],
                fragments[0].fragment_group,
            )
        )
    ordered_lines.extend(ungrouped)
    lines = sorted(ordered_lines, key=lambda item: (-item.y1, item.x0))
    merged: list[LTTextLineHorizontal] = []
    for line in lines:
        if not merged:
            merged.append(line)
            continue
        previous = merged[-1]
        vertical_overlap = min(previous.y1, line.y1) - max(previous.y0, line.y0)
        gap = line.x0 - previous.x1
        same_fragment_group = (
            previous.fragment_group is not None and previous.fragment_group is line.fragment_group
        )
        if (
            vertical_overlap > 0
            and gap <= char_margin * max(line.height, previous.height)
            and same_fragment_group
        ):
            separator = [
                LTAnno(" ") if gap > word_margin * max(line.height, previous.height) else None
            ]
            previous._objs = (
                previous._objs[:-1] + [item for item in separator if item is not None] + line._objs
            )
            previous.bbox = bbox_union((previous.bbox, line.bbox)) or previous.bbox
            previous.__post_init__()
        else:
            merged.append(line)
    return merged


def extract_pages(
    pdf_file: PdfInput,
    password: str = "",
    page_numbers: Iterable[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    laparams: LAParams | None = None,
) -> Iterator[LTPage]:
    """Yield pdfminer.six-shaped pages using core-pdf extraction evidence."""
    del caching
    params = laparams or LAParams()
    selected = set(page_numbers) if page_numbers is not None else None
    document = PdfDocument.open(pdf_file, password=password)
    try:
        yielded = 0
        for page_index, page in enumerate(document.pages):
            if selected is not None and page_index not in selected:
                continue
            if maxpages and yielded >= maxpages:
                break
            lines: list[LTTextLineHorizontal] = []
            for run in page.text_diagnostics().runs:
                if run.bbox is None:
                    continue
                span = TextSpan(
                    text=run.text,
                    bbox=Rect(*run.bbox),
                    font_name=run.font_name,
                    font_size=run.font_size,
                    sequence=run.seqno,
                )
                fragment_group = object()
                if _is_vertical_span(span):
                    chars = tuple(_characters(span))
                    items = [
                        LTChar(
                            _bbox(character.bbox),
                            character.text,
                            character.font_name,
                            character.font_size or 0.0,
                        )
                        for character in chars
                        if character.text
                    ]
                    if items:
                        vertical_items: list[LTText | LTChar] = []
                        for item in items:
                            vertical_items.extend((item, LTAnno("\n")))
                        lines.append(
                            LTTextLineHorizontal(_bbox(span.bbox), vertical_items, fragment_group)
                        )
                    continue
                for fragment in _span_fragments(span, params.char_margin):
                    chars = tuple(_characters(fragment))
                    items = [
                        LTChar(
                            _bbox(character.bbox),
                            character.text,
                            character.font_name,
                            character.font_size or 0.0,
                        )
                        for character in chars
                        if character.text
                    ]
                    if not items:
                        continue
                    lines.append(
                        LTTextLineHorizontal(
                            _bbox(fragment.bbox),
                            items + [LTAnno("\n")],
                            fragment_group,
                        )
                    )
            lines = _merge_line_fragments(
                sorted(lines, key=lambda item: (-item.y1, item.x0)),
                params.char_margin,
                params.word_margin,
            )
            boxes: list[LTItem] = list(_group_lines(lines, params.line_margin))
            yield LTPage(
                (0.0, 0.0, page.width, page.height),
                page.page_number,
                page.rotation,
                boxes,
            )
            yielded += 1
    finally:
        document.close()


def extract_text(
    pdf_file: PdfInput,
    password: str = "",
    page_numbers: Iterable[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    codec: str = "utf-8",
    laparams: LAParams | None = None,
) -> str:
    """Return text with pdfminer.six-compatible high-level semantics."""
    del codec
    pages = [
        "\n".join(item.get_text() for item in page if isinstance(item, LTText))
        for page in extract_pages(pdf_file, password, page_numbers, maxpages, caching, laparams)
    ]
    return "".join(page_text + ("\n" if page_text else "") + "\f" for page_text in pages)


def extract_text_to_fp(
    inf: BinaryIO | PdfInput,
    outfp: TextIO | BinaryIO,
    output_type: str = "text",
    codec: str = "utf-8",
    laparams: LAParams | None = None,
    maxpages: int = 0,
    page_numbers: Iterable[int] | None = None,
    password: str = "",
    **kwargs: Any,
) -> None:
    """Write locally extracted text to a file-like object.

    Text, XML, and HTML output are supported locally. HOCR and tag output are
    intentionally rejected until their accessibility-specific semantics are
    mapped to core-pdf's structured serializers.
    """
    del kwargs
    if output_type == "text":
        output = extract_text(inf, password, page_numbers, maxpages, True, codec, laparams)
    elif output_type in {"xml", "html"}:
        output = _structured_output(inf, output_type, password, page_numbers, maxpages, laparams)
    else:
        raise ValueError(f"unsupported pdfminer output_type: {output_type}")
    if isinstance(outfp, (BytesIO,)):
        outfp.write(output.encode(codec))
    else:
        cast(TextIO, outfp).write(output)


def _structured_output(
    source: BinaryIO | PdfInput,
    output_type: str,
    password: str,
    page_numbers: Iterable[int] | None,
    maxpages: int,
    laparams: LAParams | None,
) -> str:
    pages = list(extract_pages(source, password, page_numbers, maxpages, True, laparams))
    if output_type == "html":
        html_parts: list[str] = []
        for page in pages:
            html_parts.append(f'<div class="page" data-page="{page.pageid}">')
            for item in page:
                if isinstance(item, LTText) and isinstance(item, LTComponent):
                    bbox = ",".join(str(value) for value in item.bbox)
                    html_parts.append(
                        f'<div class="textbox" data-bbox="{bbox}">{escape(item.get_text())}</div>'
                    )
            html_parts.append("</div>")
        body = "".join(html_parts)
        return f"<!doctype html><html><body>{body}</body></html>"
    xml_parts: list[str] = []
    for page in pages:
        xml_parts.append(f'<page id="{page.pageid}" bbox="{page.bbox}">')
        for item in page:
            if isinstance(item, LTText) and isinstance(item, LTComponent):
                xml_parts.append(f'<textbox bbox="{item.bbox}">{escape(item.get_text())}</textbox>')
        xml_parts.append("</page>")
    return "<pages>" + "".join(xml_parts) + "</pages>"


__all__ = (
    "LAParams",
    "LTAnno",
    "LTChar",
    "LTImage",
    "LTItem",
    "LTPage",
    "LTText",
    "LTTextBox",
    "LTTextBoxHorizontal",
    "LTTextLine",
    "LTTextLineHorizontal",
    "LTTextLineVertical",
    "LTTextBoxVertical",
    "extract_pages",
    "extract_text",
    "extract_text_to_fp",
)

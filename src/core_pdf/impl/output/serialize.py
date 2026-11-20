# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from csv import writer
from functools import partial
from html import escape
from io import StringIO
from itertools import groupby
from typing import NamedTuple, Protocol
from xml.etree.ElementTree import Element, SubElement, tostring

from core_pdf.impl.base.pages import PageSelection, resolve_page_selection
from core_pdf.impl.base.types import GeneratedRecord
from core_pdf.impl.output.model import (
    Annotation,
    AssociatedFile,
    Block,
    BlockKind,
    ContentNode,
    Diagnostic,
    Document,
    Figure,
    FormField,
    JsonValue,
    Link,
    Page,
    PageElement,
    Table,
    TableAssociatedText,
    TableCell,
    TableColumnBand,
    TableRowBand,
    TextLine,
    TextSpan,
)

LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•▪◦]|(?:\d+|[^\W_])[.)])[ \t]*")


class PageIds:
    __slots__ = ("page_id",)

    def __init__(self, page_number: int) -> None:
        self.page_id = f"p{page_number}"

    def record(self, kind: str, index: int) -> str:
        return f"{self.page_id}:{kind}:{index}"

    def records(self, kind: str, count: int) -> list[JsonValue]:
        return [self.record(kind, index) for index in range(count)]

    def identities(self, kind: str, items: Iterable[object]) -> dict[int, str]:
        unique = dict.fromkeys(id(item) for item in items)
        return {identity: self.record(kind, index) for index, identity in enumerate(unique)}


def extend_records[RecordT](
    records: list[JsonValue],
    page_id: str,
    ids: Iterable[JsonValue],
    items: Iterable[RecordT],
    to_json: Callable[[RecordT], dict[str, JsonValue]],
) -> None:
    records.extend(
        {"id": record_id, "page_id": page_id, **to_json(item)}
        for record_id, item in zip(ids, items, strict=True)
    )


def node_to_json_dict(
    node: ContentNode, *, node_id: str, page_id: str, target_id: str
) -> dict[str, JsonValue]:
    return {
        "id": node_id,
        "page_id": page_id,
        "kind": node.kind,
        "target_id": target_id,
        "provenance": list(node.provenance),
    }


def diagnostic_to_json_dict(diagnostic: Diagnostic) -> dict[str, JsonValue]:
    return {
        "code": diagnostic.code,
        "message": diagnostic.message,
        "severity": diagnostic.severity,
        "page_number": diagnostic.page_number,
    }


def document_to_json_dict(document: Document) -> dict[str, JsonValue]:
    pages: list[JsonValue] = []
    nodes: list[JsonValue] = []
    blocks: list[JsonValue] = []
    lines: list[JsonValue] = []
    tables: list[JsonValue] = []
    figures: list[JsonValue] = []
    links: list[JsonValue] = []
    annotations: list[JsonValue] = []
    form_fields: list[JsonValue] = []
    seen_page_ids: set[str] = set()

    for page in document.pages:
        ids = PageIds(page.page_number)
        page_id = ids.page_id
        if page_id in seen_page_ids:
            raise ValueError(f"duplicate structured page id: {page_id}")
        seen_page_ids.add(page_id)

        page_blocks = {id(block): block for block in page.blocks}
        page_tables = {id(table): table for table in page.tables}
        page_figures = {id(figure): figure for figure in page.figures}
        block_ids = ids.identities("block", page_blocks.values())
        table_ids = ids.identities("table", page_tables.values())
        figure_ids = ids.identities("figure", page_figures.values())

        line_ids: dict[int, str] = {}
        for block in page_blocks.values():
            for line in block.lines:
                identity = id(line)
                if identity not in line_ids:
                    line_ids[identity] = ids.record("line", len(line_ids))
                    lines.append(
                        {"id": line_ids[identity], "page_id": page_id, **line_to_json_dict(line)}
                    )

        blocks.extend(
            {
                "id": block_ids[id(block)],
                "page_id": page_id,
                **block_payload(block),
                "line_ids": [line_ids[id(line)] for line in block.lines],
            }
            for block in page_blocks.values()
        )
        extend_records(
            tables, page_id, table_ids.values(), page_tables.values(), table_to_json_dict
        )
        extend_records(
            figures, page_id, figure_ids.values(), page_figures.values(), figure_to_json_dict
        )

        target_ids = {**block_ids, **table_ids, **figure_ids}
        page_node_ids: list[JsonValue] = []
        for index, node in enumerate(page.nodes):
            node_id = ids.record("node", index)
            page_node_ids.append(node_id)
            nodes.append(
                node_to_json_dict(
                    node, node_id=node_id, page_id=page_id, target_id=target_ids[id(node.payload)]
                )
            )

        page_link_ids = ids.records("link", len(page.links))
        page_annotation_ids = ids.records("annotation", len(page.annotations))
        page_field_ids = ids.records("form-field", len(page.form_fields))
        extend_records(links, page_id, page_link_ids, page.links, link_to_json_dict)
        extend_records(
            annotations,
            page_id,
            page_annotation_ids,
            page.annotations,
            partial(annotation_to_json_dict, ids=ids),
        )
        extend_records(form_fields, page_id, page_field_ids, page.form_fields, field_to_json_dict)
        pages.append(
            {
                "id": page_id,
                "page_number": page.page_number,
                "page_label": page.page_label,
                "width": page.width,
                "height": page.height,
                "user_unit": page.user_unit,
                "rotation": page.rotation,
                "cropbox": bbox_to_json(page.cropbox),
                "page_class": page.page_class,
                "base_route": page.base_route,
                "confidence": page.confidence,
                "node_ids": page_node_ids,
                "link_ids": page_link_ids,
                "annotation_ids": page_annotation_ids,
                "form_field_ids": page_field_ids,
                "header": page.header,
                "footer": page.footer,
                "diagnostics": [
                    diagnostic_to_json_dict(diagnostic) for diagnostic in page.diagnostics
                ],
            }
        )

    result: dict[str, JsonValue] = {
        "schema_version": document.schema_version,
        "metadata": json_safe(document.metadata, path="$.metadata"),
        "pages": pages,
        "nodes": nodes,
        "blocks": blocks,
        "lines": lines,
        "tables": tables,
        "figures": figures,
        "links": links,
        "annotations": annotations,
        "form_fields": form_fields,
        "diagnostics": [diagnostic_to_json_dict(diagnostic) for diagnostic in document.diagnostics],
    }
    # Present only when the document has associated files, so other output is unchanged.
    if document.associated_files:
        result["associated_files"] = [
            {"id": f"af:{index}", **associated_file_to_json_dict(associated)}
            for index, associated in enumerate(document.associated_files)
        ]
    return result


def reference_to_json(reference: tuple[int, int] | None) -> JsonValue:
    return None if reference is None else f"{reference[0]} {reference[1]} R"


def associated_file_to_json_dict(associated: AssociatedFile) -> dict[str, JsonValue]:
    page_number = associated.page_number
    page_id = None if page_number is None else f"p{page_number}"
    annotation_index = associated.annotation_index
    file = associated.file
    return {
        "owner": str(associated.owner),
        "page_id": page_id,
        "annotation_id": (
            None
            if page_number is None or annotation_index is None
            else f"{page_id}:annotation:{annotation_index}"
        ),
        "structure_type": associated.structure_type,
        "structure_id": associated.structure_id,
        "structure_ref": reference_to_json(associated.structure_reference),
        "resource_name": associated.resource_name,
        "relationship": file.relationship,
        "name": file.name,
        "file_name": file.file_name,
        "unicode_file_name": file.unicode_file_name,
        "description": file.description,
        "mime_type": file.mime_type,
        "size": file.size,
        "creation_date": file.creation_date,
        "modification_date": file.modification_date,
        "checksum": file.checksum,
        "embedded": file.embedded,
        "stream_ref": reference_to_json(file.stream_reference),
    }


def map_page_element[ElementResultT](
    element: PageElement,
    *,
    block: Callable[[Block], ElementResultT],
    table: Callable[[Table], ElementResultT],
    figure: Callable[[Figure], ElementResultT],
) -> ElementResultT:
    match element:
        case Block():
            return block(element)
        case Table():
            return table(element)
        case Figure():
            return figure(element)
    raise TypeError(f"unsupported page element: {type(element).__name__}")


def block_payload(block: Block) -> dict[str, JsonValue]:
    return {
        "order": block.order,
        "kind": block.kind.value,
        "bbox": bbox_to_json(block.bbox),
        "column_index": block.column_index,
        "rotation": block.rotation,
        "confidence": block.confidence,
        "level": block.level,
        "provenance": list(block.provenance),
    }


def span_to_json_dict(span: TextSpan) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {"text": span.text, **span.style_dict()}
    if span.expansion is not None:
        payload["expansion"] = span.expansion
    return payload


def line_to_json_dict(line: TextLine) -> dict[str, JsonValue]:
    return {
        "text": line.text,
        "break_before": line.break_before,
        "bbox": bbox_to_json(line.bbox),
        "advance_bbox": bbox_to_json(line.advance_bbox),
        "ink_bbox": bbox_to_json(line.ink_bbox),
        "kind": line.kind,
        "source": line.source,
        "confidence": line.confidence,
        **line.style_dict(),
        "spans": [span_to_json_dict(span) for span in line.styled_spans()],
        "baseline": bbox_to_json(line.baseline),
        "contributing_sources": list(line.contributing_sources),
    }


def table_to_json_dict(table: Table) -> dict[str, JsonValue]:
    return {
        "order": table.order,
        "bbox": bbox_to_json(table.bbox),
        "layout_bbox": bbox_to_json(table.layout_bbox),
        "content_bbox": bbox_to_json(table.content_bbox),
        "confidence": table.confidence,
        "title": table_associated_text_to_json_dict(table.title),
        "caption": table_associated_text_to_json_dict(table.caption),
        "row_bands": [table_row_band_to_json_dict(band) for band in table.row_bands],
        "column_bands": [table_column_band_to_json_dict(band) for band in table.column_bands],
        "metadata": json_safe(table.metadata),
        "rows": [[table_cell_to_json_dict(cell) for cell in row] for row in table.rows],
    }


def table_associated_text_to_json_dict(
    value: TableAssociatedText | None,
) -> dict[str, JsonValue] | None:
    if value is None:
        return None
    return {
        "text": value.text,
        "bbox": bbox_to_json(value.bbox),
        "kind": value.kind,
        "confidence": value.confidence,
    }


def table_row_band_to_json_dict(band: TableRowBand) -> dict[str, JsonValue]:
    return {
        "index": band.index,
        "bbox": bbox_to_json(band.bbox),
        "kind": band.kind,
        "confidence": band.confidence,
    }


def table_column_band_to_json_dict(band: TableColumnBand) -> dict[str, JsonValue]:
    return {"index": band.index, "bbox": bbox_to_json(band.bbox), "confidence": band.confidence}


def table_cell_to_json_dict(cell: TableCell) -> dict[str, JsonValue]:
    return {
        "row": cell.row,
        "column": cell.column,
        "text": cell.text,
        "row_span": cell.row_span,
        "column_span": cell.column_span,
        "bbox": bbox_to_json(cell.bbox),
    }


def figure_to_json_dict(figure: Figure) -> dict[str, JsonValue]:
    return {
        "order": figure.order,
        "bbox": bbox_to_json(figure.bbox),
        "kind": figure.kind,
        "metadata": json_safe(figure.metadata),
    }


def link_to_json_dict(link: Link) -> dict[str, JsonValue]:
    return {
        "bbox": bbox_to_json(link.bbox),
        "url": link.url,
        "link_type": link.link_type,
        "text": link.text,
    }


def annotation_to_json_dict(
    annotation: Annotation, *, ids: PageIds | None = None
) -> dict[str, JsonValue]:
    """An annotation's record. The annotations it links to are named by their ids on its
    page, ids, or without them by their indexes in Page.annotations."""

    def reference(index: int | None) -> JsonValue:
        if index is None or ids is None:
            return index
        return ids.record("annotation", index)

    return {
        "subtype": annotation.subtype,
        "bbox": bbox_to_json(annotation.bbox),
        "contents": annotation.contents,
        "destination": json_safe(annotation.destination),
        "in_reply_to": reference(annotation.in_reply_to),
        "reply_type": annotation.reply_type,
        "popup": reference(annotation.popup),
        "popup_parent": reference(annotation.popup_parent),
    }


def field_to_json_dict(field: FormField) -> dict[str, JsonValue]:
    return {
        "name": field.name,
        "field_type": field.field_type,
        "value_text": field.value_text,
        "bbox": bbox_to_json(field.bbox),
        "field_index": field.field_index,
        "required": field.required,
        "read_only": field.read_only,
        "no_export": field.no_export,
        "options": list(field.options),
        "alignment": field.alignment,
    }


def document_to_json(document: Document, *, indent: int | None, sort_keys: bool) -> str:
    return json.dumps(document_to_json_dict(document), indent=indent, sort_keys=sort_keys)


def bbox_to_json(bbox: tuple[float, float, float, float] | None) -> list[JsonValue] | None:
    return list(bbox) if bbox is not None else None


def selected_pages(document: Document, pages: PageSelection | None) -> tuple[Page, ...]:
    if pages is None:
        return document.pages
    indexes = resolve_page_selection(pages, len(document.pages))
    return tuple(document.pages[index] for index in indexes)


def iter_selected_lines(
    document: Document, pages: PageSelection | None
) -> Iterator[tuple[Page, tuple[TextLine, ...]]]:
    for page in selected_pages(document, pages):
        yield page, page.text_view.lines


def document_to_csv(document: Document, *, pages: PageSelection | None = None) -> str:
    output = StringIO()
    rows = writer(output, lineterminator="\n")
    rows.writerow(("page_number", "line_index", "text", "x0", "y0", "x1", "y1"))
    for page, lines in iter_selected_lines(document, pages):
        for index, line in enumerate(lines):
            bbox = line.bbox
            rows.writerow(
                (
                    page.page_number,
                    index,
                    line.text,
                    float(bbox[0]) if bbox is not None else "",
                    float(bbox[1]) if bbox is not None else "",
                    float(bbox[2]) if bbox is not None else "",
                    float(bbox[3]) if bbox is not None else "",
                )
            )
    return output.getvalue()


def document_to_tei(document: Document, *, pages: PageSelection | None = None) -> str:
    root = Element("TEI")
    text = SubElement(root, "text")
    body = SubElement(text, "body")
    for page, lines in iter_selected_lines(document, pages):
        SubElement(body, "pb", {"n": str(page.page_number)})
        for line in lines:
            paragraph = SubElement(body, "p")
            paragraph.text = line.text
    return tostring(root, encoding="unicode", short_empty_elements=True)


class InlineMarkup(GeneratedRecord):
    escape_text: bool
    strikeout: tuple[str, str]
    bold: tuple[str, str]
    italic: tuple[str, str]


MARKDOWN_MARKUP = InlineMarkup(False, ("~~", "~~"), ("**", "**"), ("*", "*"))
HTML_MARKUP = InlineMarkup(True, ("<del>", "</del>"), ("<strong>", "</strong>"), ("<em>", "</em>"))


# A layer of inline markup: its kind, and for an abbreviation its expansion and the span it
# marks, which keeps two abbreviations apart: layout gives each one span, its runs joined.
type InlineLayer = tuple[str, str, int]
# Outermost first, where layers opening together run on equally far.
LAYER_KINDS = ("abbr", "italic", "bold", "strikeout", "underline", "mark", "sup", "sub")
EMPHASIS_KINDS = frozenset(("italic", "bold", "strikeout"))
INLINE_TAGS = {
    "underline": ("<u>", "</u>"),
    "mark": ("<mark>", "</mark>"),
    "sup": ("<sup>", "</sup>"),
    "sub": ("<sub>", "</sub>"),
}
# CommonMark's whitespace (2.1): Unicode Zs, tab, line feed, form feed and carriage return.
COMMONMARK_SPACES = "\t\n\f\r \u00a0\u1680\u2000-\u200a\u202f\u205f\u3000"
COMMONMARK_SPACES_RE = re.compile(f"[{COMMONMARK_SPACES}]+")
INLINE_UNITS_RE = re.compile(f"[{COMMONMARK_SPACES}]+|[^{COMMONMARK_SPACES}]+")
# Text CommonMark would read as markup: escaped with a backslash in Markdown. An & only
# where it would start an entity or character reference.
MARKDOWN_SYNTAX_RE = re.compile(
    r"[\\*_~`<\[\]]|&(?=#[0-9]{1,7};|#[xX][0-9a-fA-F]{1,6};|[A-Za-z][A-Za-z0-9]{0,31};)"
)
# What starts a heading, block quote, list item, thematic break or setext underline at the
# start of a line; an ordered list marker is escaped at its . or ).
MARKDOWN_LINE_START = frozenset("#>+=-")
ORDERED_MARKER_RE = re.compile(r"^([0-9]{1,9})([.)])(?=[ \t]|$)")
# A heading's closing sequence: a run of # its text ends with, after a space or alone.
HEADING_CLOSING_RE = re.compile(r"(^|[ \t])(#+)$")
# The CommonMark list markers a list line's own marker is written as: a numbered marker
# stays, a bullet glyph becomes -, and a letter (or a longer number) stays text after -.
NUMBERED_MARKER_RE = re.compile(r"[0-9]{1,9}[.)]")
BULLET_MARKERS = frozenset("-*•▪◦")
# An inline token: ("text", text, unit index) or ("open" or "close", layer, instance).
type InlineToken = tuple[str, InlineLayer | str, int]


def is_space(text: str) -> bool:
    return COMMONMARK_SPACES_RE.fullmatch(text) is not None


def span_layers(span: TextSpan, index: int) -> tuple[InlineLayer, ...]:
    """The markup a span is wrapped in, in LAYER_KINDS order."""
    flags = (
        span.expansion is not None,
        span.italic,
        span.bold,
        span.strikeout,
        span.underline,
        span.mark,
        span.superscript,
        span.subscript and not span.superscript,
    )
    return tuple(
        (kind, span.expansion or "", index) if kind == "abbr" else (kind, "", -1)
        for kind, flag in zip(LAYER_KINDS, flags, strict=True)
        if flag
    )


def layer_markers(layer: InlineLayer, markup: InlineMarkup) -> tuple[str, str]:
    kind = layer[0]
    if kind == "abbr":
        # An abbreviation and its expansion (ISO 32000-2 14.9.5), as HTML marks one.
        return f'<abbr title="{escape(layer[1], quote=True)}">', "</abbr>"
    if kind in EMPHASIS_KINDS:
        markers: tuple[str, str] = getattr(markup, kind)
        return markers
    return INLINE_TAGS[kind]


def styled_units(
    line: TextLine, markup: InlineMarkup, start: int
) -> list[tuple[str, tuple[InlineLayer, ...]]]:
    """The line's words and runs of whitespace, each with the layers it sits in.

    Whitespace inside a span keeps the span's layers, and so does a span of spaces only,
    a fill-in blank say. Whitespace at a span's end keeps only the layers the unit beyond
    it shares, so markup that ends there closes before it and markup that starts there
    opens after it, while markup running on across it holds it.
    """
    pieces: list[tuple[str, tuple[InlineLayer, ...]]] = []
    for index, span in enumerate(line.styled_spans()):
        text = span.text
        if start:
            text = text[start:]
            start = max(0, start - len(span.text))
            if not text:
                continue
        pieces.append((escape(text) if markup.escape_text else text, span_layers(span, index)))
    units: list[tuple[str, tuple[InlineLayer, ...]]] = []
    edges: list[int] = []  # -1 a span's leading whitespace, 1 its trailing, 0 neither
    for text, layers in pieces:
        parts = INLINE_UNITS_RE.findall(text)
        blank = len(parts) == 1 and is_space(parts[0])
        for position, part in enumerate(parts):
            units.append((part, layers))
            edge = 0
            if is_space(part) and not blank:
                edge = -1 if position == 0 else 1 if position == len(parts) - 1 else 0
            edges.append(edge)
    return [
        (
            text,
            tuple(
                layer
                for layer in layers
                if edge == 0 or 0 <= index + edge < len(units) and layer in units[index + edge][1]
            ),
        )
        for index, ((text, layers), edge) in enumerate(zip(units, edges, strict=True))
    ]


def inline_tokens(
    units: list[tuple[str, tuple[InlineLayer, ...]]], markup: InlineMarkup
) -> list[InlineToken]:
    """Open and close each layer around the units it holds, keeping it open across them.

    Layers opening together nest by how far they run on, the longest outermost. In
    Markdown, whitespace opens no emphasis, which cannot start or end with whitespace, and
    whitespace before emphasis closes moves after its marker.
    """
    markdown = not markup.escape_text
    tokens: list[InlineToken] = []
    stack: list[tuple[InlineLayer, int]] = []
    instances = 0

    def reach(layer: InlineLayer, index: int) -> int:
        count = 0
        for _, later in units[index + 1 :]:
            if layer not in later:
                break
            count += 1
        return count

    def close_to(keep: int) -> None:
        for layer, instance in reversed(stack[keep:]):
            moved: list[InlineToken] = []
            if markdown and layer[0] in EMPHASIS_KINDS:
                while tokens and isinstance(tokens[-1][1], str) and is_space(tokens[-1][1]):
                    moved.append(tokens.pop())
            tokens.append(("close", layer, instance))
            tokens.extend(reversed(moved))
        del stack[keep:]

    for index, (text, layers) in enumerate(units):
        keep = 0
        while keep < len(stack) and stack[keep][0] in layers:
            keep += 1
        kept = [layer for layer, _ in stack[:keep]]
        if markdown and is_space(text):
            layers = tuple(
                layer for layer in layers if layer[0] not in EMPHASIS_KINDS or layer in kept
            )
        close_to(keep)
        opening = sorted(
            (layer for layer in layers if layer not in kept),
            key=lambda layer: reach(layer, index),
            reverse=True,
        )
        for layer in opening:
            tokens.append(("open", layer, instances))
            stack.append((layer, instances))
            instances += 1
        tokens.append(("text", text, index))
    close_to(0)
    return tokens


def flanking(before: str, after: str) -> tuple[bool, bool]:
    """Whether a delimiter run between these characters is left- and right-flanking
    (CommonMark 6.2); an empty string is the line's edge."""

    def space(character: str) -> bool:
        return not character or is_space(character)

    def punctuation(character: str) -> bool:
        return bool(character) and unicodedata.category(character)[0] in "PS"

    left = not space(after) and (not punctuation(after) or space(before) or punctuation(before))
    right = not space(before) and (not punctuation(before) or space(after) or punctuation(after))
    return left, right


def escape_markdown(text: str, *, line_start: bool) -> str:
    text = MARKDOWN_SYNTAX_RE.sub(lambda match: f"\\{match.group()}", text)
    if not line_start:
        return text
    if text[:1] in MARKDOWN_LINE_START:
        return f"\\{text}"
    return ORDERED_MARKER_RE.sub(r"\1\\\2", text, count=1)


def escaped_tokens(tokens: list[InlineToken]) -> list[InlineToken]:
    """The tokens with their text escaped for Markdown, text between two markers joined,
    and the line's text escaped as a line start unless markup comes first.

    Spaces and tabs at the line's ends go: a paragraph line loses them anyway, and kept
    they would open an indented code block (four columns at the start) or a hard line
    break (two spaces at the end).
    """
    joined: list[InlineToken] = []
    for token in tokens:
        if joined and isinstance(token[1], str) and isinstance(joined[-1][1], str):
            joined[-1] = (joined[-1][0], joined[-1][1] + token[1], joined[-1][2])
        else:
            joined.append(token)
    if joined and isinstance(joined[0][1], str):
        joined[0] = (joined[0][0], joined[0][1].lstrip(" \t"), joined[0][2])
    if joined and isinstance(joined[-1][1], str):
        joined[-1] = (joined[-1][0], joined[-1][1].rstrip(" \t"), joined[-1][2])
    return [
        (kind, escape_markdown(value, line_start=index == 0), unit)
        if isinstance(value, str)
        else (kind, value, unit)
        for index, (kind, value, unit) in enumerate(joined)
        if value != ""
    ]


def render_tokens(tokens: list[InlineToken], markup: InlineMarkup, tagged: set[int]) -> list[str]:
    rendered: list[str] = []
    for kind, value, instance in tokens:
        if isinstance(value, str):
            rendered.append(value)
            continue
        markers = layer_markers(value, HTML_MARKUP if instance in tagged else markup)
        rendered.append(markers[0] if kind == "open" else markers[1])
    return rendered


def is_delimiter(token: InlineToken, tagged: set[int]) -> bool:
    """Whether the token is a Markdown emphasis marker, not written as an HTML tag."""
    value = token[1]
    return not isinstance(value, str) and value[0] in EMPHASIS_KINDS and token[2] not in tagged


class DelimiterRun:
    """A run of one Markdown emphasis character, as CommonMark matches it."""

    __slots__ = ("can_close", "can_open", "character", "end", "left", "length", "start")

    def __init__(
        self, character: str, start: int, end: int, length: int, flanks: tuple[bool, bool]
    ):
        self.character = character
        self.start, self.end = start, end  # its first token and the token after it
        self.length = length
        self.left = length  # characters not yet matched
        self.can_open, self.can_close = flanks


def misread_at(tokens: list[InlineToken], markup: InlineMarkup, tagged: set[int]) -> int | None:
    """The first token CommonMark would read with other emphasis than its own, if any.

    This follows CommonMark's delimiter runs and their matching (6.2 and appendix A, its
    rule of 3 included, strikethrough pairing as markdown-it's), and compares the emphasis
    each word ends up in with the emphasis written around it; a delimiter left unmatched
    reads as text.
    """
    rendered = render_tokens(tokens, markup, tagged)
    runs: list[DelimiterRun] = []
    index = 0
    while index < len(tokens):
        if not is_delimiter(tokens[index], tagged):
            index += 1
            continue
        character = rendered[index][0]
        end = index + 1
        while (
            end < len(tokens)
            and is_delimiter(tokens[end], tagged)
            and rendered[end][0] == character
        ):
            end += 1
        # Neither words nor markers are empty.
        before = rendered[index - 1][-1] if index else ""
        after = rendered[end][0] if end < len(rendered) else ""
        length = sum(len(rendered[at]) for at in range(index, end))
        runs.append(DelimiterRun(character, index, end, length, flanking(before, after)))
        index = end
    pairs: list[tuple[int, int, str]] = []
    for closer_index, closer in enumerate(runs):
        while closer.can_close and closer.left:
            for opener_index in range(closer_index - 1, -1, -1):
                opener = runs[opener_index]
                if opener.character != closer.character or not opener.can_open or not opener.left:
                    continue
                if (
                    closer.character == "*"
                    and (opener.can_close or closer.can_open)
                    and (opener.length + closer.length) % 3 == 0
                    and (opener.length % 3 or closer.length % 3)
                ):
                    continue
                break
            else:
                break
            if closer.character == "~":
                count, kind = 2, "strikeout"
            else:
                count = 2 if opener.left >= 2 and closer.left >= 2 else 1
                kind = "bold" if count == 2 else "italic"
            opener.left -= count
            closer.left -= count
            pairs.append((opener.end, closer.start, kind))
            # Delimiters between the two read as text.
            for between in runs[opener_index + 1 : closer_index]:
                if between.character == closer.character:
                    between.can_open = between.can_close = False
    if unmatched := [run.start for run in runs if run.left]:
        return unmatched[0]
    written: Counter[str] = Counter()
    for at, token in enumerate(tokens):
        side, value, _ = token
        if isinstance(value, str):
            if is_space(value):
                continue
            read = {style for first, end, style in pairs if first <= at < end}
            if read != {style for style, count in written.items() if count}:
                return at
        elif is_delimiter(token, tagged):
            written[value[0]] += 1 if side == "open" else -1
    return None


def render_styled_line(line: TextLine, markup: InlineMarkup, *, start: int = 0) -> str:
    # A line keeps a span per text run, and runs split at every font or glyph-run change,
    # so neighbours often share markup: a layer stays open across the spans that share it,
    # so `UN` in two bold runs is `**UN**`, not `**U****N**`.
    tokens = inline_tokens(styled_units(line, markup, start), markup)
    tagged: set[int] = set()
    if not markup.escape_text:
        tokens = escaped_tokens(tokens)
        # Emphasis CommonMark would misread is written as HTML, which Markdown reads as
        # inline HTML: the one whose marker is nearest the first word misread, till none is.
        while (at := misread_at(tokens, markup, tagged)) is not None:
            nearest = min(
                (abs(position - at), position)
                for position, token in enumerate(tokens)
                if is_delimiter(token, tagged)
            )[1]
            tagged.add(tokens[nearest][2])
    return "".join(render_tokens(tokens, markup, tagged))


def markdown_line(line: TextLine, *, start: int = 0) -> str:
    return render_styled_line(line, MARKDOWN_MARKUP, start=start)


def html_line(line: TextLine, *, start: int = 0) -> str:
    return render_styled_line(line, HTML_MARKUP, start=start)


class ElementFormatter(Protocol):
    def format_block(self, block: Block) -> str: ...

    def format_table(self, table: Table) -> str: ...

    def format_figure(self, figure: Figure) -> str: ...

    def format_page(self, page: Page) -> str: ...

    def format_document(self, document: Document) -> str: ...


def format_elements(formatter: ElementFormatter, page: Page) -> list[str]:
    return [
        map_page_element(
            element,
            block=formatter.format_block,
            table=formatter.format_table,
            figure=formatter.format_figure,
        )
        for element in page.elements
    ]


class MarkdownFormatter:
    __slots__ = ()

    def format_block(self, block: Block) -> str:
        return block_to_markdown(block)

    def format_table(self, table: Table) -> str:
        return table_to_html(table)

    def format_figure(self, figure: Figure) -> str:
        return f"> [Figure: {figure.kind}]"

    def format_page(self, page: Page) -> str:
        # Neighbouring lists alternate their markers, - and *, . and ): a blank line alone
        # would join them into one loose list. An element rendered empty is left out, so
        # lists around it are neighbours.
        rendered: list[str] = []
        previous_list: bool | None = None  # the last list's alternate, None after another
        for element in page.elements:
            if isinstance(element, Block) and element.kind is BlockKind.LIST:
                alternate = previous_list is False
                text = block_to_markdown(element, alternate=alternate)
                if text:
                    rendered.append(text)
                    previous_list = alternate
            else:
                text = map_page_element(
                    element,
                    block=self.format_block,
                    table=self.format_table,
                    figure=self.format_figure,
                )
                if text:
                    rendered.append(text)
                    previous_list = None
        return "\n\n".join(rendered)

    def format_document(self, document: Document) -> str:
        # A form feed on a line of its own between blank lines: CommonMark reads it as a
        # paragraph of whitespace, so a page's first block starts a line and a list
        # ending the page before ends there.
        return "".join(f"{self.format_page(page)}\n\n\f\n\n" for page in document.pages)


class HtmlFormatter:
    __slots__ = ()

    def format_block(self, block: Block) -> str:
        return block_to_html(block)

    def format_table(self, table: Table) -> str:
        return table_to_html(table)

    def format_figure(self, figure: Figure) -> str:
        return f'<figure data-kind="{escape(figure.kind)}"></figure>'

    def format_page(self, page: Page) -> str:
        rendered = "\n".join(format_elements(self, page))
        return f'<section data-page-number="{page.page_number}">{rendered}</section>'

    def format_document(self, document: Document) -> str:
        pages = "\n".join(self.format_page(page) for page in document.pages)
        return f'<article data-schema-version="{escape(document.schema_version)}">{pages}</article>'


MARKDOWN_FORMATTER = MarkdownFormatter()
HTML_FORMATTER = HtmlFormatter()


def document_to_markdown(document: Document) -> str:
    return MARKDOWN_FORMATTER.format_document(document)


def page_to_markdown(page: Page) -> str:
    return MARKDOWN_FORMATTER.format_page(page)


class ListItem(NamedTuple):
    """A list line, the number its marker gives it (its digits as written, None for a
    bullet), and where its text starts."""

    line: TextLine
    number: str | None
    start: int


def list_item(line: TextLine) -> ListItem:
    """What a list line's marker makes it, in Markdown and HTML alike.

    A numbered marker gives the item its number, and a bullet glyph makes it a bullet. A
    marker counts only before whitespace or the line's end, so `1.5 million` and
    `-5 degrees` are text. A letter marker, or none, stays text in a bullet item:
    CommonMark has no letter lists, and HTML's type="a" would lose `a)` against `a.`,
    `A)` or `α)`.
    """
    prefix = list_prefix(line.text)
    marker = prefix.strip()
    # The prefix takes the spaces after its marker: none there, the line must end.
    ends = prefix.endswith((" ", "\t")) or len(prefix) == len(line.text)
    if ends and NUMBERED_MARKER_RE.fullmatch(marker):
        return ListItem(line, marker[:-1], len(prefix))
    if ends and marker in BULLET_MARKERS:
        return ListItem(line, None, len(prefix))
    return ListItem(line, None, 0)


def list_runs(block: Block) -> list[list[ListItem]]:
    """A list block's items in the lists CommonMark reads its Markdown as: a run of
    numbered items is an ordered list starting at the first one's number, a run of
    bullets a bullet list, and a change from one to the other starts the next list."""
    return [
        list(run)
        for _, run in groupby(map(list_item, block.lines), key=lambda item: item.number is None)
    ]


def markdown_list_item(line: TextLine, *, alternate: bool = False) -> tuple[str, int]:
    """The CommonMark list marker a list line is written with, and where its text starts:
    a number keeps its digits, with . (or ) when alternate), and a bullet is - (or *)."""
    _, number, start = list_item(line)
    if number is not None:
        return f"{number}{')' if alternate else '.'} ", start
    return "* " if alternate else "- ", start


def heading_level(block: Block) -> int:
    """A heading's level: 2 when the block has none, else its level within 1 to 6, the
    levels of both Markdown's ATX headings and HTML's h1 to h6."""
    return 2 if block.level is None else min(max(block.level, 1), 6)


def heading_text(block: Block, render: Callable[[TextLine], str]) -> str:
    """A heading's lines rendered and joined into its one line, as an ATX heading cannot
    break; a line whose text is blank (CommonMark's whitespace, no-break spaces among it,
    or none) is left out, whatever its markup."""
    return " ".join(render(line) for line in block.lines if line.text and not is_space(line.text))


def block_to_markdown(block: Block, *, alternate: bool = False) -> str:
    if block.kind is BlockKind.LIST:
        return "\n".join(
            f"{marker}{markdown_line(line, start=start)}"
            for line in block.lines
            for marker, start in (markdown_list_item(line, alternate=alternate),)
        )
    if block.kind is BlockKind.HEADING:
        # An ATX heading is one line, and a run of # ending it would close it.
        text = HEADING_CLOSING_RE.sub(r"\1\\\2", heading_text(block, markdown_line))
        return f"{'#' * heading_level(block)} {text}"
    # A line of spaces only would end the paragraph.
    return "\n".join(text for line in block.lines if (text := markdown_line(line)))


def document_to_html(document: Document) -> str:
    return HTML_FORMATTER.format_document(document)


def page_to_html(page: Page) -> str:
    return HTML_FORMATTER.format_page(page)


def block_to_html(block: Block) -> str:
    """The block's HTML, one element per block, read back with the structure of its
    Markdown: a heading at the same level on one line, and a list block as the lists
    list_runs finds, an <ol> or a <ul> each, in a <div> when there are several."""
    attributes = f' data-block-kind="{escape(block.kind.value)}"'

    if block.kind is BlockKind.HEADING:
        tag = f"h{heading_level(block)}"
        heading = heading_text(block, lambda line: html_line(line).strip(" \t"))
        return f"<{tag}{attributes}>{heading}</{tag}>"
    if block.kind is BlockKind.LIST:
        runs = list_runs(block)
        # The block's kind goes on its one element: its list, or the <div> around them.
        inner = attributes if len(runs) <= 1 else ""
        lists: list[str] = []
        for run in runs:
            items = "".join(f"<li>{html_line(item.line, start=item.start)}</li>" for item in run)
            number = run[0].number
            if number is None:
                lists.append(f"<ul{inner}>{items}</ul>")
            else:
                start = f' start="{int(number)}"' if int(number) != 1 else ""
                lists.append(f"<ol{inner}{start}>{items}</ol>")
        if not lists:
            return f"<ul{attributes}></ul>"
        if len(lists) == 1:
            return lists[0]
        return f"<div{attributes}>{''.join(lists)}</div>"
    text = "<br />".join(html_line(line) for line in block.lines)
    return f"<p{attributes}>{text}</p>"


def table_to_html(table: Table) -> str:
    if not table.rows:
        return "<table></table>"
    associated_text = tuple(value for value in (table.title, table.caption) if value is not None)
    prefix = "".join(
        f'<div data-table-associated="{escape(value.kind)}">{escape(value.text)}</div>'
        for value in associated_text
    )
    bands = table.row_bands
    if not bands or (
        not associated_text
        and bands[0].kind == "header"
        and all(band.kind == "body" for band in bands[1:])
    ):
        header, *body = table.rows
        head = "".join(table_cell_to_html(cell, header=True) for cell in header)
        body_html = "".join(
            f"<tr>{''.join(table_cell_to_html(cell) for cell in row)}</tr>" for row in body
        )
        return f"{prefix}<table><thead><tr>{head}</tr></thead><tbody>{body_html}</tbody></table>"

    header_rows: list[str] = []
    body_rows: list[str] = []
    for row, band in zip(table.rows, bands, strict=True):
        if band.kind in {"title", "caption"}:
            continue
        cells = "".join(table_cell_to_html(cell, header=band.kind == "header") for cell in row)
        rendered = f'<tr data-row-kind="{escape(band.kind)}">{cells}</tr>'
        (header_rows if band.kind == "header" else body_rows).append(rendered)
    head_html = f"<thead>{''.join(header_rows)}</thead>" if header_rows else ""
    body_html = f"<tbody>{''.join(body_rows)}</tbody>" if body_rows else ""
    return f"{prefix}<table>{head_html}{body_html}</table>"


def table_cell_to_html(cell: TableCell, *, header: bool = False) -> str:
    tag = "th" if header else "td"
    spans = "".join(
        (
            f' rowspan="{cell.row_span}"' if cell.row_span > 1 else "",
            f' colspan="{cell.column_span}"' if cell.column_span > 1 else "",
        )
    )
    return f"<{tag}{spans}>{escape(cell.text)}</{tag}>"


def list_prefix(text: str) -> str:
    match = LIST_PREFIX_RE.match(text)
    return match.group(0) if match is not None else ""


def json_safe(value: object, *, path: str = "$") -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"unsupported JSON metadata key at {path}: {type(key).__name__}")
            result[key] = json_safe(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [json_safe(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"unsupported JSON metadata value at {path}: {type(value).__name__}")


__all__ = (
    "block_to_html",
    "block_to_markdown",
    "annotation_to_json_dict",
    "field_to_json_dict",
    "figure_to_json_dict",
    "document_to_csv",
    "document_to_html",
    "document_to_json",
    "document_to_json_dict",
    "document_to_markdown",
    "document_to_tei",
    "json_safe",
    "page_to_html",
    "page_to_markdown",
    "table_cell_to_json_dict",
    "table_to_html",
    "table_to_json_dict",
)

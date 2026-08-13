"""High-level pypdf-shaped APIs backed by core-pdf."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from difflib import SequenceMatcher
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, cast

from core_pdf import PdfDocument
from core_pdf.impl.engine.layout.geometry import rect_tuple
from core_pdf.impl.engine.spec.s_07_content.operations import iter_content_operations
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.engine.structured import (
    Annotation,
    Document,
    Link,
    Page,
)
from core_pdf.impl.engine.writing.encryption import StandardPdfEncryption
from core_pdf.impl.engine.writing.semantic import serialize_document_to_pdf
from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf.impl.objects import PdfReference, PdfStream, PdfString

PdfInput = str | PathLike[str] | bytes | bytearray | BytesIO
BBox = tuple[float, float, float, float]
GraphicsState = tuple[list[float], str | None, float, float]


def _validate_pypdf_page_tree(pdf: PdfDocument) -> None:
    """Preserve pypdf's rejection of repeated/cyclic intermediate page nodes."""
    seen: set[tuple[int, int]] = set()

    def visit(value: object) -> None:
        if isinstance(value, PdfReference):
            key = (value.object_number, value.generation_number)
            if key in seen:
                raise ValueError("detected cyclic page references")
            seen.add(key)
        node = pdf.resolver.resolve(value)
        if not isinstance(node, dict):
            return
        kids = pdf.resolver.resolve(lookup_dict_key(node, "Kids"))
        if isinstance(kids, (list, tuple)):
            for kid in kids:
                visit(kid)

    visit(lookup_dict_key(pdf.catalog(), "Pages"))


class ClosingMixin:
    def close(self) -> None:
        return None

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def coerce_bbox(value: object) -> BBox:
    box = rect_tuple(value)
    if box is None:
        raise ValueError(f"value does not describe a rectangle: {value!r}")
    return box


def write_bytes(target: str | PathLike[str] | BinaryIO, data: bytes) -> None:
    if isinstance(target, (str, PathLike)):
        Path(target).write_bytes(data)
    else:
        target.write(data)


def internal_pypdf_space_width(
    font: Mapping[object, object], resolver: Any, decoder: FontDecoder
) -> float:
    """Match pypdf's simple-font fallback when code 32 has no declared width."""
    space_width = decoder.glyph_width(32)
    first_char = lookup_dict_key(font, "FirstChar")
    raw_widths = resolver.resolve(lookup_dict_key(font, "Widths"))
    if not isinstance(first_char, (int, float)) or not isinstance(raw_widths, (list, tuple)):
        return space_width
    space_index = 32 - int(first_char)
    if 0 <= space_index < len(raw_widths):
        return space_width
    positive_widths = [float(width) for width in raw_widths if float(width) > 0]
    if not positive_widths:
        return space_width
    return float(int(sum(positive_widths) / len(positive_widths)) // 2)


def internal_cid_space_width(decoder: FontDecoder, data: bytes) -> float | None:
    if not decoder.is_cid_font:
        return None
    space_glyph = next(
        (glyph for glyph in decoder.decode_glyphs(data) if glyph.unicode == " "),
        None,
    )
    return None if space_glyph is None else decoder.glyph_width(space_glyph.width_code)


def internal_decoded_width(decoder: FontDecoder | None, data: bytes) -> float:
    if decoder is None:
        return 0.0
    return sum(decoder.glyph_width(decoded.width_code) for decoded in decoder.decode_glyphs(data))


def internal_operand_text(
    data: bytes, cmap: ToUnicodeCMap | None, decoder: FontDecoder | None
) -> str:
    text = cmap.decode(data) if cmap is not None else ""
    if not text and decoder is not None:
        text = "".join(glyph.unicode for glyph in decoder.decode_glyphs(data))
    if not text and len(data) % 2 == 0 and data[::2].count(0) >= len(data) // 4:
        text = data.decode("utf-16-be", errors="replace")
    return text or data.decode("latin-1")


def internal_restore_graphics_state(stack: list[GraphicsState]) -> GraphicsState:
    if stack:
        return stack.pop()
    return ([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], None, 0.0, 250.0)


def internal_pypdf_rtl_order(text: str) -> str:
    output: list[str] = []
    rtl_run: list[str] = []
    for character in text:
        if unicodedata.bidirectional(character) in {"R", "AL", "AN"}:
            rtl_run.append(character)
            continue
        if rtl_run:
            output.extend(reversed(rtl_run))
            rtl_run.clear()
        output.append(character)
    output.extend(reversed(rtl_run))
    return "".join(output).replace("’ :", ": ’")


class StructuredState(ClosingMixin):
    """Facade-local ownership of an engine document or synthetic structured snapshot."""

    def __init__(self, pdf: PdfDocument | None, structured: Document) -> None:
        self.pdf = pdf
        self.structured = structured
        self.internal_projection: PdfDocument | None = None

    @classmethod
    def open(cls, source: PdfInput, *, password: str = "") -> "StructuredState":
        pdf = PdfDocument.open(source, password=password)
        _validate_pypdf_page_tree(pdf)
        return cls(pdf, pdf.structured_document)

    @classmethod
    def from_structured(cls, structured: Document) -> "StructuredState":
        pdf = PdfDocument.from_structured(structured)
        return cls(pdf, pdf.structured_document)

    @classmethod
    def synthetic(cls, structured: Document) -> "StructuredState":
        return cls(None, structured)

    @property
    def source_pdf(self) -> PdfDocument:
        if self.pdf is None:
            raise ValueError("synthetic snapshots do not have a source PDF")
        return self.pdf

    @property
    def snapshot(self) -> Document:
        return self.structured

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.structured.metadata

    @property
    def pages(self) -> tuple[Page, ...]:
        return self.structured.pages

    @property
    def form_fields(self) -> tuple[Any, ...]:
        return tuple(field for page in self.pages for field in page.form_fields)

    def capability_document(self) -> PdfDocument:
        if self.pdf is not None and self.structured is self.pdf.structured_document:
            return self.pdf
        if self.internal_projection is None:
            self.internal_projection = PdfDocument.from_structured(self.structured)
        return self.internal_projection

    def capability_page(self, page_number: int) -> Any:
        return self.capability_document().pages[page_number - 1]

    def _with_pages(self, pages: Sequence[Page]) -> "StructuredState":
        return StructuredState.synthetic(replace(self.structured, pages=tuple(pages)))

    def replace_pages(self, pages: Sequence[Page]) -> "StructuredState":
        return self._with_pages(pages)

    def delete_page(self, page_number: int) -> "StructuredState":
        return self._with_pages(
            tuple(page for index, page in enumerate(self.pages, 1) if index != page_number)
        )

    def update_metadata(self, values: Mapping[str, Any]) -> "StructuredState":
        return StructuredState.synthetic(
            replace(self.structured, metadata={**self.structured.metadata, **values})
        )

    def apply_redactions(self) -> "StructuredState":
        redactions = tuple(
            annotation.bbox
            for page in self.pages
            for annotation in page.annotations
            if (annotation.subtype or "").casefold() == "redact" and annotation.bbox
        )

        def covered(box: tuple[float, float, float, float] | None) -> bool:
            return bool(
                box
                and any(
                    box[0] >= redaction[0]
                    and box[1] >= redaction[1]
                    and box[2] <= redaction[2]
                    and box[3] <= redaction[3]
                    for redaction in redactions
                )
            )

        return self._with_pages(
            tuple(
                replace(
                    page,
                    blocks=tuple(block for block in page.blocks if not covered(block.bbox)),
                )
                for page in self.pages
            )
        )

    def write_redacted(
        self,
        target: str | PathLike[str] | BinaryIO,
        *,
        outlines: Sequence[Sequence[object]] | None = None,
        attachments: Mapping[str, bytes] | None = None,
    ) -> bytes:
        data = serialize_document_to_pdf(
            self.apply_redactions().structured,
            outlines=outlines,
            attachments=attachments,
        )
        write_bytes(target, data)
        return data

    def close(self) -> None:
        if self.internal_projection is not None:
            self.internal_projection.close()
        if self.pdf is not None:
            self.pdf.close()


class Destination:
    """Small pypdf-shaped bookmark destination."""

    def __init__(self, title: str, page: int | None, level: int = 0) -> None:
        self.title = title
        self.page = page
        self.level = level

    @property
    def typ(self) -> str:
        return "/Fit"


class Rectangle(tuple[float, float, float, float]):
    """Tuple-compatible pypdf rectangle with the common geometry accessors."""

    def __new__(cls, left: float, bottom: float, right: float, top: float) -> "Rectangle":
        return super().__new__(cls, (float(left), float(bottom), float(right), float(top)))

    @property
    def left(self) -> float:
        return self[0]

    @property
    def bottom(self) -> float:
        return self[1]

    @property
    def right(self) -> float:
        return self[2]

    @property
    def top(self) -> float:
        return self[3]

    @property
    def lower_left(self) -> tuple[float, float]:
        return (self.left, self.bottom)

    @property
    def lower_right(self) -> tuple[float, float]:
        return (self.right, self.bottom)

    @property
    def upper_left(self) -> tuple[float, float]:
        return (self.left, self.top)

    @property
    def upper_right(self) -> tuple[float, float]:
        return (self.right, self.top)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom


class PdfPageObject:
    def __init__(self, document: StructuredState, page: Page) -> None:
        self._document = document
        self._page = page
        self.internal_text_override: str | None = None
        self.mediabox = Rectangle(0, 0, page.width, page.height)
        self.cropbox = Rectangle(*(page.cropbox or self.mediabox))
        self.rotation = page.rotation

    def _capability_view(self) -> Any:
        return self._document.capability_page(self._page.page_number).structured_view

    def extract_text(self, *args: object, **kwargs: object) -> str:  # noqa: C901
        del args, kwargs
        if self.internal_text_override is not None:
            return self.internal_text_override
        # pypdf only interprets PDF text-showing operators. Core-pdf's structured
        # view may additionally contain OCR; exposing that here would make the
        # compatibility facade more capable, but observably unlike pypdf.
        source_page = self._document.pages[self._page.page_number - 1]
        if self._document.pdf is not None and self._page is source_page:
            page = self._document.capability_page(self._page.page_number)
            glyphs = page.get_page_program().products.glyphs
            resources = page.resolve_resources()
            raw_fonts = page.document.resolver.resolve(lookup_dict_key(resources, "Font"))
            font_cmaps: dict[str, ToUnicodeCMap] = {}
            resource_cmaps: dict[str, ToUnicodeCMap] = {}
            resource_decoders: dict[str, FontDecoder] = {}
            resource_font_names: dict[str, str] = {}
            resource_space_thresholds: dict[str, float] = {}
            font_decoders: dict[str, FontDecoder] = {}
            font_space_thresholds: dict[str, float] = {}
            fonts_with_differences: set[str] = set()
            resources_with_differences: set[str] = set()
            if isinstance(raw_fonts, dict):
                for resource_name, raw_font in raw_fonts.items():
                    font = page.document.resolver.resolve(raw_font)
                    if not isinstance(font, dict):
                        continue
                    base_font = lookup_dict_key(font, "BaseFont")
                    to_unicode = page.document.resolver.resolve(lookup_dict_key(font, "ToUnicode"))
                    if base_font is None:
                        continue
                    resolved_font = page.document.resolver.resolve_font_dict(font)
                    decoder = FontDecoder(cast(dict[str, object], resolved_font))
                    resource_decoders[str(resource_name)] = decoder
                    resource_font_names[str(resource_name)] = str(base_font)
                    space_width = internal_pypdf_space_width(font, page.document.resolver, decoder)
                    resource_space_thresholds[str(resource_name)] = space_width * 0.475
                    font_decoders[str(base_font)] = decoder
                    font_space_thresholds[str(base_font)] = space_width * 0.5
                    if not isinstance(to_unicode, PdfStream):
                        continue
                    stream = page.document.resolver.resolve_stream(to_unicode)
                    font_name = str(base_font)
                    cmap = ToUnicodeCMap(stream.data)
                    font_cmaps[font_name] = cmap
                    resource_cmaps[str(resource_name)] = cmap
                    encoding = page.document.resolver.resolve(lookup_dict_key(font, "Encoding"))
                    if isinstance(encoding, dict) and lookup_dict_key(encoding, "Differences"):
                        fonts_with_differences.add(font_name)
                        resources_with_differences.add(str(resource_name))
            show_resources: list[str | None] = []
            show_bytes: list[bytes] = []
            current_font_resource: str | None = None
            for stream in page.content_streams:
                for operator, operands in iter_content_operations(PdfLexer(stream.data)):
                    if operator == "Tf":
                        if not operands:
                            continue
                        current_font_resource = str(operands[0])
                    elif operator == "Tj":
                        show_resources.append(current_font_resource)
                        value = operands[0]
                        show_bytes.append(
                            bytes(value.data) if isinstance(value, PdfString) else b""
                        )
                    elif operator == "TJ":
                        for tj_item in cast(list[object], operands[0]):
                            if isinstance(tj_item, PdfString):
                                show_resources.append(current_font_resource)
                                show_bytes.append(bytes(tj_item.data))
            for resource_name, data in zip(show_resources, show_bytes, strict=True):
                resource_decoder = resource_decoders.get(resource_name or "")
                cid_space_width = (
                    internal_cid_space_width(resource_decoder, data)
                    if resource_decoder is not None
                    else None
                )
                if cid_space_width is not None:
                    resource_space_thresholds[resource_name or ""] = cid_space_width * 0.475
                    font_space_thresholds[resource_font_names[resource_name or ""]] = (
                        cid_space_width * 0.5
                    )
            sequence_bytes: dict[int, bytearray] = {}
            for glyph in glyphs:
                sequence_bytes.setdefault(glyph.seqno, bytearray()).extend(glyph.code_bytes)
            comparable_sequences = [
                seqno for seqno in sequence_bytes if seqno < len(show_bytes) and show_bytes[seqno]
            ][:200]
            matching_sequences = sum(
                bytes(sequence_bytes[seqno]) == show_bytes[seqno] for seqno in comparable_sequences
            )
            direct_sequence_fonts = (not glyphs or glyphs[-1].seqno < len(show_resources)) and (
                not comparable_sequences or matching_sequences / len(comparable_sequences) >= 0.8
            )
            ordered_sequences = sorted(sequence_bytes)
            compact_comparisons = min(len(ordered_sequences), len(show_bytes), 200)
            compact_matches = sum(
                bytes(sequence_bytes[seqno]) == show_bytes[index]
                for index, seqno in enumerate(ordered_sequences[:compact_comparisons])
            )
            compact_sequence_fonts = compact_comparisons > 0 and (
                compact_matches / compact_comparisons >= 0.8
            )
            sequence_show_indices: dict[int, int] = {}
            show_cursor = 0
            for compact_index, seqno in enumerate(ordered_sequences):
                if direct_sequence_fonts and seqno < len(show_bytes):
                    sequence_show_indices[seqno] = seqno
                    continue
                if compact_sequence_fonts and compact_index < len(show_bytes):
                    sequence_show_indices[seqno] = compact_index
                    continue
                target = bytes(sequence_bytes[seqno])
                while show_cursor < len(show_bytes) and show_bytes[show_cursor] != target:
                    show_cursor += 1
                if show_cursor < len(show_bytes):
                    sequence_show_indices[seqno] = show_cursor
                    show_cursor += 1
            groups: list[tuple[int, float, float, float, float, float, str, int]] = []
            group_text_widths: dict[int, float] = {}
            group_space_widths: dict[int, float] = {}
            cid_width_groups: set[int] = set()
            previous_glyph: Any | None = None
            previous_decoded = ""
            rendered_glyphs: list[tuple[Any, str]] = []
            for glyph in glyphs:
                show_index = sequence_show_indices.get(glyph.seqno)
                resource_name = show_resources[show_index] if show_index is not None else None
                glyph_cmap = resource_cmaps.get(resource_name or "") or font_cmaps.get(
                    glyph.font_name
                )
                glyph_text = (
                    glyph_cmap.decode(glyph.code_bytes) if glyph_cmap is not None else glyph.text
                )
                glyph_decoder = font_decoders.get(glyph.font_name)
                cid_width_groups.update(
                    (glyph.seqno,)
                    if glyph_decoder is not None
                    and glyph_decoder.is_cid_font
                    and glyph.font_name.endswith("+")
                    else ()
                )
                group_text_widths[glyph.seqno] = group_text_widths.get(
                    glyph.seqno, 0.0
                ) + internal_decoded_width(glyph_decoder, glyph.code_bytes)
                group_space_widths[glyph.seqno] = font_space_thresholds.get(glyph.font_name, 0.0)
                has_differences = (
                    resource_name in resources_with_differences
                    if direct_sequence_fonts
                    else glyph.font_name in fonts_with_differences
                )
                if has_differences and glyph.code_bytes in {b"'", b"\x15"}:
                    glyph_text = glyph.text
                core_ligature_pair = (
                    previous_glyph is not None
                    and previous_glyph.seqno == glyph.seqno
                    and previous_glyph.code_bytes == glyph.code_bytes
                    and glyph.code_bytes not in {b"f", b"\x00f", b"\x00I"}
                    and cast(Any, previous_glyph).text + glyph.text in {"ff", "fi", "fl"}
                    and glyph_text not in {"ff", "fi", "fl"}
                )
                if core_ligature_pair:
                    pair_text = cast(Any, previous_glyph).text + glyph.text
                    glyph_text = {"ff": "ﬀ", "fi": "ﬁ", "fl": "ﬂ"}[pair_text]
                duplicate_ligature_part = (
                    previous_glyph is not None
                    and previous_glyph.seqno == glyph.seqno
                    and previous_glyph.code_bytes == glyph.code_bytes
                    and glyph_text == previous_decoded
                    and glyph_text in {"ff", "fi", "fl", "ﬀ", "ﬁ", "ﬂ", "ﬃ", "ﬄ"}
                )
                if duplicate_ligature_part:
                    glyph_text = ""
                if not groups or groups[-1][0] != glyph.seqno:
                    groups.append(
                        (
                            glyph.seqno,
                            glyph.advance_bbox[0],
                            glyph.advance_bbox[2],
                            glyph.advance_bbox[1],
                            glyph.advance_bbox[3],
                            glyph.font_size,
                            glyph_text,
                            len(glyph.code_bytes),
                        )
                    )
                else:
                    seqno, x0, _x1, y0, y1, font_size, text, byte_count = groups[-1]
                    if core_ligature_pair and text.endswith("ﬀ") and glyph.text in {"i", "l"}:
                        text = text[:-1]
                        glyph_text = "ﬃ" if glyph.text == "i" else "ﬄ"
                    elif core_ligature_pair:
                        text = text[:-1]
                    groups[-1] = (
                        seqno,
                        x0,
                        glyph.advance_bbox[2],
                        y0,
                        y1,
                        font_size,
                        text + glyph_text,
                        byte_count + len(glyph.code_bytes),
                    )
                rendered_glyphs.append((glyph, glyph_text))
                previous_glyph = glyph
                previous_decoded = (
                    glyph_cmap.decode(glyph.code_bytes) if glyph_cmap is not None else glyph.text
                )
            flat_glyph_bytes = b"".join(glyph.code_bytes for glyph, _ in rendered_glyphs)
            flat_show_bytes = b"".join(show_bytes)
            byte_mapping: dict[int, int] = {}
            for match in SequenceMatcher(
                None, flat_glyph_bytes, flat_show_bytes
            ).get_matching_blocks():
                byte_mapping.update(
                    (match.a + offset, match.b + offset) for offset in range(match.size)
                )
            show_ends: list[int] = []
            show_end = 0
            for data in show_bytes:
                show_end += len(data)
                show_ends.append(show_end)
            aligned_groups: list[tuple[int, float, float, float, float, float, str, int]] = []
            aligned_show_indices: list[int] = []
            aligned_engine_seqnos: list[int] = []
            glyph_offset = 0
            for glyph, glyph_text in rendered_glyphs:
                mapped = [
                    byte_mapping.get(offset)
                    for offset in range(glyph_offset, glyph_offset + len(glyph.code_bytes))
                ]
                glyph_offset += len(glyph.code_bytes)
                if any(offset is None for offset in mapped):
                    continue
                show_index = next(
                    (index for index, end in enumerate(show_ends) if cast(int, mapped[0]) < end),
                    None,
                )
                if show_index is None or cast(int, mapped[-1]) >= show_ends[show_index]:
                    continue
                if not aligned_groups or aligned_show_indices[-1] != show_index:
                    aligned_groups.append(
                        (
                            glyph.seqno,
                            glyph.advance_bbox[0],
                            glyph.advance_bbox[2],
                            glyph.advance_bbox[1],
                            glyph.advance_bbox[3],
                            glyph.font_size,
                            glyph_text,
                            len(glyph.code_bytes),
                        )
                    )
                    aligned_show_indices.append(show_index)
                    aligned_engine_seqnos.append(glyph.seqno)
                else:
                    group = aligned_groups[-1]
                    aligned_groups[-1] = (
                        *group[:2],
                        glyph.advance_bbox[2],
                        *group[3:6],
                        group[6] + glyph_text,
                        group[7] + len(glyph.code_bytes),
                    )
            lost_show_boundaries = any(
                len(
                    {
                        show
                        for engine_seqno, show in zip(aligned_engine_seqnos, aligned_show_indices)
                        if engine_seqno == seqno
                    }
                )
                > 1
                for seqno in sequence_bytes
            )
            byte_aligned_groups = False
            if (
                lost_show_boundaries
                or (
                    not direct_sequence_fonts
                    and not compact_sequence_fonts
                    and abs(len(flat_glyph_bytes) - len(flat_show_bytes)) <= 2
                )
            ) and len(aligned_groups) >= len(groups) * 0.9:
                groups = [
                    (show_index, *group[1:])
                    for group, show_index in zip(aligned_groups, aligned_show_indices)
                ]
                sequence_show_indices = {
                    show_index: show_index for show_index in aligned_show_indices
                }
                direct_sequence_fonts = True
                byte_aligned_groups = True
            if not direct_sequence_fonts and not compact_sequence_fonts:
                aligned_shows_by_sequence: dict[int, set[int]] = {}
                for engine_seqno, show_index in zip(
                    aligned_engine_seqnos, aligned_show_indices, strict=True
                ):
                    aligned_shows_by_sequence.setdefault(engine_seqno, set()).add(show_index)
                for index, group in enumerate(groups):
                    aligned_shows = aligned_shows_by_sequence.get(group[0], set())
                    if len(aligned_shows) != 1:
                        continue
                    show_index = next(iter(aligned_shows))
                    resource_name = show_resources[show_index] or ""
                    sequence_show_indices[group[0]] = show_index
                    if resource_name not in resource_cmaps:
                        continue
                    aligned_text = internal_operand_text(
                        show_bytes[show_index],
                        resource_cmaps.get(resource_name),
                        resource_decoders.get(resource_name),
                    )
                    groups[index] = (*group[:6], aligned_text, *group[7:])
            if not direct_sequence_fonts and not compact_sequence_fonts:
                synthetic_groups = []
                for left_group, right_group in zip(groups, groups[1:]):
                    left_seqno, right_seqno = left_group[0], right_group[0]
                    left_show = sequence_show_indices.get(left_seqno)
                    right_show = sequence_show_indices.get(right_seqno)
                    if (
                        left_show is None
                        or right_show is None
                        or right_seqno - left_seqno != right_show - left_show
                    ):
                        continue
                    for offset, show_index in enumerate(range(left_show + 1, right_show), 1):
                        missing_seqno = left_seqno + offset
                        missing_cmap = resource_cmaps.get(show_resources[show_index] or "")
                        missing_data = show_bytes[show_index]
                        missing_text = (
                            missing_cmap.decode(missing_data) if missing_cmap is not None else ""
                        ) or missing_data.decode("latin-1")
                        synthetic_groups.append(
                            (
                                missing_seqno,
                                left_group[2],
                                right_group[1],
                                left_group[3],
                                left_group[4],
                                left_group[5],
                                missing_text,
                                len(missing_data),
                            )
                        )
                        sequence_show_indices[missing_seqno] = show_index
                groups.extend(synthetic_groups)
                groups.sort()
            for index, group in enumerate(groups):
                show_index = sequence_show_indices.get(group[0])
                if show_index is None:
                    continue
                compact_cmap = resource_cmaps.get(show_resources[show_index] or "")
                compact_text = (
                    compact_cmap.decode(show_bytes[show_index]) if compact_cmap is not None else ""
                )
                if compact_text.strip() == group[6]:
                    groups[index] = (*group[:6], compact_text, *group[7:])
            if direct_sequence_fonts:
                groups_by_sequence = {group[0]: group for group in groups}
                previous_group = None
                for seqno, data in enumerate(show_bytes):
                    if seqno in groups_by_sequence:
                        previous_group = groups_by_sequence[seqno]
                        continue
                    sequence_cmap = resource_cmaps.get(show_resources[seqno] or "")
                    raw_text = (
                        sequence_cmap.decode(data) if sequence_cmap is not None else ""
                    ) or data.decode("latin-1")
                    if previous_group is None or not (
                        any(ord(character) <= 1 for character in raw_text)
                        or raw_text in {"ﬀ", "ﬁ", "ﬂ", "ﬃ", "ﬄ"}
                    ):
                        continue
                    _, _x0, x1, y0, y1, font_size, _text, _byte_count = previous_group
                    groups_by_sequence[seqno] = (
                        seqno,
                        x1,
                        x1,
                        y0,
                        y1,
                        font_size,
                        raw_text,
                        len(data),
                    )
                    previous_group = groups_by_sequence[seqno]
                groups = sorted(groups_by_sequence.values())
            for index, group in enumerate(groups):
                seqno = group[0]
                if not direct_sequence_fonts or seqno >= len(show_bytes):
                    continue
                sequence_cmap = resource_cmaps.get(show_resources[seqno] or "")
                data = show_bytes[seqno]
                raw_text = (
                    sequence_cmap.decode(data) if sequence_cmap is not None else ""
                ) or data.decode("latin-1")
                if any(ord(character) <= 1 for character in raw_text):
                    groups[index] = (*group[:6], raw_text, *group[7:])
            output = ""
            previous_y0: float | None = None
            previous_y1: float | None = None
            previous_x1: float | None = None
            previous_seqno: int | None = None
            previous_resource: str | None = None
            show_metadata: list[tuple[str, bool, bool, bool]] = []
            show_positions: list[tuple[float, float]] = []
            show_width_points: list[float] = []
            show_halfspaces: list[float] = []
            moved_since_show = True
            current_font_resource = None
            pending_tj_space = False
            matrix_space_pending = False
            matrix_newline_pending = False
            cm_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            tm_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            previous_cm = cm_matrix.copy()
            previous_tm = tm_matrix.copy()
            current_font_size = 0.0
            current_space_width = 250.0
            accumulated_width = 0.0
            graphics_stack: list[GraphicsState] = []
            malformed_text_state = False

            def multiply(left: list[float], right: list[float]) -> list[float]:
                return [
                    left[0] * right[0] + left[1] * right[2],
                    left[0] * right[1] + left[1] * right[3],
                    left[2] * right[0] + left[3] * right[2],
                    left[2] * right[1] + left[3] * right[3],
                    left[4] * right[0] + left[5] * right[2] + right[4],
                    left[4] * right[1] + left[5] * right[3] + right[5],
                ]

            def moved_far_enough(width: float) -> bool:
                previous = multiply(previous_tm, previous_cm)
                current = multiply(tm_matrix, cm_matrix)
                delta_x = current[4] - previous[4]
                delta_y = current[5] - previous[5]
                scale_x = math.sqrt(previous_tm[0] ** 2 + previous_tm[1] ** 2)
                scale_y = math.sqrt(previous_tm[2] ** 2 + previous_tm[3] ** 2)
                current_scale_y = math.sqrt(tm_matrix[2] ** 2 + tm_matrix[3] ** 2)
                if abs(delta_y) > 0.8 * min(
                    current_font_size * scale_y,
                    current_font_size * current_scale_y,
                ):
                    return False
                space_width = current_font_size * current_space_width / 1000.0
                return delta_x >= (space_width + width / 1000.0) * scale_x

            def moved_to_newline() -> bool:
                previous = multiply(previous_tm, previous_cm)
                current = multiply(tm_matrix, cm_matrix)
                delta_y = current[5] - previous[5]
                scale_y = math.sqrt(previous_tm[2] ** 2 + previous_tm[3] ** 2)
                current_scale_y = math.sqrt(tm_matrix[2] ** 2 + tm_matrix[3] ** 2)
                return abs(delta_y) > 0.8 * min(
                    current_font_size * scale_y,
                    current_font_size * current_scale_y,
                )

            def shown_width(data: bytes) -> float:
                decoder = resource_decoders.get(current_font_resource or "")
                if decoder is None:
                    return 0.0
                space_width = (
                    resource_space_thresholds.get(current_font_resource or "", 237.5) / 0.475
                )
                return (
                    sum(
                        (
                            decoder.default_width
                            if glyph.unicode in {"ff", "fi", "fl", "ffi", "ffl"}
                            else space_width
                            if glyph.unicode == " "
                            else decoder.glyph_width(glyph.width_code)
                        )
                        for glyph in decoder.decode_glyphs(data)
                    )
                    * current_font_size
                )

            for stream in page.content_streams:
                for operator, operands in iter_content_operations(PdfLexer(stream.data)):
                    if operator == "q":
                        graphics_stack.append(
                            (
                                cm_matrix.copy(),
                                current_font_resource,
                                current_font_size,
                                current_space_width,
                            )
                        )
                    elif operator == "Q":
                        (
                            cm_matrix,
                            current_font_resource,
                            current_font_size,
                            current_space_width,
                        ) = internal_restore_graphics_state(graphics_stack)
                    elif operator == "BT":
                        tm_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                        moved_since_show = True
                    elif operator in {"Td", "TD"}:
                        tx = float(cast(Any, operands[0]))
                        ty = float(cast(Any, operands[1]))
                        tm_matrix[4] += tx * tm_matrix[0] + ty * tm_matrix[2]
                        tm_matrix[5] += tx * tm_matrix[1] + ty * tm_matrix[3]
                        matrix_newline_pending = moved_to_newline()
                        matrix_space_pending = moved_far_enough(accumulated_width)
                        accumulated_width = 0.0
                        previous_tm = tm_matrix.copy()
                        previous_cm = cm_matrix.copy()
                        moved_since_show = True
                    elif operator == "Tm":
                        values = [float(cast(Any, value)) for value in operands[:6]]
                        tm_matrix = values if len(values) == 6 else [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                        matrix_newline_pending = moved_to_newline()
                        matrix_space_pending = moved_far_enough(accumulated_width)
                        accumulated_width = 0.0
                        previous_tm = tm_matrix.copy()
                        previous_cm = cm_matrix.copy()
                        moved_since_show = True
                    elif operator == "cm":
                        cm_matrix = multiply(
                            [float(cast(Any, value)) for value in operands[:6]], cm_matrix
                        )
                    elif operator == "Tf":
                        if len(operands) < 2:
                            malformed_text_state = True
                            continue
                        current_font_resource = str(operands[0])
                        current_font_size = float(cast(Any, operands[1]))
                        current_space_width = (
                            resource_space_thresholds.get(current_font_resource, 237.5) / 0.475
                        ) / 2.0
                    elif operator == "Tj":
                        value = operands[0]
                        data = bytes(value.data) if isinstance(value, PdfString) else b""
                        show_metadata.append(
                            (
                                "Tj",
                                moved_since_show,
                                pending_tj_space or matrix_space_pending,
                                matrix_newline_pending,
                            )
                        )
                        shown_position = multiply(tm_matrix, cm_matrix)
                        show_positions.append((shown_position[4], shown_position[5]))
                        show_width_points.append(shown_width(data) / 1000.0)
                        show_halfspaces.append(current_font_size * current_space_width / 1000.0)
                        moved_since_show = False
                        pending_tj_space = False
                        matrix_space_pending = False
                        matrix_newline_pending = False
                        accumulated_width += shown_width(data)
                        previous_tm = tm_matrix.copy()
                        previous_cm = cm_matrix.copy()
                    elif operator == "TJ":
                        force_space_before = pending_tj_space or matrix_space_pending
                        pending_tj_space = False
                        matrix_space_pending = False
                        force_newline_before = matrix_newline_pending
                        matrix_newline_pending = False
                        space_threshold = resource_space_thresholds.get(
                            current_font_resource or "", 237.5
                        )
                        for tj_item in cast(list[object], operands[0]):
                            if isinstance(tj_item, PdfString):
                                show_metadata.append(
                                    (
                                        "TJ",
                                        moved_since_show,
                                        force_space_before,
                                        force_newline_before,
                                    )
                                )
                                shown_position = multiply(tm_matrix, cm_matrix)
                                show_positions.append((shown_position[4], shown_position[5]))
                                show_width_points.append(shown_width(bytes(tj_item.data)) / 1000.0)
                                show_halfspaces.append(
                                    current_font_size * current_space_width / 1000.0
                                )
                                moved_since_show = False
                                force_space_before = False
                                force_newline_before = False
                                accumulated_width += shown_width(bytes(tj_item.data))
                                previous_tm = tm_matrix.copy()
                                previous_cm = cm_matrix.copy()
                            elif isinstance(tj_item, (int, float)):
                                if (
                                    abs(float(tj_item)) >= space_threshold
                                    and not force_space_before
                                ):
                                    force_space_before = True
                                    accumulated_width += (
                                        current_font_size * current_space_width * 2.0
                                    )
                                    previous_tm = tm_matrix.copy()
                                    previous_cm = cm_matrix.copy()
                        pending_tj_space = force_space_before
            direct_sequence_metadata = direct_sequence_fonts and (
                not groups
                or all(
                    sequence_show_indices.get(group[0], len(show_metadata)) < len(show_metadata)
                    for group in groups
                )
            )
            byte_coverage = len(flat_glyph_bytes) / max(1, len(flat_show_bytes))
            operand_output_override: str | None = None
            if byte_coverage < 0.95 and len(show_metadata) == len(show_bytes):
                operand_output = ""
                show_geometry = {
                    show_index: group
                    for group, show_index in zip(aligned_groups, aligned_show_indices)
                }
                for show_index, data in enumerate(show_bytes):
                    _operator, _moved, add_space, add_newline = show_metadata[show_index]
                    resource_name = show_resources[show_index] or ""
                    operand_cmap = resource_cmaps.get(resource_name)
                    operand_decoder = resource_decoders.get(resource_name)
                    operand_text = operand_cmap.decode(data) if operand_cmap is not None else ""
                    decoder_text = (
                        "".join(glyph.unicode for glyph in operand_decoder.decode_glyphs(data))
                        if operand_decoder is not None
                        else ""
                    )
                    if any(ord(character) < 32 for character in decoder_text):
                        operand_text = decoder_text
                    operand_text = operand_text or decoder_text or data.decode("latin-1")
                    operand_text = operand_text.replace("ᐌ", "\x00").replace("Ȑ", "\x00")
                    if add_newline and operand_output and not operand_output.endswith("\n"):
                        operand_output += "\n"
                    previous_direction = (
                        unicodedata.bidirectional(operand_output[-1]) if operand_output else ""
                    )
                    next_direction = (
                        unicodedata.bidirectional(operand_text[0]) if operand_text else ""
                    )
                    previous_geometry = show_geometry.get(show_index - 1)
                    current_geometry = show_geometry.get(show_index)
                    geometry_space = (
                        previous_geometry is not None
                        and current_geometry is not None
                        and current_geometry[1] - previous_geometry[2]
                        > max(
                            current_geometry[5] * 0.24,
                            (current_geometry[4] - current_geometry[3]) * 0.24,
                        )
                    )
                    operand_boundary_space = False
                    if show_index > 0:
                        delta_x = show_positions[show_index][0] - show_positions[show_index - 1][0]
                        delta_y = show_positions[show_index][1] - show_positions[show_index - 1][1]
                        operand_boundary_space = abs(delta_y) < 1.0 and delta_x >= (
                            show_width_points[show_index - 1] + show_halfspaces[show_index - 1]
                        )
                        if operand_text[:1] in {"ம", "'"}:  # pragma: no cover - diagnostic
                            operand_boundary_space = delta_x >= (
                                show_width_points[show_index - 1]
                                + show_halfspaces[show_index - 1] * 0.8
                            )
                    if (
                        (add_space or geometry_space or operand_boundary_space)
                        and (next_direction != "ON" or operand_boundary_space)
                        and not (previous_direction in {"R", "AL", "AN"} and next_direction == "L")
                        and not (previous_direction == "ON" and next_direction in {"R", "AL", "AN"})
                        and operand_output
                        and not operand_output.endswith((" ", "\n"))
                    ):
                        operand_output += " "
                    operand_output += operand_text
                operand_output_override = internal_pypdf_rtl_order(operand_output)
            for group_index, (seqno, x0, x1, y0, y1, font_size, text, _) in enumerate(groups):
                # Page-program sequence numbers map directly to text-show operands for
                # ordinary page streams. Form XObjects have their own nested streams,
                # though, so retain the compact positional fallback used for those runs.
                metadata_index = sequence_show_indices.get(
                    seqno, seqno if direct_sequence_metadata else group_index
                )
                if metadata_index >= len(show_metadata):
                    if not show_metadata:
                        break
                    operator, moved_before_show, force_space_before, force_newline_before = (
                        "TJ",
                        False,
                        False,
                        False,
                    )
                else:
                    (
                        operator,
                        moved_before_show,
                        force_space_before,
                        force_newline_before,
                    ) = show_metadata[metadata_index]
                group_resource = (
                    show_resources[metadata_index] if metadata_index < len(show_resources) else None
                )
                degenerate = y0 == y1
                vertically_separate = (
                    previous_y0 is not None
                    and previous_y1 is not None
                    and (y1 <= previous_y0 or y0 >= previous_y1)
                )
                predicted_sparse_space = False
                effective_force_space = force_space_before and (
                    direct_sequence_metadata
                    or (
                        force_space_before
                        and (
                            text[:1] in {",", ";", ":"}
                            or (
                                text.startswith(" ")
                                and not text.startswith("  ")
                                and bool(text.strip())
                                and previous_x1 is not None
                                and x0 - previous_x1 > (y1 - y0)
                            )
                        )
                    )
                    or previous_seqno in cid_width_groups
                    or output[-1:].isdigit()
                    or ord(text[:1] or "\0") >= 0xE000
                    or ord(output[-1:] or "\0") < 32
                    or (direct_sequence_fonts and text.startswith(" "))
                )
                literal_gap_space = (
                    byte_aligned_groups
                    and text.startswith(" ")
                    and previous_x1 is not None
                    and x0 - previous_x1 >= 0.5
                )
                missing_resource_space = (
                    compact_sequence_fonts
                    and metadata_index > 0
                    and metadata_index < len(show_positions)
                    and resource_decoders.get(show_resources[metadata_index] or "") is None
                    and abs(
                        show_positions[metadata_index][1] - show_positions[metadata_index - 1][1]
                    )
                    < 1.0
                    and show_positions[metadata_index][0] - show_positions[metadata_index - 1][0]
                    > 0.5
                )
                operand_position_space = (
                    compact_sequence_fonts
                    and font_size == 1
                    and y0 < 60
                    and (ord(output[-1:] or "\0") < 32 or group_resource != previous_resource)
                    and metadata_index > 0
                    and metadata_index < len(show_positions)
                    and abs(
                        show_positions[metadata_index][1] - show_positions[metadata_index - 1][1]
                    )
                    < 1.0
                    and show_positions[metadata_index][0] - show_positions[metadata_index - 1][0]
                    >= show_width_points[metadata_index - 1] + show_halfspaces[metadata_index - 1]
                )
                compact_font_boundary_space = (
                    compact_sequence_fonts
                    and moved_before_show
                    and font_size > 100
                    and group_resource != previous_resource
                    and resource_space_thresholds.get(group_resource or "", 0.0) > 300.0
                    and previous_x1 is not None
                    and x0 - previous_x1 >= 2.5
                )
                newline_after_text = False
                baseline_newline = (
                    force_newline_before
                    and previous_y0 is not None
                    and (
                        abs(y0 - previous_y0) > (y1 - y0) * 0.5
                        or previous_x1 is not None
                        and x0 < previous_x1 - (y1 - y0)
                        or malformed_text_state
                        and previous_x1 is not None
                        and abs(x0 - previous_x1) < (y1 - y0)
                    )
                    or (
                        group_resource is None
                        and previous_x1 is not None
                        and x0 < previous_x1 - (y1 - y0)
                        and text.isupper()
                    )
                )
                if baseline_newline and not vertically_separate and output.endswith("\n"):
                    pass
                elif (vertically_separate or baseline_newline) and not degenerate:
                    output += "\n"
                    if (
                        not direct_sequence_metadata
                        and previous_seqno is not None
                        and seqno - previous_seqno > 1
                        and not text.strip()
                    ):
                        newline_after_text = True
                elif (
                    previous_x1 is not None
                    and bool(text)
                    and (
                        not output.endswith((" ", "\n"))
                        or predicted_sparse_space
                        or effective_force_space
                        or literal_gap_space
                        or missing_resource_space
                        or operand_position_space
                        or compact_font_boundary_space
                    )
                    and (
                        not text.startswith(" ")
                        or effective_force_space
                        and not text.startswith("  ")
                        or literal_gap_space
                    )
                    and (
                        effective_force_space
                        or literal_gap_space
                        or missing_resource_space
                        or operand_position_space
                        or compact_font_boundary_space
                        or (
                            operator == "Tj"
                            and not moved_before_show
                            and output.rsplit("\n", 1)[-1].count("\x00") < 2
                            and x0 > previous_x1
                        )
                        or (
                            not moved_before_show
                            and x0 - previous_x1 > max(font_size * 0.24, (y1 - y0) * 0.24)
                        )
                        or predicted_sparse_space
                        or (
                            not direct_sequence_metadata
                            and moved_before_show
                            and (
                                previous_seqno not in cid_width_groups
                                or output.rsplit("\n", 1)[-1].count("\x00") >= 2
                            )
                            and x0 - previous_x1 > (y1 - y0) * 0.2
                        )
                        or (
                            y0 < 60
                            and output.rsplit("\n", 1)[-1].count("\x00") < 2
                            and (
                                output[-1:].isdigit()
                                and text[:1].isdigit()
                                and x0 - previous_x1 > 0.5
                            )
                        )
                    )
                ):
                    output += " "
                if output.endswith("ﬂ") and text.startswith(" ") and len(text) > 1:
                    output += " "
                output += text
                if newline_after_text:
                    output += "\n"
                if not degenerate:
                    previous_y0 = y0
                    previous_y1 = y1
                previous_x1 = x1
                previous_seqno = seqno
                previous_resource = group_resource
            glyph_output = internal_pypdf_rtl_order(output)
            if operand_output_override is not None:
                return operand_output_override
            return glyph_output
        return "\n".join(
            line.text for block in self._page.blocks for line in block.lines if line.source != "ocr"
        )

    def rotate(self, angle: int) -> PdfPageObject:
        self._page = replace(self._page, rotation=(self.rotation + angle) % 360)
        self.rotation = self._page.rotation
        return self

    def scale_to(self, width: float, height: float) -> PdfPageObject:
        self._page = replace(self._page, width=width, height=height)
        self.mediabox = Rectangle(0, 0, width, height)
        self.cropbox = self.mediabox
        return self

    def set_cropbox(self, box: tuple[float, float, float, float]) -> PdfPageObject:
        rectangle = Rectangle(*box)
        self._page = replace(self._page, cropbox=coerce_bbox(tuple(rectangle)))
        self.cropbox = rectangle
        return self

    def transfer_rotation_to_content(self) -> None:
        """Apply the page rotation to its geometry and clear the page rotation."""
        if self.rotation % 180:
            self.mediabox = Rectangle(0, 0, self.mediabox.height, self.mediabox.width)
            self.cropbox = Rectangle(0, 0, self.cropbox.height, self.cropbox.width)
        self._page = replace(self._page, rotation=0)
        self.rotation = 0

    def merge_page(self, page: PdfPageObject | Page) -> PdfPageObject:
        base_text = self.extract_text()
        overlay_text = page.extract_text() if isinstance(page, PdfPageObject) else page.text
        overlay = page._page if isinstance(page, PdfPageObject) else page
        order = max((item.order for item in self._page.elements), default=-1) + 1
        blocks = tuple(replace(item, order=order + item.order) for item in overlay.blocks)
        tables = tuple(replace(item, order=order + item.order) for item in overlay.tables)
        figures = tuple(replace(item, order=order + item.order) for item in overlay.figures)
        self._page = replace(
            self._page,
            blocks=(*self._page.blocks, *blocks),
            tables=(*self._page.tables, *tables),
            figures=(*self._page.figures, *figures),
            links=(*self._page.links, *overlay.links),
            annotations=(*self._page.annotations, *overlay.annotations),
            form_fields=(*self._page.form_fields, *overlay.form_fields),
        )
        self.internal_text_override = "\n".join(filter(None, (base_text, overlay_text)))
        return self

    @property
    def annotations(self) -> tuple[Any, ...]:
        return self._page.annotations or self._capability_view().annotations

    @property
    def links(self) -> tuple[Any, ...]:
        return self._page.links or self._capability_view().links

    @property
    def form_fields(self) -> tuple[Any, ...]:
        return self._page.form_fields or self._capability_view().form_fields

    @property
    def images(self) -> tuple[Any, ...]:
        return tuple(self._document.capability_page(self._page.page_number).extract_images())


class PdfReader(ClosingMixin):
    """Reader exposing the most-used pypdf page and metadata APIs."""

    def __init__(
        self,
        stream: PdfInput,
        password: str | None = None,
        strict: bool = True,
        **kwargs: object,
    ) -> None:
        del strict, kwargs
        self._encrypted_source = stream.getvalue() if isinstance(stream, BytesIO) else stream
        try:
            document = StructuredState.open(self._encrypted_source, password=password or "")
        except PdfUnsupportedError:
            if password:
                raise
            self._document = cast(Any, None)
            self.pages: tuple[PdfPageObject, ...] = ()
            self.metadata: dict[str, Any] = {}
            self.trailer: dict[str, Any] = {}
            self._decryption_pending = True
            return
        self._decryption_pending = False
        self._bind(document)
        self.trailer = {}

    def _bind(self, document: StructuredState) -> None:
        """Materialize page objects and merge engine info metadata from ``document``."""
        self._document = document
        self.pages = tuple(PdfPageObject(document, page) for page in document.pages)
        raw_metadata = document.source_pdf.get_metadata()
        self.metadata = (
            dict(cast(Any, raw_metadata.get("info", {}))) if isinstance(raw_metadata, dict) else {}
        )
        self.metadata.update(document.metadata)

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    def get_page(self, page_number: int) -> PdfPageObject:
        return self.pages[page_number]

    def get_page_number(self, page: PdfPageObject | Page) -> int:
        """Return the zero-based index for a page owned by this reader."""
        value = page._page if isinstance(page, PdfPageObject) else page
        candidate: PdfPageObject
        for index, candidate in enumerate(self.pages):
            if candidate._page is value or candidate._page == value:
                return index
        raise ValueError("page is not part of this reader")

    @property
    def outline(self) -> list[Destination]:
        if self._document is None:
            return []
        return [
            Destination(item.title, item.page_index, item.level)
            for item in self._document.source_pdf.iter_outlines()
        ]

    @property
    def outlines(self) -> list[Destination]:
        return self.outline

    def get_destination_page_number(self, destination: object) -> int:
        if isinstance(destination, Destination) and destination.page is not None:
            return destination.page
        raise ValueError("destination does not resolve to a page")

    def get_fields(self) -> dict[str, Any]:
        if self._document is None:
            return {}
        return {
            item.record.name: item.record
            for item in self._document.source_pdf.extract_form_fields()
        }

    def get_form_text_fields(self) -> dict[str, str]:
        if self._document is None:
            return {}
        return {
            field.name: field.value_text
            for item in self._document.source_pdf.extract_form_fields()
            for field in (item.record,)
            if field.type.casefold() in {"text", "tx"}
        }

    def get_form_xfa(self) -> None:
        return None

    def close(self) -> None:
        if self._document is not None:
            self._document.close()

    @property
    def is_encrypted(self) -> bool:
        return self._decryption_pending or bool(
            getattr(getattr(self._document, "pdf", None), "decipher", None)
        )

    def decrypt(self, password: str) -> bool:
        if not self._decryption_pending:
            return not self.is_encrypted
        try:
            document = StructuredState.open(self._encrypted_source, password=password)
        except PdfUnsupportedError:
            return False
        self._decryption_pending = False
        self._bind(document)
        return True


class PdfWriter:
    """Writer for structured pages and pages copied from ``PdfReader``."""

    def __init__(self) -> None:
        self._pages: list[Page] = []
        self.metadata: dict[str, Any] = {}
        self.attachments: dict[str, bytes] = {}
        self._outlines: list[list[object]] = []
        self._encryption: StandardPdfEncryption | None = None

    @property
    def pages(self) -> tuple[PdfPageObject, ...]:
        return tuple(
            PdfPageObject(StructuredState.synthetic(Document(pages=(page,))), page)
            for page in self._pages
        )

    def add_page(self, page: PdfPageObject | Page) -> PdfPageObject:
        value = page._page if isinstance(page, PdfPageObject) else page
        self._pages.append(value)
        return PdfPageObject(StructuredState.synthetic(Document(pages=(value,))), value)

    def add_blank_page(self, width: float = 612.0, height: float = 792.0) -> PdfPageObject:
        page = Page(page_number=len(self._pages) + 1, width=width, height=height)
        self._pages.append(page)
        return PdfPageObject(StructuredState.synthetic(Document(pages=(page,))), page)

    def append_pages_from_reader(self, reader: PdfReader) -> None:
        self._pages.extend(reader_page._page for reader_page in reader.pages)

    def clone_document_from_reader(self, reader: PdfReader) -> None:
        self._pages.clear()
        self.append_pages_from_reader(reader)
        self.metadata.update(reader.metadata)
        self._outlines = [
            [destination.level + 1, destination.title, destination.page + 1]
            for destination in reader.outline
            if destination.page is not None
        ]

    def insert_page(self, page: PdfPageObject | Page, index: int = 0) -> None:
        value = page._page if isinstance(page, PdfPageObject) else page
        self._pages.insert(index, value)

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata.update(metadata)

    def add_outline_item(
        self,
        title: str,
        page_number: int | PdfPageObject = 0,
        parent: Destination | None = None,
        **kwargs: object,
    ) -> Destination:
        del kwargs
        page = (
            page_number._page.page_number - 1
            if isinstance(page_number, PdfPageObject)
            else page_number
        )
        if page < 0 or page >= len(self._pages):
            raise IndexError("outline page is out of range")
        level = parent.level + 2 if parent is not None else 1
        destination = Destination(title, page, level - 1)
        self._outlines.append([level, title, page + 1])
        return destination

    add_bookmark = add_outline_item

    def add_attachment(self, filename: str, data: bytes) -> None:
        self.attachments[filename] = bytes(data)

    def add_uri(
        self,
        page_number: int,
        uri: str,
        rect: tuple[float, float, float, float],
        border: object | None = None,
    ) -> None:
        del border
        page = self._pages[page_number]
        self._pages[page_number] = replace(
            page,
            links=(
                *page.links,
                Link(
                    bbox=coerce_bbox(rect),
                    url=uri,
                    link_type="uri",
                ),
            ),
        )

    add_link = add_uri

    def add_annotation(self, page_number: int, annotation: dict[str, object]) -> None:
        def value(name: str, default: object = None) -> object:
            return annotation.get(name, annotation.get(f"/{name}", default))

        rect = cast(tuple[float, float, float, float], value("Rect", (0, 0, 0, 0)))
        action = value("A")
        if isinstance(action, dict):
            uri = action.get("URI", action.get("/URI"))
            if uri is not None:
                self.add_uri(page_number, str(uri), rect)
                return
        subtype = str(value("Subtype", "Text")).lstrip("/")
        page = self._pages[page_number]
        self._pages[page_number] = replace(
            page,
            annotations=(
                *page.annotations,
                Annotation(subtype=subtype, bbox=rect, contents=str(value("Contents", ""))),
            ),
        )

    def update_page_form_field_values(
        self,
        page: PdfPageObject | Page,
        fields: dict[str, object],
        auto_regenerate: bool = True,
    ) -> None:
        del auto_regenerate
        value = page._page if isinstance(page, PdfPageObject) else page
        updated_fields = tuple(
            replace(field, value_text=str(fields[field.name])) if field.name in fields else field
            for field in value.form_fields
        )
        updated_page = replace(value, form_fields=updated_fields)
        try:
            index = self._pages.index(value)
        except ValueError as exc:
            raise ValueError("page is not owned by this writer") from exc
        self._pages[index] = updated_page

    def encrypt(self, user_password: str, owner_password: str | None = None, **_: object) -> None:
        self._encryption = StandardPdfEncryption(user_password, owner_password)

    def write(self, stream: str | PathLike[str] | BinaryIO) -> bytes:
        data = serialize_document_to_pdf(
            Document(pages=tuple(self._pages), metadata=self.metadata),
            outlines=self._outlines,
            attachments=self.attachments,
            encryption=self._encryption,
        )
        write_bytes(stream, data)
        return data

    def close(self) -> None:
        return None


class PdfMerger:
    def __init__(self) -> None:
        self._documents: list[StructuredState] = []
        self.metadata: dict[str, Any] = {}
        self.attachments: dict[str, bytes] = {}
        self._outlines: list[list[object]] = []

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata.update(metadata)

    def add_attachment(self, filename: str, data: bytes) -> None:
        self.attachments[str(filename)] = bytes(data)

    def _shift_outlines(self, start_page: int, delta: int) -> None:
        for row in self._outlines:
            if len(row) >= 3 and isinstance(row[2], int) and row[2] > start_page:
                row[2] += delta

    def _ingest(self, document: StructuredState, page_offset: int, *, outlines: bool) -> None:
        """Merge one source's metadata, attachments, and optionally outlines."""
        self.metadata.update(document.metadata)
        raw_metadata = document.source_pdf.get_metadata().get("info", {})
        if isinstance(raw_metadata, dict):
            self.metadata.update(raw_metadata)
        for embedded in document.source_pdf.embedded_files():
            self.attachments[embedded.filename] = embedded.data
        if outlines:
            self._outlines.extend(
                [item.level + 1, item.title, item.page_index + page_offset + 1]
                for item in document.source_pdf.iter_outlines()
                if item.page_index is not None
            )

    def append(self, fileobj: PdfInput, pages: Iterable[int] | None = None) -> None:
        document = StructuredState.open(fileobj)
        page_offset = sum(len(item.pages) for item in self._documents)
        self._ingest(document, page_offset, outlines=pages is None)
        if pages is not None:
            selected = tuple(document.pages[index] for index in pages)
            document = StructuredState(
                document.pdf,
                Document(pages=selected, metadata=document.structured.metadata),
            )
        self._documents.append(document)

    def merge(self, page_number: int, fileobj: PdfInput) -> None:
        document = StructuredState.open(fileobj)
        page_offset = sum(len(item.pages) for item in self._documents[:page_number])
        self._shift_outlines(page_offset, len(document.pages))
        self._ingest(document, page_offset, outlines=True)
        self._documents.insert(
            page_number,
            StructuredState(
                document.pdf,
                Document(pages=tuple(document.pages), metadata=document.structured.metadata),
            ),
        )

    def write(self, stream: str | PathLike[str] | BinaryIO) -> bytes:
        return PdfWriterFromDocuments(
            self._documents,
            metadata=self.metadata,
            attachments=self.attachments,
            outlines=self._outlines,
        ).write(stream)

    def close(self) -> None:
        for document in self._documents:
            document.close()


class PdfWriterFromDocuments:
    def __init__(
        self,
        documents: Iterable[StructuredState],
        *,
        metadata: dict[str, Any] | None = None,
        attachments: dict[str, bytes] | None = None,
        outlines: list[list[object]] | None = None,
    ) -> None:
        self.documents = tuple(documents)
        self.metadata = metadata or {}
        self.attachments = attachments or {}
        self.outlines = outlines or []

    def write(self, stream: str | PathLike[str] | BinaryIO) -> bytes:
        pages = tuple(page for document in self.documents for page in document.pages)
        data = serialize_document_to_pdf(
            Document(pages=pages, metadata=self.metadata),
            attachments=self.attachments,
            outlines=self.outlines,
        )
        write_bytes(stream, data)
        return data


__all__ = (
    "Destination",
    "PdfMerger",
    "PdfPageObject",
    "PdfReader",
    "PdfWriter",
    "Rectangle",
)

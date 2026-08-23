"""LlamaIndex-compatible native-text projection over core-pdf engine evidence."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping
from difflib import SequenceMatcher
from typing import Any, cast

from core_pdf import PdfDocument
from core_pdf.impl.engine.spec.s_07_content.operations import iter_content_operations
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_08_graphics.matrix import multiply_affine
from core_pdf.impl.engine.spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.engine.spec.s_09_fonts.glyphs import ensure_glyph_map
from core_pdf.impl.engine.structured import Page
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.primitives import PdfString

GraphicsState = tuple[list[float], str | None, float, float]


def internal_llamaindex_space_width(
    font: Mapping[object, object], resolver: Any, decoder: FontDecoder
) -> float:
    """Return a simple-font space width suitable for engine geometry."""
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


def internal_llamaindex_tj_space_width(
    font: Mapping[object, object], resolver: Any, decoder: FontDecoder
) -> float:
    """Return the target reader's width threshold for explicit TJ adjustments."""
    first_char = lookup_dict_key(font, "FirstChar")
    raw_widths = resolver.resolve(lookup_dict_key(font, "Widths"))
    if not isinstance(first_char, (int, float)) or not isinstance(raw_widths, (list, tuple)):
        return 200.0
    encoding = lookup_dict_key(font, "Encoding")
    difference_space = next(
        (code for code, glyph_name in decoder.differences.items() if glyph_name == "space"),
        None,
    )
    cmap_space = next(
        (
            code[0]
            for code, text in (
                decoder.to_unicode.mappings.items() if decoder.to_unicode is not None else ()
            )
            if len(code) == 1 and text == " "
        ),
        None,
    )
    space_code = (
        difference_space
        if difference_space is not None
        else cmap_space
        if not isinstance(encoding, dict) and cmap_space is not None
        else 32
    )
    space_index = space_code - int(first_char)
    if 0 <= space_index < len(raw_widths):
        return float(raw_widths[space_index])
    return 200.0


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


def internal_llamaindex_rtl_order(text: str) -> str:
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
    return "".join(output)


class NativeTextProjection:
    def __init__(self, pdf: PdfDocument, page_number: int, page: Page) -> None:
        self._pdf = pdf
        self._page_number = page_number
        self._page = page

    def extract_text(self) -> str:  # noqa: C901
        # The target reader only interprets PDF text-showing operators. Core-pdf's
        # structured view may additionally contain OCR, which must stay excluded.
        page = self._pdf.pages[self._page_number - 1]
        if self._page is self._pdf.structured_document.pages[self._page_number - 1]:
            glyphs = page.get_page_program().products.glyphs
            resources = page.resolve_resources()
            raw_fonts = page.document.resolver.resolve(lookup_dict_key(resources, "Font"))
            font_cmaps: dict[str, ToUnicodeCMap] = {}
            resource_cmaps: dict[str, ToUnicodeCMap] = {}
            resource_decoders: dict[str, FontDecoder] = {}
            resource_font_names: dict[str, str] = {}
            resource_space_thresholds: dict[str, float] = {}
            resource_tj_space_thresholds: dict[str, float] = {}
            font_decoders: dict[str, FontDecoder] = {}
            font_space_thresholds: dict[str, float] = {}
            fonts_with_differences: set[str] = set()
            resources_with_differences: set[str] = set()
            uninterpretable_type3_resources: set[str] = set()
            if isinstance(raw_fonts, dict):
                for resource_name, raw_font in raw_fonts.items():
                    font = page.document.resolver.resolve(raw_font)
                    if not isinstance(font, dict):
                        continue
                    base_font = lookup_dict_key(font, "BaseFont")
                    to_unicode = page.document.resolver.resolve(lookup_dict_key(font, "ToUnicode"))
                    char_procs = page.document.resolver.resolve(lookup_dict_key(font, "CharProcs"))
                    if (
                        str(lookup_dict_key(font, "Subtype")) == "Type3"
                        and not isinstance(to_unicode, PdfStream)
                        and isinstance(char_procs, dict)
                        and any(str(name) not in ensure_glyph_map() for name in char_procs)
                    ):
                        uninterpretable_type3_resources.add(str(resource_name))
                    if base_font is None:
                        continue
                    resolved_font = page.document.resolver.resolve_font_dict(font)
                    decoder = FontDecoder(cast(dict[str, object], resolved_font))
                    resource_decoders[str(resource_name)] = decoder
                    resource_font_names[str(resource_name)] = str(base_font)
                    space_width = internal_llamaindex_space_width(
                        font, page.document.resolver, decoder
                    )
                    resource_space_thresholds[str(resource_name)] = space_width * 0.475
                    resource_tj_space_thresholds[str(resource_name)] = (
                        internal_llamaindex_tj_space_width(font, page.document.resolver, decoder)
                        * 0.475
                    )
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
                    resource_tj_space_thresholds[resource_name or ""] = cid_space_width * 0.475
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
                glyph_font_name = glyph.font_name or ""
                show_index = sequence_show_indices.get(glyph.seqno)
                resource_name = show_resources[show_index] if show_index is not None else None
                glyph_cmap = resource_cmaps.get(resource_name or "") or font_cmaps.get(
                    glyph_font_name
                )
                glyph_text = (
                    glyph_cmap.decode(glyph.code_bytes) if glyph_cmap is not None else glyph.text
                )
                if resource_name in uninterpretable_type3_resources:
                    glyph_text = glyph.code_bytes.decode("latin-1")
                glyph_decoder = font_decoders.get(glyph_font_name)
                cid_width_groups.update(
                    (glyph.seqno,)
                    if glyph_decoder is not None
                    and glyph_decoder.is_cid_font
                    and glyph_font_name.endswith("+")
                    else ()
                )
                group_text_widths[glyph.seqno] = group_text_widths.get(
                    glyph.seqno, 0.0
                ) + internal_decoded_width(glyph_decoder, glyph.code_bytes)
                group_space_widths[glyph.seqno] = font_space_thresholds.get(glyph_font_name, 0.0)
                has_differences = (
                    resource_name in resources_with_differences
                    if direct_sequence_fonts
                    else glyph_font_name in fonts_with_differences
                )
                if has_differences and glyph.code_bytes in {b"'", b"\x15"}:
                    glyph_text = glyph.text
                core_ligature_pair = (
                    previous_glyph is not None
                    and previous_glyph.seqno == glyph.seqno
                    and previous_glyph.code_bytes == glyph.code_bytes
                    and glyph.code_bytes not in {b"f", b"\x00f", b"\x00I"}
                    and cast(Any, previous_glyph).text + glyph.text in {"ff", "fi", "fl"}
                    and glyph_text not in {"ff", "fi", "fl", "ffi", "ffl", "ﬀ", "ﬁ", "ﬂ", "ﬃ", "ﬄ"}
                )
                if core_ligature_pair:
                    pair_text = cast(Any, previous_glyph).text + glyph.text
                    glyph_text = {"ff": "ﬀ", "fi": "ﬁ", "fl": "ﬂ"}[pair_text]
                duplicate_ligature_part = (
                    previous_glyph is not None
                    and previous_glyph.seqno == glyph.seqno
                    and previous_glyph.code_bytes == glyph.code_bytes
                    and glyph_text == previous_decoded
                    and glyph_text
                    in {
                        "ff",
                        "fi",
                        "fl",
                        "ffi",
                        "ffl",
                        "ﬀ",
                        "ﬁ",
                        "ﬂ",
                        "ﬃ",
                        "ﬄ",
                    }
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
            aligned_engine_sequences = set(aligned_engine_seqnos)
            unmatched_text_groups = [
                group
                for group in groups
                if group[0] not in aligned_engine_sequences and group[6].strip()
            ]
            byte_aligned_groups = False
            if (
                lost_show_boundaries
                and not unmatched_text_groups
                or (
                    not direct_sequence_fonts
                    and not compact_sequence_fonts
                    and abs(len(flat_glyph_bytes) - len(flat_show_bytes)) <= 2
                    and not unmatched_text_groups
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

            def moved_far_enough(width: float) -> bool:
                previous = multiply_affine(previous_tm, previous_cm)
                current = multiply_affine(tm_matrix, cm_matrix)
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
                previous = multiply_affine(previous_tm, previous_cm)
                current = multiply_affine(tm_matrix, cm_matrix)
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
                        cm_matrix = multiply_affine(
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
                        shown_position = multiply_affine(tm_matrix, cm_matrix)
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
                        space_threshold = resource_tj_space_thresholds.get(
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
                                shown_position = multiply_affine(tm_matrix, cm_matrix)
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
                operand_output_override = internal_llamaindex_rtl_order(operand_output)
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
                    or seqno in sequence_show_indices
                    or output.endswith(".")
                    and text[:1].isupper()
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
                heading_number_space = (
                    previous_x1 is not None
                    and output.rsplit("\n", 1)[-1].isalpha()
                    and output.rsplit("\n", 1)[-1].isupper()
                    and text[:1].isdigit()
                    and y1 - y0 > 15
                    and x0 - previous_x1 > (y1 - y0) * 0.15
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
                        or (
                            malformed_text_state
                            or not direct_sequence_fonts
                            and not compact_sequence_fonts
                        )
                        and previous_x1 is not None
                        and abs(x0 - previous_x1) < (y1 - y0)
                        or previous_x1 is not None
                        and abs(x0 - previous_x1) < (y1 - y0)
                        and (
                            output.endswith(".")
                            and text[:1].isupper()
                            or output.endswith("’")
                            and text[:1].islower()
                        )
                    )
                    or (
                        group_resource is None
                        and previous_x1 is not None
                        and x0 < previous_x1 - (y1 - y0)
                        and text.isupper()
                    )
                )
                unmapped_baseline_newline = (
                    not direct_sequence_fonts
                    and not compact_sequence_fonts
                    and previous_y0 is not None
                    and previous_x1 is not None
                    and x0 < previous_x1
                    and abs(y0 - previous_y0) > (y1 - y0) * 0.8
                )
                current_line = output.rsplit("\n", 1)[-1]
                wide_column_newline = (
                    force_newline_before
                    and previous_y0 is not None
                    and previous_x1 is not None
                    and len(current_line.split()) == 1
                    and x0 - previous_x1 > (y1 - y0) * 5
                    and abs(y0 - previous_y0) > 0.5
                    and abs(y0 - previous_y0) < (y1 - y0) * 0.2
                )
                if baseline_newline and not vertically_separate and output.endswith("\n"):
                    pass
                elif (
                    vertically_separate
                    or baseline_newline
                    or unmapped_baseline_newline
                    or wide_column_newline
                ) and not degenerate:
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
                        or literal_gap_space
                        or missing_resource_space
                        or operand_position_space
                        or compact_font_boundary_space
                        or heading_number_space
                    )
                    and (
                        not text.startswith(" ")
                        or effective_force_space
                        and not text.startswith("  ")
                        or literal_gap_space
                        or text.isspace()
                        and len(text) > 10
                        and operator == "Tj"
                        and not moved_before_show
                    )
                    and (
                        effective_force_space
                        or literal_gap_space
                        or missing_resource_space
                        or operand_position_space
                        or compact_font_boundary_space
                        or heading_number_space
                        or (
                            operator == "Tj"
                            and not moved_before_show
                            and output[-1:] not in {":", "/"}
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
                if text.isspace() and len(text) > 10 and operator == "Tj":
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
            glyph_output = internal_llamaindex_rtl_order(output)
            if operand_output_override is not None:
                return operand_output_override
            return glyph_output
        return "\n".join(
            line.text for block in self._page.blocks for line in block.lines if line.source != "ocr"
        )

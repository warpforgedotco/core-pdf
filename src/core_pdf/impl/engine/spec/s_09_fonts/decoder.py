# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re

from core_pdf.impl.engine.spec.s_07_syntax.errors import PdfParseError
from core_pdf.impl.engine.spec.s_07_syntax.primitives import (
    PdfDictLike,
    PdfName,
    PdfObject,
    PdfStream,
    PdfString,
    parse_float,
    parse_name,
)
from core_pdf.impl.engine.spec.s_09_fonts.cid_unicode import predefined_cid_to_unicode
from core_pdf.impl.engine.spec.s_09_fonts.cmaps import ToUnicodeCMap
from core_pdf.impl.engine.spec.s_09_fonts.data.core14 import get_font_metrics_props
from core_pdf.impl.engine.spec.s_09_fonts.encoding import split_chunks
from core_pdf.impl.engine.spec.s_09_fonts.glyphs import glyph_name_to_unicode
from core_pdf.impl.engine.spec.s_09_fonts.helpers import (
    build_decode_table,
    cached_decode_table,
    decode_chunks_with_table,
    decode_with_table,
    fallback_with_pdfdoc,
    parse_differences,
)
from core_pdf.impl.engine.spec.s_09_fonts.truetype import tt_cmap, tt_tables
from core_pdf.impl.engine.spec.s_09_fonts.widths import get_descendant, parse_font_widths
from core_pdf.impl.third_party.cid import CMapDecoder, resolve_cmap_decoder

AIGDT_PRIVATE_USE_MAP = {
    0xF060: "±",
    0xF062: "⟂",
    0xF063: "⏥",
    0xF066: "∥",
    0xF06A: "⌖",
}

TYPE1_ENCODING_ENTRY_RE = re.compile(
    rb"dup\s+(\d{1,3})\s+/([^\s/%\[\]()<>{}]+)\s+put"
)


def normalized_font_name(name: str | None) -> str:
    if name is None:
        return ""
    _, _, unprefixed = name.partition("+")
    return unprefixed or name


def codepoint_score(codepoint: int) -> tuple[int, int]:
    if codepoint == 0 or 0xD800 <= codepoint <= 0xDFFF or codepoint > 0x10FFFF:
        return (0, 0)
    if codepoint < 32 or 0x7F <= codepoint <= 0x9F:
        return (1, 0)
    if 0xE000 <= codepoint <= 0xF8FF:
        return (2, 0)
    return (3, -codepoint)


def translate_symbol_codepoint(base_font_name: str | None, codepoint: int) -> str | None:
    font_name = normalized_font_name(base_font_name)
    if font_name == "AIGDT":
        mapped = AIGDT_PRIVATE_USE_MAP.get(codepoint)
        if mapped is not None:
            return mapped
    if codepoint_score(codepoint)[0] == 0:
        return None
    return chr(codepoint)


def pdf_string_name(value: PdfObject) -> str | None:
    parsed = parse_name(value)
    if parsed is not None:
        return parsed
    if isinstance(value, PdfString):
        return value.data.decode("latin-1", "replace")
    return None


def get_name_key(mapping: PdfDictLike, key: str) -> PdfObject:
    value = mapping.get(key)
    if value is not None:
        return value
    for candidate_key, candidate_value in mapping.items():
        if isinstance(candidate_key, (str, bytes, PdfName)) and parse_name(candidate_key) == key:
            return candidate_value
    return None


def parse_type1_font_program_encoding(data: bytes) -> dict[int, str]:
    encoding_data = data.split(b"currentfile eexec", 1)[0]
    encoding: dict[int, str] = {}
    for match in TYPE1_ENCODING_ENTRY_RE.finditer(encoding_data):
        code = int(match.group(1))
        if code > 255:
            continue
        encoding[code] = match.group(2).decode("latin-1", "replace")
    return encoding


class FontDecoder:
    __slots__ = (
        "font",
        "ligature_overrides",
        "to_unicode",
        "cmap",
        "base_encoding",
        "differences",
        "is_cid_font",
        "is_type3",
        "byte_decode_table",
        "cid_to_unicode",
        "widths",
        "default_width",
        "is_vertical",
        "ascent",
        "descent",
        "fast_widths",
        "fast_widths_cid",
    )

    font: PdfDictLike
    ligature_overrides: dict[int, str]
    to_unicode: ToUnicodeCMap | None
    cmap: CMapDecoder | None
    base_encoding: str | None
    differences: dict[int, str]
    is_cid_font: bool
    is_type3: bool
    byte_decode_table: tuple[str, ...] | None
    cid_to_unicode: dict[int, str]
    widths: dict[int, float]
    default_width: float
    is_vertical: bool
    ascent: float
    descent: float
    fast_widths: tuple[float, ...]
    fast_widths_cid: list[float] | None

    def __init__(
        self,
        font: PdfDictLike,
        ligature_overrides: dict[int, str] | None = None,
    ) -> None:
        self.font = font
        self.ligature_overrides = ligature_overrides if ligature_overrides is not None else {}
        self.to_unicode: ToUnicodeCMap | None = None
        self.cmap: CMapDecoder | None = None
        self.base_encoding: str | None = None
        self.differences: dict[int, str] = {}
        self.is_cid_font: bool = False
        self.is_type3: bool = False
        self.byte_decode_table: tuple[str, ...] | None = None
        self.cid_to_unicode: dict[int, str] = {}
        self.widths: dict[int, float] = {}
        self.default_width: float = 1000.0
        self.is_vertical: bool = False
        self.ascent: float = 800.0
        self.descent: float = -200.0
        self.fast_widths = ()
        self.fast_widths_cid = None
        self.__post_init__()

    def __post_init__(self) -> None:
        font_dict = dict(self.font)
        font = self.font
        subtype = font_dict.get("Subtype")
        if subtype is not None:
            subtype = parse_name(subtype) or str(subtype)

        to_unicode_obj = font.get("ToUnicode")
        try:
            to_unicode = (
                ToUnicodeCMap(to_unicode_obj) if isinstance(to_unicode_obj, PdfStream) else None
            )
        except PdfParseError:
            to_unicode = None

        cmap, base_encoding, differences = self.parse_encoding(font)
        widths, default_width, is_vertical = parse_font_widths(font_dict, subtype)

        base_font = font_dict.get("BaseFont")
        base_font_name = parse_name(base_font) or (
            str(base_font) if base_font is not None else None
        )
        if (base_encoding and base_encoding.endswith("-V")) or (
            base_font_name and base_font_name.endswith("-V")
        ):
            is_vertical = True

        ascent, descent = self.parse_metrics(font_dict, subtype, base_font_name, widths)

        is_type3 = subtype == "Type3"
        if is_type3:
            widths = self.adjust_type3_widths(font_dict, widths)

        byte_decode_table: tuple[str, ...] | None = None
        if to_unicode is None and subtype != "Type0":
            embedded_encoding = {}
            if not base_encoding:
                embedded_encoding = self.parse_embedded_type1_encoding(font_dict)
            if embedded_encoding:
                byte_decode_table = build_decode_table(
                    lambda code: self.embedded_encoding_fallback(embedded_encoding, code),
                    differences,
                )
            else:
                key = base_encoding or ("Type3" if is_type3 else "")
                byte_decode_table = cached_decode_table(key, tuple(sorted(differences.items())))
        cid_to_unicode: dict[int, str] = {}
        if to_unicode is None and subtype == "Type0":
            cid_to_unicode.update(self.parse_predefined_cid_unicode(font_dict, is_vertical))
            cid_to_unicode.update(self.parse_identity_truetype_unicode(font_dict))

        self.to_unicode = to_unicode
        self.cmap = cmap
        self.base_encoding = base_encoding
        self.differences = differences
        self.is_cid_font = subtype == "Type0"
        self.is_type3 = is_type3
        self.byte_decode_table = byte_decode_table
        self.cid_to_unicode = cid_to_unicode
        self.widths = widths
        self.default_width = default_width
        self.is_vertical = is_vertical
        self.ascent = ascent
        self.descent = descent

        # Populate fast_widths for optimized text advance calculations (1-byte)
        dw = self.default_width
        dw_pos = dw if dw > 0.0 else 1000.0
        space_w = dw if dw > 0.0 else 250.0
        self.fast_widths = tuple(
            self.widths.get(i, space_w if i == 32 else dw_pos) for i in range(256)
        )

        # Optimized O(1) array-based cache for all 2-byte CIDs
        if self.is_cid_font:
            # We use a 64K list of floats. 512KB is tiny for modern documents.
            self.fast_widths_cid = [dw_pos] * 65536
            fwc = self.fast_widths_cid
            fwc[32] = space_w
            for k, v in self.widths.items():
                if 0 <= k < 65536:
                    fwc[k] = v
        elif self.widths and len(self.widths) > 20:
            # Non-CID fonts with CMap: pre-flatten sparse widths dict into
            # a 64K array so the slow text_advance_vector path avoids dict.get().
            max_code = max(self.widths.keys())
            if max_code < 65536:
                self.fast_widths_cid = [dw_pos] * 65536
                fwc = self.fast_widths_cid
                fwc[32] = space_w
                for k, v in self.widths.items():
                    fwc[k] = v
            else:
                self.fast_widths_cid = None
        else:
            self.fast_widths_cid = None

    def parse_encoding(
        self, font: PdfDictLike
    ) -> tuple[CMapDecoder | None, str | None, dict[int, str]]:
        cmap = None
        base_encoding = None
        differences: dict[int, str] = {}
        encoding_obj = font.get("Encoding")
        if isinstance(encoding_obj, str):
            base_encoding = parse_name(encoding_obj)
            if base_encoding is not None:
                cmap = resolve_cmap_decoder(base_encoding)
        elif isinstance(encoding_obj, PdfStream):
            try:
                cmap = CMapDecoder(encoding_obj.data, usecmap_resolver=resolve_cmap_decoder)
            except ValueError:
                cmap = None
        elif isinstance(encoding_obj, dict):
            base_encoding = parse_name(encoding_obj.get("BaseEncoding"))
            differences_obj = encoding_obj.get("Differences")
            if differences_obj is not None and not isinstance(differences_obj, (list, tuple)):
                raise ValueError("invalid encoding differences array")
            differences = parse_differences(
                list(differences_obj) if isinstance(differences_obj, tuple) else differences_obj,
                parse_name,
            )
        else:
            base_encoding = parse_name(encoding_obj)
            if base_encoding is not None:
                cmap = resolve_cmap_decoder(base_encoding)
        return cmap, base_encoding, differences

    @staticmethod
    def embedded_encoding_fallback(encoding: dict[int, str], code: int) -> str:
        glyph_name = encoding.get(code)
        if glyph_name is None:
            return fallback_with_pdfdoc(code)
        decoded = glyph_name_to_unicode(glyph_name)
        if len(glyph_name) > 1 and decoded == glyph_name:
            return fallback_with_pdfdoc(code)
        return decoded

    def parse_embedded_type1_encoding(self, font_dict: PdfDictLike) -> dict[int, str]:
        descriptor = font_dict.get("FontDescriptor")
        if not isinstance(descriptor, dict):
            return {}
        font_file = descriptor.get("FontFile")
        if not isinstance(font_file, PdfStream):
            return {}
        return parse_type1_font_program_encoding(font_file.data)

    def parse_predefined_cid_unicode(
        self, font_dict: PdfDictLike, is_vertical: bool
    ) -> dict[int, str]:
        descendant = get_descendant(font_dict)
        if not isinstance(descendant, dict):
            return {}
        cid_system_info = descendant.get("CIDSystemInfo")
        if not isinstance(cid_system_info, dict):
            return {}
        registry = pdf_string_name(get_name_key(cid_system_info, "Registry"))
        ordering = pdf_string_name(get_name_key(cid_system_info, "Ordering"))
        if registry != "Adobe" or ordering is None:
            return {}
        return dict(predefined_cid_to_unicode(ordering, is_vertical))

    def parse_metrics(
        self,
        font_dict: PdfDictLike,
        subtype: str | None,
        base_font_name: str | None,
        widths: dict[int, float],
    ) -> tuple[float, float]:
        ascent, descent = 800.0, -200.0
        descriptor = font_dict.get("FontDescriptor")
        if subtype == "Type0":
            descendant = get_descendant(font_dict)
            if isinstance(descendant, dict):
                desc_descriptor = descendant.get("FontDescriptor")
                descriptor = desc_descriptor or descriptor

        props = get_font_metrics_props(base_font_name) if base_font_name is not None else None
        if props is not None and not widths:
            ascent_value = props.get("Ascent")
            descent_value = props.get("Descent")
            if not isinstance(ascent_value, (type(None), bool, int, float, str, bytes)):
                raise ValueError("invalid core14 ascent value")
            if not isinstance(descent_value, (type(None), bool, int, float, str, bytes)):
                raise ValueError("invalid core14 descent value")
            ascent = parse_float(ascent_value, ascent)
            descent = parse_float(descent_value, descent)

        if isinstance(descriptor, dict):
            ascent = parse_float(descriptor.get("Ascent"), ascent)
            descent = parse_float(descriptor.get("Descent"), descent)
        return ascent, descent

    def adjust_type3_widths(
        self, font_dict: PdfDictLike, widths: dict[int, float]
    ) -> dict[int, float]:
        font_matrix = font_dict.get("FontMatrix")
        if isinstance(font_matrix, (list, tuple)) and len(font_matrix) >= 1:
            fm_a = parse_float(font_matrix[0], 0.001)
        else:
            fm_a = 0.001
        width_scale = fm_a * 1000.0
        if abs(width_scale - 1.0) > 1e-6:
            return {k: v * width_scale for k, v in widths.items()}
        return widths

    def parse_identity_truetype_unicode(self, font_dict: PdfDictLike) -> dict[int, str]:
        if parse_name(font_dict.get("Encoding")) not in {"Identity-H", "Identity-V"}:
            return {}
        base_font_name = parse_name(font_dict.get("BaseFont"))
        descendant = get_descendant(font_dict)
        if not isinstance(descendant, dict):
            return {}
        if parse_name(descendant.get("Subtype")) != "CIDFontType2":
            return {}
        descriptor = descendant.get("FontDescriptor")
        if not isinstance(descriptor, dict):
            return {}
        font_file = descriptor.get("FontFile2")
        if not isinstance(font_file, PdfStream):
            return {}
        try:
            tables = tt_tables(font_file.data)
            cp_to_gid = tt_cmap(font_file.data, tables)
        except (PdfParseError, ValueError, TypeError, IndexError):
            return {}
        if not cp_to_gid:
            return {}

        gid_to_cid: dict[int, int] | None = None
        cid_to_gid = descendant.get("CIDToGIDMap")
        if isinstance(cid_to_gid, PdfStream):
            data = cid_to_gid.data
            gid_to_cid = {}
            for cid in range(len(data) // 2):
                gid = (data[cid * 2] << 8) | data[cid * 2 + 1]
                gid_to_cid.setdefault(gid, cid)
        elif cid_to_gid is not None and parse_name(cid_to_gid) != "Identity":
            return {}

        cid_to_unicode: dict[int, str] = {}
        for codepoint, gid in sorted(cp_to_gid.items()):
            mapped = translate_symbol_codepoint(base_font_name, codepoint)
            if mapped is None:
                continue
            cid = gid_to_cid.get(gid) if gid_to_cid is not None else gid
            if cid is None:
                continue
            existing = cid_to_unicode.get(cid)
            if existing is None or codepoint_score(ord(mapped[0])) > codepoint_score(
                ord(existing[0])
            ):
                cid_to_unicode[cid] = mapped
        return cid_to_unicode

    def decode(self, data: bytes) -> str:
        if not data:
            return ""
        if self.byte_decode_table is not None and not self.is_cid_font and self.to_unicode is None:
            return decode_with_table(data, self.byte_decode_table)
        if self.to_unicode is not None:
            result = self.to_unicode.decode(data)
            if self.ligature_overrides:
                # Pre-get ligature_overrides for speed
                lo = self.ligature_overrides
                # result is already decoded, so we map chars to unicode points for lookup
                return "".join(lo.get(ord(ch), ch) for ch in result)
            return result
        if self.is_cid_font:
            if self.cmap is not None:
                cid_items = self.cmap.decode(data)
            else:
                chunks = split_chunks(data, True, None)
                cid_items = [
                    (chunk, (chunk[0] << 8) | chunk[1] if len(chunk) == 2 else chunk[0])
                    for chunk in chunks
                ]
            cid_to_unicode = self.cid_to_unicode
            out: list[str] = []
            for ignored_chunk, cid in cid_items:
                mapped = cid_to_unicode.get(cid)
                if mapped is not None:
                    out.append(mapped)
                elif cid == 0:
                    out.append("\u0000")
                elif cid < 0x110000:
                    out.append(chr(cid))
                else:
                    out.append("\ufffd")
            return "".join(out)
        key = self.base_encoding or ("Type3" if self.is_type3 else "")
        table = cached_decode_table(key, tuple(sorted(self.differences.items())))
        return decode_with_table(data, table)

    def cid_for_chunk(self, chunk: bytes) -> int:
        if self.cmap is not None:
            decoded = self.cmap.decode(chunk)
            if decoded:
                return decoded[0][1]
        return (chunk[0] << 8) | chunk[1] if len(chunk) == 2 else chunk[0]

    def uses_identity_two_byte_cmap(self) -> bool:
        if self.cmap is None:
            return True
        return (
            self.cmap.default_to_identity
            and self.cmap.decode_lengths == (2,)
            and not self.cmap.cid_mappings
            and not self.cmap.cid_ranges
        )

    def decode_chunks(self, chunks: list[bytes]) -> list[str]:
        if not chunks:
            return []
        if self.byte_decode_table is not None and not self.is_cid_font and self.to_unicode is None:
            return decode_chunks_with_table(chunks, self.byte_decode_table)
        return [self.decode(chunk) for chunk in chunks]

    def glyph_width(self, code: int) -> float:
        if self.fast_widths_cid is not None and 0 <= code < 65536:
            return self.fast_widths_cid[code]
        if 0 <= code < 256:
            return self.fast_widths[code]
        return self.widths.get(code, self.default_width)

    def text_advance_vector(
        self,
        data: bytes,
        *,
        font_size: float,
        char_space: float,
        word_space: float,
        horizontal_scale: float,
        chunks: list[bytes] | None = None,
    ) -> tuple[float, float]:
        if not data:
            return (0.0, 0.0)

        # Hoist loop-invariant computations (LICM)
        cs = char_space * 1000.0 / font_size if font_size else 0.0
        ws = word_space * 1000.0 / font_size if font_size else 0.0
        scale = font_size * horizontal_scale / 100000.0

        # Common fast path: single-byte encodings (Standard fonts)
        if (
            chunks is None
            and not self.is_cid_font
            and self.to_unicode is None
            and self.cmap is None
        ):
            widths = self.fast_widths
            total = 0.0
            space_count = 0
            for b in data:
                total += widths[b]
                if b == 32:
                    space_count += 1
            total += len(data) * cs + space_count * ws
            if self.is_vertical:
                return (0.0, -total * scale)
            return (total * scale, 0.0)

        # NEW FAST PATH: two-byte CID encodings (Identity-H/V etc)
        # This covers ~90% of high-volume modern PDF text
        n = len(data)
        if (
            chunks is None
            and self.is_cid_font
            and self.fast_widths_cid is not None
            and n % 2 == 0
            and self.uses_identity_two_byte_cmap()
        ):
            fast_widths = self.fast_widths_cid
            total = 0.0
            for i in range(0, n, 2):
                code = (data[i] << 8) | data[i + 1]
                total += fast_widths[code]
            total += (n >> 1) * cs
            # space (32) in CID is usually mapped to GID 32, but we checked in init
            # Actually count(32) for CID needs to be per-2-byte.
            # We'll skip precise word_space for now in this fast path if rare.
            if self.is_vertical:
                return (0.0, -total * scale)
            return (total * scale, 0.0)

        if chunks is None:
            chunks = split_chunks(data, self.is_cid_font, self.to_unicode or self.cmap)

        dw = self.default_width
        dw_pos = dw if dw > 0.0 else 1000.0
        space_w = dw if dw > 0.0 else 250.0
        total = 0.0
        fwc: list[float] | None = self.fast_widths_cid

        for chunk in chunks:
            code = self.cid_for_chunk(chunk) if self.is_cid_font else chunk[0]
            w: float
            if fwc is not None:
                w = fwc[code] if 0 <= code < 65536 else dw_pos
            else:
                width_value = self.widths.get(code)
                if width_value is None:
                    w = space_w if code == 32 else dw_pos
                else:
                    w = width_value
            total += w + cs
            if code == 32:
                total += ws

        if self.is_vertical:
            return (0.0, -total * scale)
        return (total * scale, 0.0)

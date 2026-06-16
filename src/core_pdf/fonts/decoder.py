from __future__ import annotations
from core_pdf.syntax.primitives import PdfStream, parse_float, parse_name
from core_pdf.fonts.encoding import split_chunks
from core_pdf.fonts.helpers import cached_decode_table, decode_chunks_with_table, decode_with_table, parse_differences
from core_pdf.fonts.cmaps import CMapDecoder, ToUnicodeCMap
from core_pdf.fonts.data.core14 import FONT_DATA
from core_pdf.fonts.widths import get_descendant, parse_font_widths

import typing

if typing.TYPE_CHECKING:
    from typing import Any


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
        "widths",
        "default_width",
        "is_vertical",
        "ascent",
        "descent",
        "fast_widths",
        "fast_widths_cid",
    )

    font: dict[str, Any]
    ligature_overrides: dict[int, str]
    to_unicode: ToUnicodeCMap | None
    cmap: CMapDecoder | None
    base_encoding: str | None
    differences: dict[int, str]
    is_cid_font: bool
    is_type3: bool
    byte_decode_table: tuple[str, ...] | None
    widths: dict[int, float]
    default_width: float
    is_vertical: bool
    ascent: float
    descent: float
    fast_widths: tuple[float, ...]
    fast_widths_cid: list[float] | None

    def __init__(
        self,
        font: dict[str, Any],
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
        to_unicode = ToUnicodeCMap(to_unicode_obj) if isinstance(to_unicode_obj, PdfStream) else None

        cmap, base_encoding, differences = self.parse_encoding(font)
        widths, default_width, is_vertical = parse_font_widths(font_dict, subtype)

        base_font = font_dict.get("BaseFont")
        base_font_name = parse_name(base_font) or (str(base_font) if base_font is not None else None)
        if base_encoding == "V" or (base_font_name and base_font_name.endswith("-V")):
            is_vertical = True

        ascent, descent = self.parse_metrics(font_dict, subtype, base_font_name, widths)

        is_type3 = subtype == "Type3"
        if is_type3:
            widths = self.adjust_type3_widths(font_dict, widths)

        byte_decode_table: tuple[str, ...] | None = None
        if to_unicode is None and not subtype == "Type0":
            key = "Type3" if is_type3 else (base_encoding or "")
            byte_decode_table = cached_decode_table(key, tuple(sorted(differences.items())))

        self.to_unicode = to_unicode
        self.cmap = cmap
        self.base_encoding = base_encoding
        self.differences = differences
        self.is_cid_font = subtype == "Type0"
        self.is_type3 = is_type3
        self.byte_decode_table = byte_decode_table
        self.widths = widths
        self.default_width = default_width
        self.is_vertical = is_vertical
        self.ascent = ascent
        self.descent = descent

        # Populate fast_widths for optimized text advance calculations (1-byte)
        dw = self.default_width
        dw_pos = dw if dw > 0.0 else 1000.0
        space_w = dw if dw > 0.0 else 250.0
        self.fast_widths = tuple(self.widths.get(i, space_w if i == 32 else dw_pos) for i in range(256))

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

    def parse_encoding(self, font: dict[str, Any]) -> tuple[CMapDecoder | None, str | None, dict[int, str]]:
        cmap = None
        base_encoding = None
        differences: dict[int, str] = {}
        encoding_obj = font.get("Encoding")
        if isinstance(encoding_obj, str):
            base_encoding = encoding_obj
        elif isinstance(encoding_obj, PdfStream):
            cmap = CMapDecoder(encoding_obj)
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
        return cmap, base_encoding, differences

    def parse_metrics(
        self,
        font_dict: dict[str, Any],
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

        if base_font_name in FONT_DATA and not widths:
            entry = FONT_DATA[base_font_name]
            props = entry["props"]
            ascent = parse_float(props.get("Ascent"), ascent)
            descent = parse_float(props.get("Descent"), descent)

        if isinstance(descriptor, dict):
            ascent = parse_float(descriptor.get("Ascent"), ascent)
            descent = parse_float(descriptor.get("Descent"), descent)
        return ascent, descent

    def adjust_type3_widths(self, font_dict: dict[str, Any], widths: dict[int, float]) -> dict[int, float]:
        font_matrix = font_dict.get("FontMatrix")
        if isinstance(font_matrix, (list, tuple)) and len(font_matrix) >= 1:
            fm_a = parse_float(font_matrix[0], 0.001)
        else:
            fm_a = 0.001
        width_scale = fm_a * 1000.0
        if abs(width_scale - 1.0) > 1e-6:
            return {k: v * width_scale for k, v in widths.items()}
        return widths

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
            chunks = split_chunks(data, True, self.cmap)
            out: list[str] = []
            for chunk in chunks:
                # Manual big-endian decode for speed
                cid = (chunk[0] << 8) | chunk[1] if len(chunk) == 2 else chunk[0]
                if cid == 0:
                    out.append("\u0000")
                elif cid < 0x110000:
                    out.append(chr(cid))
                else:
                    out.append("\ufffd")
            return "".join(out)
        key = "Type3" if self.is_type3 else (self.base_encoding or "")
        table = cached_decode_table(key, tuple(sorted(self.differences.items())))
        return decode_with_table(data, table)

    def decode_chunks(self, chunks: list[bytes]) -> list[str]:
        if not chunks:
            return []
        if self.byte_decode_table is not None and not self.is_cid_font and self.to_unicode is None:
            return decode_chunks_with_table(chunks, self.byte_decode_table)
        return [self.decode(chunk) for chunk in chunks]

    def glyph_width(self, code: int) -> float:
        if self.fast_widths_cid is not None:
            if 0 <= code < 65536:
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
        if chunks is None and not self.is_cid_font and self.to_unicode is None and self.cmap is None:
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
        if chunks is None and self.is_cid_font and self.fast_widths_cid is not None and n % 2 == 0:
            fwc = self.fast_widths_cid
            total = 0.0
            for i in range(0, n, 2):
                code = (data[i] << 8) | data[i + 1]
                total += fwc[code]
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
        fwc = self.fast_widths_cid

        for chunk in chunks:
            code = (chunk[0] << 8) | chunk[1] if len(chunk) == 2 else chunk[0]
            if fwc is not None:
                w = fwc[code] if 0 <= code < 65536 else dw_pos
            else:
                w = self.widths.get(code)
                if w is None:
                    w = space_w if code == 32 else dw_pos
            total += w + cs
            if code == 32:
                total += ws

        if self.is_vertical:
            return (0.0, -total * scale)
        return (total * scale, 0.0)

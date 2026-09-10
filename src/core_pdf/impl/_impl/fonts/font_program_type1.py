"""Application outline and damaged-program adapter for Type 1 fonts."""

from __future__ import annotations

import re
from collections.abc import Iterator

from core_pdf._vendor.fontTools.misc.psCharStrings import T1CharString
from core_pdf._vendor.fontTools.pens.boundsPen import BoundsPen
from core_pdf._vendor.fontTools.pens.recordingPen import RecordingPen
from core_pdf._vendor.fontTools.pens.transformPen import TransformPen
from core_pdf.impl._impl.fonts.font_program_truetype import internal_recording_to_contours
from core_pdf.impl._impl.fonts.raster_kernel import Point, rasterize_contours, transform_contours
from core_pdf_spec.s_09_fonts.font_program_type1 import (
    binary_entries,
    decode_charstring,
    decode_eexec_payload,
    decrypt_type1,
)

internal_LEN_IV_RE = re.compile(rb"/lenIV\s+(-?\d+)\s+def\b")


internal_FONT_MATRIX_RE = re.compile(
    rb"/FontMatrix\s*\[\s*([-+.\dEe]+)\s+([-+.\dEe]+)\s+"
    rb"([-+.\dEe]+)\s+([-+.\dEe]+)\s+([-+.\dEe]+)\s+([-+.\dEe]+)\s*\]"
)


internal_SUBR_RE = re.compile(rb"\bdup\s+(\d+)\s+(\d+)\s+(?:RD|-\|)[ \t\r\n]")


internal_CHARSTRING_RE = re.compile(rb"/([^\s/]+)\s+(\d+)\s+(?:RD|-\|)[ \t\r\n]")


internal_HEX_BYTES = frozenset(b"0123456789abcdefABCDEF \t\r\n")


internal_MAX_SUBROUTINES = 4096


class internal_Type1FontProgram:
    """A bounded decoder for the outlines in one embedded Type 1 program."""

    __slots__ = (
        "charstrings",
        "font_matrix",
        "glyph_names",
        "glyph_name_to_id",
        "subrs",
    )

    def __init__(self, data: bytes, *, length1: int | None = None) -> None:
        private = self.decode_private(data, length1)
        len_iv_match = internal_LEN_IV_RE.search(private)
        len_iv = int(len_iv_match.group(1)) if len_iv_match is not None else 4
        if len_iv < -1 or len_iv > 32:
            raise ValueError("invalid Type 1 lenIV")

        subr_data = {
            int(index): payload for index, payload in self.binary_entries(private, internal_SUBR_RE)
        }
        subr_count = max(subr_data, default=-1) + 1
        if subr_count > internal_MAX_SUBROUTINES:
            raise ValueError("Type 1 subroutine index exceeds decoder limit")
        empty = T1CharString(b"\x0b", subrs=[])
        subrs = [empty for _ in range(subr_count)]
        for index, encrypted in subr_data.items():
            subrs[index] = self.prepare_charstring(encrypted, len_iv, subrs)
        for subr in subrs:
            subr.subrs = subrs
        self.subrs = subrs

        charstrings = {
            name.decode("latin-1"): payload
            for name, payload in self.binary_entries(private, internal_CHARSTRING_RE)
        }
        self.charstrings = {
            name: self.prepare_charstring(encrypted, len_iv, subrs)
            for name, encrypted in charstrings.items()
        }
        if not self.charstrings:
            raise ValueError("Type 1 CharStrings are missing")
        self.glyph_names = tuple(self.charstrings)
        self.glyph_name_to_id = {name: gid for gid, name in enumerate(self.glyph_names)}

        matrix_match = internal_FONT_MATRIX_RE.search(data)
        self.font_matrix = (
            tuple(float(value) for value in matrix_match.groups())
            if matrix_match is not None
            else (0.001, 0.0, 0.0, 0.001, 0.0, 0.0)
        )

    def glyph_id_for_name(self, glyph_name: str) -> int | None:
        glyph_id = self.glyph_name_to_id.get(glyph_name)
        if glyph_id is not None:
            return glyph_id
        return self.glyph_name_to_id.get(".notdef")

    def has_glyph_id(self, glyph_id: int) -> bool:
        return 0 <= glyph_id < len(self.glyph_names)

    def glyph_bbox_for_gid(self, glyph_id: int) -> tuple[float, float, float, float] | None:
        if not self.has_glyph_id(glyph_id):
            return None
        glyph_name = self.glyph_names[glyph_id]
        charstring = self.charstrings.get(glyph_name) or self.charstrings.get(".notdef")
        if charstring is None:
            return None
        bounds_pen = BoundsPen(self.charstrings)
        a, b, c, d, e, f = self.font_matrix
        normalized_pen = TransformPen(
            bounds_pen,
            (a * 1000.0, b * 1000.0, c * 1000.0, d * 1000.0, e * 1000.0, f * 1000.0),
        )
        charstring.draw(normalized_pen)
        bounds = bounds_pen.bounds
        if bounds is None:
            return None
        x_min, y_min, x_max, y_max = bounds
        return (float(x_min), float(y_min), float(x_max), float(y_max))

    @staticmethod
    def binary_entries(data: bytes, pattern: re.Pattern[bytes]) -> Iterator[tuple[bytes, bytes]]:
        return binary_entries(data, pattern)

    @staticmethod
    def prepare_charstring(
        encrypted: bytes, len_iv: int, subrs: list[T1CharString]
    ) -> T1CharString:
        return T1CharString(decode_charstring(encrypted, len_iv), subrs=subrs)

    @staticmethod
    def decode_private(data: bytes, length1: int | None) -> bytes:
        return decode_eexec_payload(data, length1)


def internal_eexec_payload(data: bytes, length1: int | None) -> bytes:
    if length1 is not None and 0 < length1 < len(data):
        encrypted = data[length1:]
    else:
        marker = data.find(b"currentfile eexec")
        if marker < 0:
            raise ValueError("Type 1 eexec section is missing")
        encrypted = data[marker + len(b"currentfile eexec") :].lstrip()
    sample = encrypted[: min(len(encrypted), 512)]
    if sample and all(byte in internal_HEX_BYTES for byte in sample):
        compact = bytes(byte for byte in encrypted if byte not in b" \t\r\n")
        if len(compact) % 2:
            compact = compact[:-1]
        try:
            encrypted = bytes.fromhex(compact.decode("ascii"))
        except ValueError as exc:
            raise ValueError("invalid hexadecimal Type 1 eexec section") from exc
    decrypted = decrypt_type1(encrypted, 55665)
    if len(decrypted) < 4:
        raise ValueError("truncated Type 1 eexec section")
    return decrypted[4:]


class Type1FontProgram(internal_Type1FontProgram):
    __slots__ = ()

    @staticmethod
    def binary_entries(data: bytes, pattern: re.Pattern[bytes]) -> Iterator[tuple[bytes, bytes]]:
        for match in pattern.finditer(data):
            length = int(match.group(2))
            start = match.end()
            if length >= 0 and start + length <= len(data):
                yield match.group(1), data[start : start + length]

    @staticmethod
    def prepare_charstring(
        encrypted: bytes, len_iv: int, subrs: list[T1CharString]
    ) -> T1CharString:
        decoded = decrypt_type1(encrypted, 4330)
        return T1CharString(decoded[len_iv:] if len_iv >= 0 else decoded, subrs=subrs)

    @staticmethod
    def decode_private(data: bytes, length1: int | None) -> bytes:
        return internal_eexec_payload(data, length1)

    def glyph_bbox_for_gid(self, glyph_id: int) -> tuple[float, float, float, float] | None:
        try:
            return super().glyph_bbox_for_gid(glyph_id)
        except Exception:
            return None

    def normalized_glyph_contours(self, glyph_id: int) -> tuple[tuple[Point, ...], ...]:
        if not self.has_glyph_id(glyph_id):
            return ()
        return self.glyph_contours(self.glyph_names[glyph_id])

    def glyph_bitmap_for_gid(
        self, glyph_id: int, *, width: int = 24, height: int = 32
    ) -> tuple[int, ...]:
        contours = self.normalized_glyph_contours(glyph_id)
        return rasterize_contours(contours, width=width, height=height) if contours else ()

    def glyph_contours(self, glyph_name: str) -> tuple[tuple[Point, ...], ...]:
        charstring = self.charstrings.get(glyph_name) or self.charstrings.get(".notdef")
        if charstring is None:
            return ()
        try:
            pen = RecordingPen()
            charstring.draw(pen)
            contours = internal_recording_to_contours(pen.value)
            return transform_contours(contours, self.font_matrix)
        except Exception:
            return ()


TYPE1_ENCODING_ENTRY_RE = re.compile(rb"\bdup\s+(\d{1,3})\s+/([A-Za-z0-9_.]+)\s+put\b")


def parse_type1_font_program_encoding(font_program: bytes | memoryview) -> dict[int, str]:
    data = bytes(font_program)
    eexec_pos = data.find(b"currentfile eexec")
    if eexec_pos >= 0:
        data = data[:eexec_pos]

    differences: dict[int, str] = {}
    for match in TYPE1_ENCODING_ENTRY_RE.finditer(data):
        code = int(match.group(1))
        if 0 <= code <= 255:
            differences[code] = match.group(2).decode("latin-1")
    return differences

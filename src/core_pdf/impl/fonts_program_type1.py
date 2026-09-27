from __future__ import annotations

import re

from core_adobe_fonts.type1 import program
from core_adobe_fonts.type1.program import binary_entries, eexec_ciphertext
from core_pdf._vendor.fontTools.misc.psCharStrings import T1CharString
from core_pdf._vendor.fontTools.pens.boundsPen import BoundsPen
from core_pdf._vendor.fontTools.pens.recordingPen import (
    RecordingPen,
)
from core_pdf._vendor.fontTools.pens.transformPen import TransformPen
from core_pdf.impl.fonts_program_base import GlyphNaming, GlyphProgram
from core_pdf.impl.fonts_program_truetype import recording_to_contours
from core_pdf.impl.fonts_raster_kernel import (
    Contours,
    Point,
    transform_contours,
)
from core_pdf.impl.types import Rectangle
from core_pdf_cythonized import decrypt_type1

LEN_IV_RE = re.compile(rb"/lenIV\s+(-?\d+)\s+def\b")
FONT_MATRIX_RE = re.compile(
    rb"/FontMatrix\s*\[\s*([-+.\dEe]+)\s+([-+.\dEe]+)\s+"
    rb"([-+.\dEe]+)\s+([-+.\dEe]+)\s+([-+.\dEe]+)\s+([-+.\dEe]+)\s*\]"
)
SUBR_RE = re.compile(rb"\bdup\s+(\d+)\s+(\d+)\s+(?:RD|-\|)[ \t\r\n]")
CHARSTRING_RE = re.compile(rb"/([^\s/]+)\s+(\d+)\s+(?:RD|-\|)[ \t\r\n]")
MAX_SUBROUTINES = 4096


def type1_charstring(encrypted: bytes, len_iv: int, subrs: list[T1CharString]) -> T1CharString:
    if len_iv == -1:
        return T1CharString(encrypted, subrs=subrs)
    return T1CharString(decrypt_type1(encrypted, 4330)[len_iv:], subrs=subrs)


def eexec_payload(data: bytes, length1: int | None) -> bytes:
    decrypted = decrypt_type1(eexec_ciphertext(data, length1, tolerant=True), 55665)
    if len(decrypted) < 4:
        raise ValueError("truncated Type 1 eexec section")
    return decrypted[4:]


class Type1FontProgram(GlyphProgram):
    __slots__ = (
        "builtin_encoding",
        "charstrings",
        "font_matrix",
        "glyph_names",
        "glyph_name_to_id",
        "subrs",
    )

    def __init__(self, data: bytes, *, length1: int | None = None) -> None:
        private = eexec_payload(data, length1)
        len_iv_match = LEN_IV_RE.search(private)
        len_iv = int(len_iv_match.group(1)) if len_iv_match is not None else 4
        if len_iv < -1 or len_iv > 32:
            raise ValueError("invalid Type 1 lenIV")

        subr_data = {
            int(index): payload
            for index, payload in binary_entries(private, SUBR_RE, skip_truncated=True)
        }
        subr_count = max(subr_data, default=-1) + 1
        if subr_count > MAX_SUBROUTINES:
            raise ValueError("Type 1 subroutine index exceeds decoder limit")
        empty = T1CharString(b"\x0b", subrs=[])
        subrs = [empty for _ in range(subr_count)]
        for index, encrypted in subr_data.items():
            subrs[index] = type1_charstring(encrypted, len_iv, subrs)
        for subr in subrs:
            subr.subrs = subrs
        self.subrs = subrs

        charstrings = {
            name.decode("latin-1"): payload
            for name, payload in binary_entries(private, CHARSTRING_RE, skip_truncated=True)
        }
        self.charstrings = {
            name: type1_charstring(encrypted, len_iv, subrs)
            for name, encrypted in charstrings.items()
        }
        if not self.charstrings:
            raise ValueError("Type 1 CharStrings are missing")
        self.glyph_names = tuple(self.charstrings)
        self.glyph_name_to_id = {name: gid for gid, name in enumerate(self.glyph_names)}

        matrix_match = FONT_MATRIX_RE.search(data)
        self.font_matrix = (
            tuple(float(value) for value in matrix_match.groups())
            if matrix_match is not None
            else (0.001, 0.0, 0.0, 0.001, 0.0, 0.0)
        )
        self.builtin_encoding = parse_type1_font_program_encoding(data)

    def font_builtin_encoding(self) -> tuple[dict[int, str], bool]:
        encoding = self.builtin_encoding
        return encoding, bool(encoding)

    def glyph_id_for_name(self, glyph_name: str) -> int | None:
        glyph_id = self.glyph_name_to_id.get(glyph_name)
        if glyph_id is not None:
            return glyph_id
        return self.glyph_name_to_id.get(".notdef")

    def has_glyph_id(self, glyph_id: int) -> bool:
        return 0 <= glyph_id < len(self.glyph_names)

    def glyph_id_for_code(self, code: int, naming: GlyphNaming) -> int | None:
        return self.glyph_id_for_name(naming.glyph_name(code))

    def glyph_bbox_for_gid(self, glyph_id: int) -> Rectangle | None:
        try:
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
        except Exception:
            return None

    def normalized_glyph_contours(self, glyph_id: int) -> Contours:
        if not self.has_glyph_id(glyph_id):
            return ()
        return self.glyph_contours(self.glyph_names[glyph_id])

    def glyph_contours(self, glyph_name: str) -> tuple[tuple[Point, ...], ...]:
        charstring = self.charstrings.get(glyph_name) or self.charstrings.get(".notdef")
        if charstring is None:
            return ()
        try:
            pen = RecordingPen()
            charstring.draw(pen)
            contours = recording_to_contours(pen.value)
            return transform_contours(contours, self.font_matrix)
        except Exception:
            return ()


def parse_type1_font_program_encoding(font_program: bytes | memoryview) -> dict[int, str]:
    return program.parse_type1_font_program_encoding(font_program, skip_out_of_range=True)

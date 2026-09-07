"""Application outline and damaged-program adapter for Type 1 fonts."""

from __future__ import annotations

import re
from collections.abc import Iterator

from core_pdf._vendor.fontTools.misc.psCharStrings import T1CharString
from core_pdf._vendor.fontTools.pens.recordingPen import RecordingPen
from core_pdf.impl._impl.fonts.font_program_truetype import internal_recording_to_contours
from core_pdf.impl._impl.fonts.raster_kernel import Point, rasterize_contours, transform_contours
from core_pdf.impl.spec.s_09_fonts.font_program_type1 import Type1FontProgram as PdfType1FontProgram
from core_pdf.impl.spec.s_09_fonts.font_program_type1 import internal_decrypt, internal_HEX_BYTES


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
    decrypted = internal_decrypt(encrypted, 55665)
    if len(decrypted) < 4:
        raise ValueError("truncated Type 1 eexec section")
    return decrypted[4:]


class Type1FontProgram(PdfType1FontProgram):
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
        decoded = internal_decrypt(encrypted, 4330)
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

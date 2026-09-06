"""OpenType CFF/CFF2 container semantics."""

from __future__ import annotations

from io import BytesIO

from core_pdf._vendor.fontTools.ttLib import TTFont


def parse_opentype_program(data: bytes) -> TTFont:
    font = TTFont(BytesIO(data), lazy=True, recalcBBoxes=False, recalcTimestamp=False)
    if not ({"CFF ", "CFF2"} & set(font.keys())):
        raise ValueError("OpenType font has no CFF outline table")
    return font

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from io import BytesIO
from typing import Any

from core_pdf._vendor.fontTools.ttLib import TTFont
from core_pdf.impl.fonts_program_base import NULL_PROGRAM, GlyphProgram
from core_pdf.impl.fonts_program_cff import CFFFont
from core_pdf.impl.fonts_program_truetype import (
    OpenTypeFontProgram,
    TrueTypeFontProgram,
    cached_truetype_program,
)
from core_pdf.impl.fonts_program_type1 import Type1FontProgram
from core_pdf.impl.fonts_widths import recover_descendant
from core_pdf.impl.pdf_values import recover_pdf_name
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_09_fonts.dictionaries import (
    FontProgramInputs,
    prepare_font_program_inputs,
)


def tt_font(inputs: FontProgramInputs) -> TrueTypeFontProgram | None:
    if inputs.subtype not in {"CIDFontType2", "TrueType"}:
        return None
    font_file = inputs.font_file2
    if font_file is None:
        return None
    cid_to_gid = None
    if inputs.descendant is not None:
        cid_to_gid_obj = inputs.descendant.get("CIDToGIDMap")
        if isinstance(cid_to_gid_obj, PdfStream):
            cid_to_gid = cid_to_gid_obj.data
    try:
        return cached_truetype_program(
            font_file.data, cid_to_gid, use_cmap=inputs.descendant is None
        )
    except ValueError:
        return None


def cff_font(inputs: FontProgramInputs) -> CFFFont | None:
    if inputs.descendant is not None:
        if inputs.subtype != "CIDFontType0":
            return None
    elif inputs.subtype not in {"Type1", "MMType1"}:
        return None
    font_file = inputs.font_file3
    if font_file is None:
        return None
    subtype = recover_pdf_name(font_file.dictionary.get("Subtype"))
    if inputs.descendant is None and subtype not in {"Type1C", "OpenType"}:
        return None
    font_data = source = font_file.data
    if subtype == "OpenType":
        cff_table = extract_cff_table(font_data)
        if cff_table is None:
            return None
        font_data = cff_table
    try:
        return CFFFont(font_data, source_size=len(source))
    except ValueError:
        return None


def extract_cff_table(data: bytes) -> bytes | None:
    font: TTFont | None = None
    try:
        font = TTFont(BytesIO(data), lazy=True, recalcBBoxes=False, recalcTimestamp=False)
        reader = font.reader
        if reader is None:
            return None
        table = reader.tables.get("CFF ")
        return table.loadData(reader.file) if table is not None else None
    except Exception:
        return None
    finally:
        if font is not None:
            with suppress(AttributeError):
                font.close()


def type1_font(inputs: FontProgramInputs) -> Type1FontProgram | None:
    if inputs.original_subtype not in {"Type1", "MMType1"}:
        return None
    font_file = inputs.font_file
    if font_file is None:
        return None
    length1_value = font_file.dictionary.get("Length1")
    try:
        length1 = int(length1_value) if isinstance(length1_value, (int, float)) else None
        return Type1FontProgram(font_file.data, length1=length1)
    except TypeError, ValueError, OverflowError:
        return None


def opentype_font(inputs: FontProgramInputs) -> OpenTypeFontProgram | None:
    font_file = inputs.font_file3
    if font_file is None:
        return None
    if recover_pdf_name(font_file.dictionary.get("Subtype")) != "OpenType":
        return None
    try:
        return OpenTypeFontProgram(font_file.data)
    except ValueError:
        return None


def recover_descriptor(value: object) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def recover_font_file(descriptor: dict[str, Any] | None, key: str) -> PdfStream | None:
    value = descriptor.get(key) if descriptor is not None else None
    return value if isinstance(value, PdfStream) else None


FONT_PROGRAM_LOADERS: tuple[Callable[[FontProgramInputs], GlyphProgram | None], ...] = (
    cff_font,
    tt_font,
    type1_font,
    opentype_font,
)


def font_program_for_pdf_font(font: dict[str, Any]) -> GlyphProgram | None:
    inputs = prepare_font_program_inputs(
        font,
        read_name=recover_pdf_name,
        read_descendant=recover_descendant,
        read_descriptor=recover_descriptor,
        read_font_file=recover_font_file,
    )
    for loader in FONT_PROGRAM_LOADERS:
        program = loader(inputs)
        if program is not None:
            return program
    return None


def load_glyph_program(font: dict[str, Any]) -> GlyphProgram:
    program = font_program_for_pdf_font(font)
    return program if program is not None else NULL_PROGRAM

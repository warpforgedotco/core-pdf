"""PyMuPDF text policy over native PDF glyph observations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from core_pdf.api.compat._shared import BBox, float32
from core_pdf.api.compat.pymupdf.geometry import Rect
from core_pdf.api.document import PdfPage
from core_pdf.impl._impl.fonts.decoder import FontDecoder
from core_pdf.impl._impl.model.glyphs import GlyphObservation

# Metrics of the reference reader's standard-font substitutes, in em units.
internal_STANDARD_METRICS = {
    "Helvetica": (1.075, -0.299),
    "Helvetica-Bold": (1.070, -0.307),
    "Helvetica-Oblique": (1.070, -0.284),
    "Helvetica-BoldOblique": (1.073, -0.309),
    "Times-Roman": (1.053, -0.281),
    "Times-Bold": (1.044, -0.341),
    "Times-Italic": (0.951, -0.270),
    "Times-BoldItalic": (0.972, -0.324),
    "Courier": (0.932, -0.317),
    "Courier-Bold": (1.007, -0.393),
    "Courier-Oblique": (0.920, -0.317),
    "Courier-BoldOblique": (0.997, -0.393),
    "Symbol": (1.010, -0.293),
    "ZapfDingbats": (0.819, -0.144),
}


# Built-in symbolic fonts use raw character codes for substitute advances,
# even when an explicit encoding decodes those codes as Latin characters.
# Values in thousandths of an em, measured with the 1.28.2 reference reader.
# fmt: off
internal_SYMBOL_WIDTHS = {
    "Symbol": (
        460, 460, 460, 460, 460, 460, 460, 460,  # 0x00
        460, 460, 460, 460, 460, 460, 460, 460,  # 0x08
        460, 460, 460, 460, 460, 460, 460, 460,  # 0x10
        460, 460, 460, 460, 460, 460, 460, 460,  # 0x18
        250, 333, 713, 500, 549, 833, 778, 439,  # 0x20
        333, 333, 500, 549, 250, 549, 250, 278,  # 0x28
        500, 500, 500, 500, 500, 500, 500, 500,  # 0x30
        500, 500, 278, 278, 549, 549, 549, 444,  # 0x38
        549, 722, 667, 722, 612, 611, 763, 603,  # 0x40
        722, 333, 631, 722, 686, 889, 722, 722,  # 0x48
        768, 741, 556, 592, 611, 690, 439, 768,  # 0x50
        645, 795, 611, 333, 863, 333, 658, 500,  # 0x58
        500, 631, 549, 549, 494, 439, 521, 411,  # 0x60
        603, 329, 603, 549, 549, 576, 521, 549,  # 0x68
        549, 521, 549, 603, 439, 576, 713, 686,  # 0x70
        493, 686, 494, 480, 200, 480, 549, 460,  # 0x78
        460, 460, 460, 460, 460, 460, 460, 460,  # 0x80
        460, 460, 460, 460, 460, 460, 460, 460,  # 0x88
        460, 460, 460, 460, 460, 460, 460, 460,  # 0x90
        460, 460, 460, 460, 460, 460, 460, 460,  # 0x98
        250, 620, 247, 549, 167, 713, 500, 753,  # 0xA0
        753, 753, 753, 1042, 713, 603, 987, 603,  # 0xA8
        400, 549, 411, 549, 549, 576, 494, 460,  # 0xB0
        549, 549, 549, 549, 1000, 603, 1000, 658,  # 0xB8
        823, 686, 795, 987, 768, 768, 823, 768,  # 0xC0
        768, 713, 713, 713, 713, 713, 713, 713,  # 0xC8
        768, 713, 790, 790, 890, 823, 549, 549,  # 0xD0
        713, 603, 603, 1042, 987, 603, 987, 603,  # 0xD8
        494, 329, 790, 790, 786, 713, 384, 384,  # 0xE0
        384, 384, 384, 384, 494, 494, 494, 494,  # 0xE8
        460, 329, 274, 686, 686, 686, 384, 549,  # 0xF0
        384, 384, 384, 384, 494, 494, 494, 460,  # 0xF8
    ),
    "ZapfDingbats": (
        788, 788, 788, 788, 788, 788, 788, 788,  # 0x00
        788, 788, 788, 788, 788, 788, 788, 788,  # 0x08
        788, 788, 788, 788, 788, 788, 788, 788,  # 0x10
        788, 788, 788, 788, 788, 788, 788, 788,  # 0x18
        278, 974, 961, 974, 980, 719, 789, 790,  # 0x20
        791, 690, 960, 939, 549, 855, 911, 933,  # 0x28
        911, 945, 974, 755, 846, 762, 761, 571,  # 0x30
        677, 763, 760, 759, 754, 494, 552, 537,  # 0x38
        577, 692, 786, 788, 788, 790, 793, 794,  # 0x40
        816, 823, 789, 841, 823, 833, 816, 831,  # 0x48
        923, 744, 723, 749, 790, 792, 695, 776,  # 0x50
        768, 792, 759, 707, 708, 682, 701, 826,  # 0x58
        815, 789, 789, 707, 687, 696, 689, 786,  # 0x60
        787, 713, 791, 785, 791, 873, 761, 762,  # 0x68
        762, 759, 759, 892, 892, 788, 784, 438,  # 0x70
        138, 277, 415, 392, 392, 668, 668, 788,  # 0x78
        788, 788, 788, 788, 788, 788, 788, 788,  # 0x80
        788, 788, 788, 788, 788, 788, 788, 788,  # 0x88
        788, 788, 788, 788, 788, 788, 788, 788,  # 0x90
        788, 788, 788, 788, 788, 788, 788, 788,  # 0x98
        788, 732, 544, 544, 910, 667, 760, 760,  # 0xA0
        776, 595, 694, 626, 788, 788, 788, 788,  # 0xA8
        788, 788, 788, 788, 788, 788, 788, 788,  # 0xB0
        788, 788, 788, 788, 788, 788, 788, 788,  # 0xB8
        788, 788, 788, 788, 788, 788, 788, 788,  # 0xC0
        788, 788, 788, 788, 788, 788, 788, 788,  # 0xC8
        788, 788, 788, 788, 894, 838, 1016, 458,  # 0xD0
        748, 924, 748, 918, 927, 928, 928, 834,  # 0xD8
        873, 828, 924, 924, 917, 930, 931, 463,  # 0xE0
        883, 836, 836, 867, 867, 696, 696, 874,  # 0xE8
        788, 874, 760, 946, 771, 865, 771, 888,  # 0xF0
        967, 888, 831, 873, 927, 970, 788, 788,  # 0xF8
    ),
}
# fmt: on


def internal_union(boxes: list[BBox]) -> BBox:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


@dataclass
class internal_Character:
    text: str
    bbox: BBox
    origin: tuple[float, float]
    end: tuple[float, float]
    size: float
    direction: tuple[float, float]


@dataclass
class internal_Line:
    characters: list[internal_Character] = field(default_factory=list)

    @property
    def bbox(self) -> BBox:
        return internal_union([char.bbox for char in self.characters])

    @property
    def text(self) -> str:
        return "".join(char.text for char in self.characters)


@dataclass
class TextProjection:
    blocks: list[list[internal_Line]] = field(default_factory=list)

    def text(self, *, sort: bool = False) -> str:
        blocks = (
            sorted(self.blocks, key=lambda b: (b[0].bbox[3], b[0].bbox[0])) if sort else self.blocks
        )
        return "".join(line.text + "\n" for block in blocks for line in block)

    def words(self, *, sort: bool = False, delimiters: str = "") -> list[tuple[Any, ...]]:
        output: list[tuple[Any, ...]] = []
        for block_index, block in enumerate(self.blocks):
            for line_index, line in enumerate(block):
                pending: list[internal_Character] = []
                index = 0
                for char in [*line.characters, None]:
                    if char is not None and not char.text.isspace() and char.text not in delimiters:
                        pending.append(char)
                        continue
                    if pending:
                        output.append(
                            (
                                *internal_union([c.bbox for c in pending]),
                                "".join(c.text for c in pending),
                                block_index,
                                line_index,
                                index,
                            )
                        )
                        pending = []
                        index += 1
        if sort:
            output.sort(key=lambda word: (word[3], word[0]))
        return output

    def block_records(self, *, sort: bool = False) -> list[tuple[Any, ...]]:
        records = [
            (
                *internal_union([line.bbox for line in block]),
                "".join(line.text + "\n" for line in block),
                index,
                0,
            )
            for index, block in enumerate(self.blocks)
        ]
        return sorted(records, key=lambda block: (block[3], block[0])) if sort else records


def internal_metrics(glyph: GlyphObservation) -> tuple[float, float]:
    decoder = glyph.font_decoder
    if not isinstance(decoder, FontDecoder):
        return float32(0.8), float32(-0.2)
    name = (glyph.font_name or "").split("+", 1)[-1]
    if decoder.font_program is None and name in internal_STANDARD_METRICS:
        ascender, descender = internal_STANDARD_METRICS[name]
    else:
        ascender, descender = decoder.ascent / 1000, decoder.descent / 1000
    height = ascender - descender
    if 0 < height < 1:
        ascender, descender = ascender / height, descender / height
    return float32(ascender), float32(descender)


def capture_text(page: PdfPage, *, flags: int = 195, clip: object = None) -> TextProjection:
    crop = Rect(page.crop_box or page.media_box)
    unit_value = page.document.resolver.resolve(page.page_dict.get("UserUnit"))
    unit = float32(unit_value) if isinstance(unit_value, (int, float)) else 1.0
    media = Rect(page.media_box or crop)
    bounds = Rect(
        (media.x0 - crop.x0) * unit,
        (crop.y1 - media.y1) * unit,
        (media.x1 - crop.x0) * unit,
        (crop.y1 - media.y0) * unit,
    ).normalize()
    clip_box = Rect(clip) if clip is not None else bounds
    projection = TextProjection()
    previous_raw_end: tuple[float, float] | None = None
    previous_end: tuple[float, float] | None = None
    previous_group: object = None
    previous: internal_Character | None = None
    for glyph in page.get_page_program().glyphs:
        if not glyph.text or glyph.baseline is None or not glyph.font_size:
            continue
        x, y, ex, ey = glyph.baseline
        provenance = dict(glyph.provenance)
        group = (
            glyph.text_object_id,
            provenance.get("line_matrix_origin"),
            provenance.get("text_matrix"),
        )
        if group == previous_group and previous_raw_end is not None and previous_end is not None:
            px = float32(previous_end[0] + float32((x - previous_raw_end[0]) * unit))
            py = float32(previous_end[1] - float32((y - previous_raw_end[1]) * unit))
        else:
            px = float32(float32(float32(x) - float32(crop.x0)) * unit)
            py = float32(float32(float32(crop.y1) - float32(y)) * unit)
        # PDF widths use thousandths of an em. Quantize before advancing the reader cursor.
        scale = glyph.effective_font_size
        distance = math.hypot(ex - x, ey - y)
        width = distance / scale * 1000 if scale else 0
        builtin_widths = internal_SYMBOL_WIDTHS.get(glyph.font_name or "")
        if (
            builtin_widths is not None
            and isinstance(glyph.font_decoder, FontDecoder)
            and glyph.font_decoder.font_program is None
            and glyph.char_code is not None
            and 0 <= glyph.char_code < 256
            and "Widths" not in glyph.font_decoder.font
        ):
            width = builtin_widths[glyph.char_code]
        advance = float32(float32(scale) * float32(width * float32(0.001)))
        dx, dy = ((ex - x) / distance, (ey - y) / distance) if distance else (1.0, 0.0)
        end = (
            float32(px + float32(advance * dx * unit)),
            float32(py - float32(advance * dy * unit)),
        )
        previous_raw_end, previous_end, previous_group = (ex, ey), end, group
        ascender, descender = internal_metrics(glyph)
        matrix = glyph.glyph_transform
        if matrix is None:
            vx = float32(-dy * glyph.effective_font_height * unit)
            vy = float32(-dx * glyph.effective_font_height * unit)
        else:
            vx, vy = float32(matrix[2] * 1000 * unit), float32(-matrix[3] * 1000 * unit)
        corners = [
            (float32(ox + float32(vx * metric)), float32(oy + float32(vy * metric)))
            for ox, oy in ((px, py), end)
            for metric in (ascender, descender)
        ]
        bbox: BBox = (
            min(p[0] for p in corners),
            min(p[1] for p in corners),
            max(p[0] for p in corners),
            max(p[1] for p in corners),
        )
        if (clip is not None or flags & 64) and not clip_box.intersects(bbox):
            continue
        size = abs(float32(glyph.effective_font_height * unit))
        char = internal_Character(glyph.text, bbox, (px, py), end, size, (dx, -dy))
        new_line = previous is None
        new_block = previous is None
        gap = 0.0
        if previous is not None:
            ux, uy = previous.direction
            delta_x, delta_y = px - previous.end[0], py - previous.end[1]
            gap = delta_x * ux + delta_y * uy
            perpendicular = delta_y * ux - delta_x * uy
            same_direction = abs(ux - dx) < 0.01 and abs(uy + dy) < 0.01
            em = max(size, 1e-9)
            new_line = not same_direction or abs(perpendicular) > em * 0.8 or abs(gap) > em * 0.8
            new_block = not same_direction or abs(perpendicular) > em * 1.5
        if new_block:
            projection.blocks.append([])
        if new_line:
            projection.blocks[-1].append(internal_Line())
        line = projection.blocks[-1][-1]
        if (
            not new_line
            and previous is not None
            and gap > size * 0.15
            and not flags & 8
            and not previous.text.isspace()
            and not char.text.isspace()
        ):
            line.characters.append(
                internal_Character(
                    " ",
                    (
                        previous.bbox[2],
                        min(previous.bbox[1], bbox[1]),
                        bbox[0],
                        max(previous.bbox[3], bbox[3]),
                    ),
                    previous.end,
                    char.origin,
                    size,
                    char.direction,
                )
            )
        line.characters.append(char)
        previous = char
    return projection


__all__ = ("TextProjection", "capture_text")

# SPDX-License-Identifier: AGPL-3.0-only
"""High-level pdfminer.six-compatible extraction functions."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

from core_pdf.impl.engine.extraction.document import PdfDocument
from core_pdf.impl.engine.layout.glyphs import GlyphObservation
from core_pdf.impl.integrations.pdfminer.layout import (
    LAParams,
    LTAnno,
    LTChar,
    LTContainer,
    LTExpandableContainer,
    LTImage,
    LTPage,
    LTTextBox,
    LTTextBoxHorizontal,
    LTTextBoxVertical,
    LTTextLine,
    LTTextLineHorizontal,
    LTTextLineVertical,
)
from core_pdf.impl.types import PdfSource


class _ImageStream:
    def __init__(self, attrs: dict[str, Any]) -> None:
        self.attrs = attrs

    def get_any(self, keys: tuple[str, ...], default: Any = None) -> Any:
        for key in keys:
            if key in self.attrs:
                return self.attrs[key]
        return default


def _provenance_value(provenance: tuple[tuple[str, object], ...], name: str, default: Any) -> Any:
    for key, value in provenance:
        if key == name:
            return value
    return default


def _glyph_chars(run: Any) -> Iterator[LTChar]:
    for cluster in run.glyph_clusters:
        glyphs = cluster.glyphs
        if not glyphs:
            continue
        for glyph in glyphs:
            yield _ltchar_from_glyph(glyph, run)


def _ltchar_from_glyph(glyph: GlyphObservation, run: Any) -> LTChar:
    matrix = glyph.device_matrix or glyph.text_matrix
    if matrix is None:
        matrix = (1.0, 0.0, 0.0, 1.0, glyph.advance_bbox[0], glyph.advance_bbox[1])
    baseline = glyph.baseline
    if baseline is None:
        adv = glyph.advance_bbox[2] - glyph.advance_bbox[0]
    elif glyph.writing_mode == "vertical":
        adv = baseline[3] - baseline[1]
    else:
        adv = baseline[2] - baseline[0]
    render_mode = int(_provenance_value(run.provenance, "text_render_mode", 0))
    return LTChar.from_core(
        text=glyph.text,
        bbox=glyph.advance_bbox,
        matrix=matrix,
        fontname=glyph.font_name or run.font_name or "unknown",
        fontsize=glyph.font_size or run.font_size,
        adv=adv,
        upright=glyph.rotation_angle % 180 == 0,
        rendermode=render_mode,
    )


def _line_from_core(line: Any, laparams: LAParams) -> LTTextLine:
    result: LTTextLine
    if line.is_vertical:
        result = LTTextLineVertical(laparams.word_margin)
    else:
        result = LTTextLineHorizontal(laparams.word_margin)
    chars: list[LTChar] = []
    runs = sorted(line.runs, key=lambda item: item.order)
    text_parts: list[str] = []
    previous_run: Any = None
    for run in runs:
        if previous_run is not None and text_parts:
            gap = run.x0 - previous_run.x1
            margin = laparams.word_margin * max(run.space_width, 1.0)
            if gap > margin and not text_parts[-1].endswith(tuple(" \t\r\n")):
                text_parts.append(" ")
        text_parts.append(run.text)
        previous_run = run
        run_chars = tuple(_glyph_chars(run))
        if run_chars:
            chars.extend(run_chars)
            continue
        # Recovery and OCR runs may not have glyph observations. Preserve text
        # and geometry so consumers still receive a complete layout tree.
        chars.append(
            LTChar.from_core(
                text=run.text,
                bbox=run.advance_bbox,
                matrix=(1.0, 0.0, 0.0, 1.0, run.tx, run.ty),
                fontname=run.font_name or "unknown",
                fontsize=run.font_size,
                adv=run.x1 - run.x0,
                upright=run.rotation_angle % 180 == 0,
                rendermode=int(_provenance_value(run.provenance, "text_render_mode", 0)),
            )
        )
    target_text = "".join(text_parts)
    position = 0
    for char in chars:
        char_text = char.get_text()
        next_position = target_text.find(char_text, position)
        if next_position >= position:
            if next_position > position:
                LTContainer.add(result, LTAnno(target_text[position:next_position]))
            position = next_position + len(char_text)
        LTExpandableContainer.add(result, char)
    if position < len(target_text):
        LTContainer.add(result, LTAnno(target_text[position:]))
    # pdfminer adds the virtual line break during layout analysis.
    LTContainer.add(result, LTAnno("\n"))
    return result


def _box_from_core(line: Any, laparams: LAParams) -> LTTextBox:
    result: LTTextBox = LTTextBoxVertical() if line.is_vertical else LTTextBoxHorizontal()
    result.add(_line_from_core(line, laparams))
    return result


def _page_layout(page: Any, laparams: LAParams) -> LTPage:
    media_box = page.media_box or (0.0, 0.0, page.width, page.height)
    layout = LTPage(page.page_number, media_box, page.rotation)
    for annotation in page.get_annotations():
        if annotation.subtype != "Link" or annotation.rect is None:
            continue
        link_box = LTTextBoxHorizontal()
        link_box.set_bbox(annotation.rect)
        layout.add(link_box)
    for line in page.get_text_lines():
        layout.add(_box_from_core(line, laparams))
    for index, image in enumerate(page.extract_images()):
        bbox = image.get("bbox")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        attrs = {
            "Width": image.get("width"),
            "Height": image.get("height"),
            "BitsPerComponent": image.get("bits_per_component"),
            "ColorSpace": image.get("color_space"),
        }
        layout.add(LTImage(str(image.get("name", index)), _ImageStream(attrs), bbox))
    return layout


def extract_pages(
    pdf_file: PdfSource,
    password: str = "",
    page_numbers: set[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    laparams: LAParams | None = None,
) -> Iterator[LTPage]:
    """Yield page layouts with the same public shape as pdfminer.high_level.extract_pages."""
    del caching
    params = laparams or LAParams()
    with PdfDocument.open(pdf_file, password=password) as document:
        yielded = 0
        for index, page in enumerate(document.pages):
            if page_numbers is not None and index not in page_numbers:
                continue
            if maxpages and yielded >= maxpages:
                break
            yield _page_layout(page, params)
            yielded += 1


def extract_text(
    pdf_file: PdfSource,
    password: str = "",
    page_numbers: set[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    codec: str = "utf-8",
    laparams: LAParams | None = None,
) -> str:
    """Return text using pdfminer.six-compatible arguments and page separators."""
    del caching, codec, laparams
    with PdfDocument.open(pdf_file, password=password) as document:
        texts: list[str] = []
        for index, page in enumerate(document.pages):
            if page_numbers is not None and index not in page_numbers:
                continue
            if maxpages and len(texts) >= maxpages:
                break
            texts.append(cast(Any, page).extract_text())
        return "\f".join(texts) + ("\f" if texts else "")


__all__ = ("extract_pages", "extract_text")

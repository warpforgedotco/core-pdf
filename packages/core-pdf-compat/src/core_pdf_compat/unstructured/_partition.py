from __future__ import annotations

import re
from os import PathLike
from pathlib import Path
from typing import Any

from core_pdf import PdfDocument
from core_pdf.impl.exceptions import PdfEmptySourceError, PdfError, PdfPasswordError
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf_compat.pdfminer._extract import extract_document_pages
from core_pdf_compat.pdfminer._layout import LAParams, LTFigure, LTTextBox
from core_pdf_compat.pdfminer._projection import UNSTRUCTURED_POLICY

from ._classification import (
    BULLET,
    classify_element,
)
from ._elements import (
    Element,
    ElementMetadata,
    ListItem,
    PageBreak,
)
from ._regions import (
    TextRegion,
    clean_text,
    combine_list_regions,
    figure_text_snippets,
    layout_regions,
    region_order,
)

GRAPHICS_OPS = re.compile(
    rb"(?:^|(?<=\s))(?:m|l|c|v|y|h|re|S|s|f|F|f\*|B|B\*|b|b\*|n|W|W\*|cm|q|Q|"
    rb"Do|g|G|rg|RG|k|K|cs|CS|w|J|j|M|d|i|gs)(?=\s|$)"
)

TEXT_OPS = re.compile(rb"(?:^|(?<=\s))(?:Tj|TJ|'|\"|Tf|Td|TD|Tm|T\*|BT|ET)(?=\s|$)")


def pdf_too_complex(filename: object, password: str) -> bool:
    # Unstructured's pre-flight opens the file with every revision scanned,
    # which its layout pass then does not, so the two opens stay separate.
    # A Type0 font with no descendants is judged here, as the layout pass
    # under UNSTRUCTURED_POLICY does not check resources.
    with PdfDocument.open(filename, password=password) as document:  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        strict_xref_error = document.strict_xref_validation_error()
        if strict_xref_error == "invalid hex string" or (
            strict_xref_error is not None and document.raw_data.find(b"%%EOF") < 0
        ):
            return True
        if document.xref_recovery_reason == "xref section loop detected":
            return True
        for page in document.pages:
            fonts = page.resources.get("Font")
            if not isinstance(fonts, dict):
                continue
            for raw_font in fonts.values():
                font = document.resolver.resolve(raw_font)
                if (
                    isinstance(font, dict)
                    and recover_pdf_name(font.get("Subtype")) == "Type0"
                    and font.get("DescendantFonts") is None
                ):
                    return True
        if len(document.raw_data) < 1_048_576:
            return False
        for page in document.pages:
            raw_data = b"".join(stream.data for stream in page.content_streams)
            if len(raw_data) < 100_000:
                continue
            graphics = len(GRAPHICS_OPS.findall(raw_data))
            if graphics <= 10_000:
                continue
            text = len(TEXT_OPS.findall(raw_data))
            if graphics / max(text, 1) > 20.0:
                return True
    return False


def partition_pdf(filename: object, **kwargs: object) -> list[Element]:
    if (
        isinstance(filename, (str, PathLike))
        and not (isinstance(filename, str) and filename.startswith("%PDF"))
        and not Path(filename).exists()
    ):
        return []
    include_page_breaks = bool(kwargs.pop("include_page_breaks", False))
    include_metadata = bool(kwargs.pop("include_metadata", True))
    word_margin = float(kwargs.pop("pdfminer_word_margin", 0.185) or 0.185)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    password = str(kwargs.pop("password", "") or "")
    try:
        if pdf_too_complex(filename, password):
            return []
    except PdfPasswordError, PdfEmptySourceError:
        return []
    result: list[Element] = []
    document = PdfDocument.open(
        filename,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        password=password,
        recovery_scan_all_revisions=False,
    )
    try:
        pages = extract_document_pages(
            document, LAParams(word_margin=word_margin), policy=UNSTRUCTURED_POLICY
        )
        source_pages = iter(document.pages)
        for page in pages:
            source_page = next(source_pages, None)
            text_boxes = [item for item in page if isinstance(item, LTTextBox)]
            regions = layout_regions(text_boxes)
            for figure in (item for item in page if isinstance(item, LTFigure)):
                for figure_text in figure_text_snippets(figure):
                    if cleaned_figure_text := clean_text(figure_text):
                        regions.append(TextRegion(cleaned_figure_text, figure.bbox))
            fields: tuple[Any, ...] | list[Any]
            if source_page is None:
                fields = ()
            else:
                try:
                    fields = source_page.get_fields()
                except PdfError, ValueError:
                    fields = ()
            field_regions: list[TextRegion] = []
            for field in fields:
                if field.rect is None or field.type not in {"Tx", "Ch"} or not field.value_text:
                    continue
                left, bottom, right, top = (float(value) for value in field.rect)
                field_regions.append(
                    TextRegion(
                        field.value_text,
                        (left, bottom, right, top),
                    )
                )
            regions = combine_list_regions(regions, page.height)
            regions.extend(field_regions)
            for element_index in region_order(regions, page.height):
                region = regions[element_index]
                text, bbox = region.text, region.bbox
                element_class = region.element_class or classify_element(text, bbox, page.height)
                if element_class is ListItem and region.element_class is None:
                    text = BULLET.sub("", text, count=1).strip()
                metadata = (
                    ElementMetadata(
                        {
                            "element_id": f"p{page.pageid}-e{element_index}",
                            "page_number": page.pageid,
                            "bbox": bbox,
                        }
                    )
                    if include_metadata
                    else ElementMetadata()
                )
                result.append(element_class(text, metadata))
            if include_page_breaks:
                result.append(PageBreak("", ElementMetadata(page_number=page.pageid)))
    finally:
        document.close()
    return result

from __future__ import annotations

import re
from os import PathLike
from pathlib import Path
from typing import Any, cast

from core_pdf import PdfDocument
from core_pdf.impl._impl.pdf_names import recover_pdf_name
from core_pdf.impl.exceptions import PdfError, PdfSourceError, PdfUnsupportedError
from core_pdf_compat.pdfminer._extract import internal_extract_document_pages
from core_pdf_compat.pdfminer._layout import LAParams, LTFigure, LTTextBox

from ._classification import (
    internal_BULLET,
    internal_element_class,
)
from ._elements import (
    Element,
    ElementMetadata,
    ListItem,
    PageBreak,
)
from ._regions import (
    internal_clean_text,
    internal_combine_list_regions,
    internal_figure_text_snippets,
    internal_layout_regions,
    internal_region_order,
    internal_TextRegion,
)

internal_GRAPHICS_OPS = re.compile(
    rb"(?:^|(?<=\s))(?:m|l|c|v|y|h|re|S|s|f|F|f\*|B|B\*|b|b\*|n|W|W\*|cm|q|Q|"
    rb"Do|g|G|rg|RG|k|K|cs|CS|w|J|j|M|d|i|gs)(?=\s|$)"
)

internal_TEXT_OPS = re.compile(rb"(?:^|(?<=\s))(?:Tj|TJ|'|\"|Tf|Td|TD|Tm|T\*|BT|ET)(?=\s|$)")


def internal_pdf_too_complex(filename: object, password: str) -> bool:
    with PdfDocument.open(cast(Any, filename), password=password) as document:
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
            graphics = len(internal_GRAPHICS_OPS.findall(raw_data))
            if graphics <= 10_000:
                continue
            text = len(internal_TEXT_OPS.findall(raw_data))
            if graphics / max(text, 1) > 20.0:
                return True
    return False


def partition_pdf(filename: object, **kwargs: object) -> list[Element]:
    if (
        isinstance(filename, (str, PathLike))
        and not (isinstance(filename, str) and filename.startswith("%PDF"))
        and not Path(cast(str | PathLike[str], filename)).exists()
    ):
        return []
    include_page_breaks = bool(kwargs.pop("include_page_breaks", False))
    include_metadata = bool(kwargs.pop("include_metadata", True))
    word_margin = float(cast(Any, kwargs.pop("pdfminer_word_margin", 0.185) or 0.185))
    password = str(kwargs.pop("password", "") or "")
    try:
        if internal_pdf_too_complex(filename, password):
            return []
    except PdfUnsupportedError as error:
        if str(error) == "Incorrect password":
            return []
        raise
    except PdfSourceError as error:
        if str(error) == "PDF source is empty":
            return []
        raise
    result: list[Element] = []
    document = PdfDocument.open(
        cast(Any, filename),
        password=password,
        recovery_scan_all_revisions=False,
    )
    try:
        pages = internal_extract_document_pages(
            document, LAParams(word_margin=word_margin), unstructured_mode=True
        )
        source_pages = iter(document.pages)
        for page in pages:
            source_page = next(source_pages, None)
            text_boxes = [item for item in page if isinstance(item, LTTextBox)]
            regions = internal_layout_regions(text_boxes)
            for figure in (item for item in page if isinstance(item, LTFigure)):
                for figure_text in internal_figure_text_snippets(figure):
                    if cleaned_figure_text := internal_clean_text(figure_text):
                        regions.append(internal_TextRegion(cleaned_figure_text, figure.bbox))
            fields: tuple[Any, ...] | list[Any]
            if source_page is None:
                fields = ()
            else:
                try:
                    fields = source_page.get_fields()
                except PdfError, ValueError:
                    fields = ()
            field_regions: list[internal_TextRegion] = []
            for field in fields:
                if field.rect is None or field.type not in {"Tx", "Ch"} or not field.value_text:
                    continue
                left, bottom, right, top = (float(value) for value in field.rect)
                field_regions.append(
                    internal_TextRegion(
                        field.value_text,
                        (left, bottom, right, top),
                    )
                )
            regions = internal_combine_list_regions(regions, page.height)
            regions.extend(field_regions)
            for element_index in internal_region_order(regions, page.height):
                region = regions[element_index]
                text, bbox = region.text, region.bbox
                element_class = region.element_class or internal_element_class(
                    text, bbox, page.height
                )
                if element_class is ListItem and region.element_class is None:
                    text = internal_BULLET.sub("", text, count=1).strip()
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

from __future__ import annotations

import io
import os
import re
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Any

from core_pdf import PdfDocument
from core_pdf.impl.exceptions import PdfEmptySourceError, PdfError, PdfPasswordError
from core_pdf.impl.pdf_values import recover_pdf_name
from core_pdf_compat.pdfminer._extract import extract_document_pages
from core_pdf_compat.pdfminer._layout import LAParams, LTFigure, LTTextBox
from core_pdf_compat.pdfminer._projection import UNSTRUCTURED_POLICY
from core_pdf_compat.pypdf import PdfReader
from core_pdf_compat.pypdf.generic import ArrayObject, IndirectObject

from ._classification import (
    BULLET,
    classify_element,
)
from ._elements import (
    CoordinatesMetadata,
    Element,
    ElementMetadata,
    ListItem,
    PageBreak,
)
from ._languages import LanguageFallback, apply_languages, check_languages_argument
from ._links import page_links
from ._metadata import assign_element_ids, assign_parent_ids
from ._regions import (
    TextRegion,
    clean_text,
    combine_list_regions,
    figure_text_snippets,
    layout_regions,
    region_order,
)

# unstructured 0.25's is_pdf_too_complex constants (Apache-2.0).
GRAPHICS_OPS = re.compile(
    rb"(?:^|(?<=\s))(?:m|l|c|v|y|h|re|S|s|f|F|f\*|B|B\*|b|b\*|n|W|W\*|cm|q|Q|"
    rb"Do|g|G|rg|RG|k|K|cs|CS|w|J|j|M|d|i|gs)(?=\s|$)",
    re.MULTILINE,
)

TEXT_OPS = re.compile(rb"(?:^|(?<=\s))(?:Tj|TJ|'|\"|Tf|Td|TD|Tm|T\*|BT|ET)(?=\s|$)", re.MULTILINE)

MIN_FILE_SIZE_BYTES = 1024 * 1024

MIN_RAW_STREAM_BYTES = 100_000


def pdf_too_complex(filename: object, password: str) -> bool:
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
                font = document.resolver.as_dict(raw_font)
                if (
                    font is not None
                    and recover_pdf_name(font.get("Subtype")) == "Type0"
                    and font.get("DescendantFonts") is None
                ):
                    return True
    return graphics_heavy(filename)


def graphics_heavy(filename: object) -> bool:
    # Ported from unstructured 0.25's is_pdf_too_complex (Apache-2.0), on the pypdf facade
    # as unstructured reads the file with pypdf: a file of 1 MB or more is too complex when
    # a page's content holds over 10,000 graphics operators, 20 for each text operator. Any
    # failure passes the file. unstructured 0.27 inspects every file, and fails closed on
    # oversized pages and documents (CVE-2026-33123).
    try:
        if isinstance(filename, (bytes, str)) and (
            isinstance(filename, bytes) or filename.startswith("%PDF")
        ):
            data = filename if isinstance(filename, bytes) else filename.encode("latin-1")
            if len(data) < MIN_FILE_SIZE_BYTES:
                return False
            reader = PdfReader(io.BytesIO(data))
        else:
            if os.path.getsize(filename) < MIN_FILE_SIZE_BYTES:  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                return False
            reader = PdfReader(filename)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        if not reader.pages:
            return False
        for page in reader.pages:
            # dict.get leaves /Contents unresolved, so an indirect array has no data.
            contents = page.get("/Contents")
            if contents is None:
                continue
            chunks: list[bytes] = []
            try:
                if isinstance(contents, ArrayObject):
                    for item in contents:
                        obj = item.get_object() if isinstance(item, IndirectObject) else item
                        if hasattr(obj, "get_data"):
                            chunks.append(obj.get_data())
                else:
                    obj = (
                        contents.get_object() if isinstance(contents, IndirectObject) else contents
                    )
                    if hasattr(obj, "get_data"):
                        chunks.append(obj.get_data())
            except Exception:  # noqa: BLE001
                continue
            raw_data = b"".join(chunks)
            if len(raw_data) < MIN_RAW_STREAM_BYTES:
                continue
            graphics = len(GRAPHICS_OPS.findall(raw_data))
            if graphics <= 10_000:
                continue
            text = len(TEXT_OPS.findall(raw_data))
            if graphics / max(text, 1) > 20.0:
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


STRATEGIES = frozenset({"auto", "fast", "hi_res", "ocr_only"})


def file_last_modified(filename: str | PathLike[str]) -> str | None:
    # Looked up on the public module, which stands for unstructured.partition.pdf, so
    # patching it there takes effect as it does upstream.
    package: Any = import_module("core_pdf_compat.unstructured")
    return package.get_last_modified_date(filename)


def partition_pdf(
    filename: object = None,
    file: Any = None,
    strategy: object = "auto",
    metadata_filename: str | None = None,
    starting_page_number: int = 1,
    languages: list[str] | None = None,
    detect_language_per_element: bool = False,
    metadata_last_modified: str | None = None,
    unique_element_ids: bool = False,
    language_fallback: LanguageFallback | None = None,
    **kwargs: object,
) -> list[Element]:
    if str(strategy) not in STRATEGIES:
        raise ValueError(f"{strategy} is not a valid strategy.")
    if filename is None and file is None:
        raise ValueError("Exactly one of filename and file must be specified.")
    check_languages_argument(languages)
    display_name = metadata_filename
    file_directory = None
    last_modified = metadata_last_modified
    if filename is None:
        file.seek(0)
        filename = file.read()
    elif isinstance(filename, (str, PathLike)) and not str(filename).startswith("%PDF"):
        display_name = metadata_filename or Path(filename).name
        file_directory = str(Path(filename).parent)
        last_modified = last_modified or file_last_modified(filename)
    if metadata_filename:
        # As unstructured does: a path given here also names the directory.
        named_directory, display_name = os.path.split(metadata_filename)
        file_directory = named_directory or file_directory
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
            links = [] if source_page is None else page_links(source_page, page.height)
            regions = layout_regions(text_boxes, links, page.height)
            for figure in (item for item in page if isinstance(item, LTFigure)):
                for figure_text in figure_text_snippets(figure):
                    if cleaned_figure_text := clean_text(figure_text):
                        regions.append(TextRegion(cleaned_figure_text, figure.bbox, links=[]))
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
                            "page_number": page.pageid + starting_page_number - 1,
                            "bbox": bbox,
                            "coordinates": CoordinatesMetadata(
                                (
                                    (bbox[0], page.height - bbox[3]),
                                    (bbox[0], page.height - bbox[1]),
                                    (bbox[2], page.height - bbox[1]),
                                    (bbox[2], page.height - bbox[3]),
                                ),
                                width=page.width,
                                height=page.height,
                            ),
                            "filename": display_name,
                            "file_directory": file_directory,
                            "filetype": "application/pdf",
                            "last_modified": last_modified,
                            "links": region.links,
                        }
                    )
                    if include_metadata
                    else ElementMetadata()
                )
                result.append(element_class(text, metadata))
            if include_page_breaks:
                result.append(
                    PageBreak(
                        "",
                        ElementMetadata(
                            filename=display_name,
                            file_directory=file_directory,
                            filetype="application/pdf",
                            last_modified=metadata_last_modified,
                        ),
                    )
                )
    finally:
        document.close()
    if include_metadata:
        apply_languages(
            result,
            languages,
            per_element=detect_language_per_element,
            fallback=language_fallback,
        )
    assign_element_ids(result, unique=unique_element_ids)
    assign_parent_ids(result)
    return result

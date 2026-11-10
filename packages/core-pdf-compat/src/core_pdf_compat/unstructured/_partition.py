from __future__ import annotations

import enum
import io
import os
import re
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import IO, Any

from core_pdf import PdfDocument
from core_pdf.impl.exceptions import PdfEmptySourceError, PdfError, PdfPasswordError
from core_pdf.impl.pdf_values import recover_pdf_name
from core_pdf_compat.pdfminer._extract import extract_document_pages
from core_pdf_compat.pdfminer._layout import LAParams, LTFigure, LTTextBox
from core_pdf_compat.pdfminer._projection import UNSTRUCTURED_POLICY

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
from .errors import UnprocessableEntityError

# unstructured 0.27.37's pdf_complexity constants (Apache-2.0).
GRAPHICS_OPS_PATTERN = re.compile(
    rb"(?:^|(?<=\s))(?:m|l|c|v|y|h|re|S|s|f|F|f\*|B|B\*|b|b\*|n|W|W\*|cm|q|Q|"
    rb"Do|g|G|rg|RG|k|K|cs|CS|w|J|j|M|d|i|gs)(?=\s|$)",
    re.MULTILINE,
)

TEXT_OPS_PATTERN = re.compile(
    rb"(?:^|(?<=\s))(?:Tj|TJ|'|\"|Tf|Td|TD|Tm|T\*|BT|ET)(?=\s|$)", re.MULTILINE
)

DEFAULT_MIN_FILE_SIZE_BYTES = 0

DEFAULT_MIN_RAW_STREAM_BYTES = 100_000

# These caps fail closed against crafted content streams (CVE-2026-33123): per page, then
# per document.
DEFAULT_MAX_RAW_STREAM_BYTES = 50 * 1024 * 1024

DEFAULT_MAX_CONTENT_STREAM_ARRAY_ENTRIES = 10_000

DEFAULT_MAX_TOTAL_STREAM_BYTES = 1024 * 1024 * 1024

DEFAULT_MAX_TOTAL_ARRAY_ENTRIES = 1_000_000


class PdfComplexity(enum.Enum):
    """Why PDFMiner text extraction should be avoided for a PDF."""

    VECTOR_HEAVY = "vector_heavy"
    """Graphics operators dominate a page's text operators, as in CAD drawings."""
    EXCEEDS_LIMITS = "exceeds_limits"
    """A page's content streams exceed the decoded-byte or array-entry caps."""


def limit_reached(error: Exception) -> bool:
    # pypdf's LimitReachedError, from the pypdf facade or from pypdf itself: callers may hand
    # in a reader of either, as unstructured's own tests do.
    return any(cls.__name__ == "LimitReachedError" for cls in type(error).__mro__)


def pdf_complexity(
    filename: str = "",
    file: bytes | IO[bytes] | None = None,
    max_graphics_ops: int = 10_000,
    min_graphics_to_text_ratio: float = 20.0,
    min_file_size_bytes: int = DEFAULT_MIN_FILE_SIZE_BYTES,
    min_raw_stream_bytes: int = DEFAULT_MIN_RAW_STREAM_BYTES,
    max_raw_stream_bytes: int = DEFAULT_MAX_RAW_STREAM_BYTES,
    max_content_stream_array_entries: int = DEFAULT_MAX_CONTENT_STREAM_ARRAY_ENTRIES,
    max_total_stream_bytes: int = DEFAULT_MAX_TOTAL_STREAM_BYTES,
    max_total_array_entries: int = DEFAULT_MAX_TOTAL_ARRAY_ENTRIES,
) -> PdfComplexity | None:
    """Return why PDFMiner text extraction should be avoided for a PDF, or None.

    Ported from unstructured 0.27.37's pdf_complexity (Apache-2.0), reading the file with
    the pypdf facade as unstructured reads it with pypdf. A page is vector-heavy when its
    content holds over max_graphics_ops graphics operators and min_graphics_to_text_ratio
    for each text operator; a vector-heavy page does not end the scan, because the
    CVE-2026-33123 caps on the later pages take precedence. A stream pypdf refuses to decode
    fails closed; any other unreadable stream skips itself, and a file that cannot be read
    keeps the verdict reached so far.
    """
    # Looked up on the public module, which stands for unstructured.partition.pdf, so
    # patching PdfReader there takes effect as it does upstream.
    reader_class: Any = import_module("core_pdf_compat.unstructured").PdfReader
    original_pos: int | None = None
    vector_heavy = False
    try:
        if file is not None and not isinstance(file, bytes) and hasattr(file, "tell"):
            original_pos = file.tell()
        if file is not None:
            if isinstance(file, bytes):
                file_size = len(file)
            else:
                file.seek(0, 2)
                file_size = file.tell()
                file.seek(original_pos or 0)
        elif filename:
            file_size = os.path.getsize(filename)
        else:
            return None
        if file_size < min_file_size_bytes:
            return None
        if file is not None:
            if isinstance(file, bytes):
                reader = reader_class(io.BytesIO(file))
            else:
                file.seek(0)
                reader = reader_class(file)
        else:
            reader = reader_class(filename)
        if not reader.pages:
            return None
        total_raw_bytes = 0
        total_array_entries = 0
        for page in reader.pages:
            contents = page.get("/Contents")
            if contents is None:
                continue
            # dict.get leaves /Contents unresolved; an indirect array takes the array branch.
            try:
                if hasattr(contents, "get_object"):
                    contents = contents.get_object()
            except Exception:  # noqa: BLE001
                continue
            raw_data: bytes | bytearray = b""
            # Both pypdfs' ArrayObject subclass list.
            if isinstance(contents, list):
                if len(contents) > max_content_stream_array_entries:
                    return PdfComplexity.EXCEEDS_LIMITS
                # Every slot is charged, streams or not.
                total_array_entries += len(contents)
                if total_array_entries > max_total_array_entries:
                    return PdfComplexity.EXCEEDS_LIMITS
                accumulated = bytearray()
                for item in contents:
                    try:
                        obj = item.get_object() if hasattr(item, "get_object") else item
                        if not hasattr(obj, "get_data"):
                            continue
                        chunk = obj.get_data()
                    except Exception as error:  # noqa: BLE001
                        if limit_reached(error):
                            return PdfComplexity.EXCEEDS_LIMITS
                        continue
                    total_raw_bytes += len(chunk)
                    if total_raw_bytes > max_total_stream_bytes:
                        return PdfComplexity.EXCEEDS_LIMITS
                    if len(accumulated) + len(chunk) > max_raw_stream_bytes:
                        return PdfComplexity.EXCEEDS_LIMITS
                    accumulated.extend(chunk)
                raw_data = accumulated
            elif hasattr(contents, "get_data"):
                try:
                    chunk = contents.get_data()
                except Exception as error:  # noqa: BLE001
                    if limit_reached(error):
                        return PdfComplexity.EXCEEDS_LIMITS
                    continue
                total_raw_bytes += len(chunk)
                if total_raw_bytes > max_total_stream_bytes or len(chunk) > max_raw_stream_bytes:
                    return PdfComplexity.EXCEEDS_LIMITS
                raw_data = chunk
            if vector_heavy or len(raw_data) < min_raw_stream_bytes:
                continue
            graphics = sum(1 for _ in GRAPHICS_OPS_PATTERN.finditer(raw_data))
            if graphics <= max_graphics_ops:
                continue
            text = sum(1 for _ in TEXT_OPS_PATTERN.finditer(raw_data))
            if graphics / max(text, 1) > min_graphics_to_text_ratio:
                vector_heavy = True
    except Exception:  # noqa: BLE001
        return PdfComplexity.VECTOR_HEAVY if vector_heavy else None
    finally:
        if (
            file is not None
            and not isinstance(file, bytes)
            and hasattr(file, "seek")
            and original_pos is not None
        ):
            file.seek(original_pos)
    return PdfComplexity.VECTOR_HEAVY if vector_heavy else None


def pdfminer_fails(filename: object, password: str) -> bool:
    # The files on which unstructured's PDFMiner extraction raises, so it reads no text.
    with PdfDocument.open(filename, password=password) as document:  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        try:
            strict_xref_error = document.strict_xref_validation_error()
        except PdfError as error:
            # An unrecognized header version stops the strict reading at startxref.
            strict_xref_error = str(error)
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
    return False


def complexity_of(source: bytes | str | PathLike[str]) -> PdfComplexity | None:
    # unstructured.partition.pdf.pdf_complexity, looked up on the public module so patching
    # it there takes effect, and called by keyword as unstructured calls it.
    package: Any = import_module("core_pdf_compat.unstructured")
    if isinstance(source, bytes):
        return package.pdf_complexity(filename="", file=source)
    if isinstance(source, str) and source.startswith("%PDF"):
        return package.pdf_complexity(filename="", file=source.encode("latin-1"))
    return package.pdf_complexity(filename=os.fspath(source), file=None)


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
    # unstructured 0.27.37: the fast strategy reads text only through PDFMiner, so a
    # vector-heavy PDF is still extracted and one over the decoding limits is refused rather
    # than returned empty. The other strategies skip PDFMiner on either, and their hi_res
    # and OCR fallbacks need models the facade has none of.
    complexity = complexity_of(filename)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    fast = str(strategy) == "fast"
    if complexity is PdfComplexity.EXCEEDS_LIMITS and fast:
        raise UnprocessableEntityError(
            "PDF content streams exceed the decoding limits for text extraction, and the"
            " fast strategy has no other way to read the document's text."
        )
    if complexity is not None and not fast:
        return []
    try:
        if pdfminer_fails(filename, password):
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

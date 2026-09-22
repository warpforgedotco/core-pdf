from __future__ import annotations

from collections.abc import Iterable, Iterator
from html import escape
from io import BytesIO
from typing import Any, BinaryIO, TextIO, cast

from core_pdf import PdfDocument, PdfPage
from core_pdf.impl.exceptions import PdfError

from .._shared import PdfInput
from ._layout import (
    LAParams,
    LTComponent,
    LTPage,
    LTText,
)
from ._pages import (
    pdfminer_resolvable_pages,
)
from ._projection import (
    project_page,
)


def extract_pages(
    pdf_file: PdfInput,
    password: str = "",
    page_numbers: Iterable[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    laparams: LAParams | None = None,
    _unstructured_mode: bool = False,
) -> Iterator[LTPage]:
    del caching
    params = laparams or LAParams()
    selected = set(page_numbers) if page_numbers is not None else None
    document = PdfDocument.open(
        pdf_file,
        password=password,
        recovery_scan_all_revisions=False,
    )
    try:
        yield from extract_document_pages(
            document, params, selected, maxpages, unstructured_mode=_unstructured_mode
        )
    finally:
        document.close()


def extract_document_pages(
    document: PdfDocument,
    params: LAParams,
    selected: set[int] | None = None,
    maxpages: int = 0,
    *,
    unstructured_mode: bool = False,
) -> Iterator[LTPage]:
    yielded = 0
    page_source: Iterable[tuple[int, PdfPage]]
    if unstructured_mode:
        try:
            page_source = tuple(pdfminer_resolvable_pages(document))
        except PdfError:
            page_source = tuple(enumerate(document.pages))
    else:
        page_source = pdfminer_resolvable_pages(document)
    for page_index, page in page_source:
        if selected is not None and page_index not in selected:
            continue
        if maxpages and yielded >= maxpages:
            break
        yield project_page(page, params, unstructured_mode=unstructured_mode)
        yielded += 1


def extract_text(
    pdf_file: PdfInput,
    password: str = "",
    page_numbers: Iterable[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    codec: str = "utf-8",
    laparams: LAParams | None = None,
) -> str:
    del codec
    pages = [
        "\n".join(item.get_text() for item in page if isinstance(item, LTText))
        for page in extract_pages(pdf_file, password, page_numbers, maxpages, caching, laparams)
    ]
    return "".join(page_text + ("\n" if page_text else "") + "\f" for page_text in pages)


def extract_text_to_fp(
    inf: BinaryIO | PdfInput,
    outfp: TextIO | BinaryIO,
    output_type: str = "text",
    codec: str = "utf-8",
    laparams: LAParams | None = None,
    maxpages: int = 0,
    page_numbers: Iterable[int] | None = None,
    password: str = "",
    **kwargs: Any,
) -> None:
    del kwargs
    if output_type == "text":
        output = extract_text(inf, password, page_numbers, maxpages, True, codec, laparams)
    elif output_type in {"xml", "html"}:
        output = _structured_output(inf, output_type, password, page_numbers, maxpages, laparams)
    else:
        raise ValueError(f"unsupported pdfminer output_type: {output_type}")
    if isinstance(outfp, (BytesIO,)):
        outfp.write(output.encode(codec))
    else:
        cast(TextIO, outfp).write(output)


def _structured_output(
    source: BinaryIO | PdfInput,
    output_type: str,
    password: str,
    page_numbers: Iterable[int] | None,
    maxpages: int,
    laparams: LAParams | None,
) -> str:
    pages = list(extract_pages(source, password, page_numbers, maxpages, True, laparams))
    if output_type == "html":
        html_parts: list[str] = []
        for page in pages:
            html_parts.append(f'<div class="page" data-page="{page.pageid}">')
            for item in page:
                if isinstance(item, LTText) and isinstance(item, LTComponent):
                    bbox = ",".join(str(value) for value in item.bbox)
                    html_parts.append(
                        f'<div class="textbox" data-bbox="{bbox}">{escape(item.get_text())}</div>'
                    )
            html_parts.append("</div>")
        body = "".join(html_parts)
        return f"<!doctype html><html><body>{body}</body></html>"
    xml_parts: list[str] = []
    for page in pages:
        xml_parts.append(f'<page id="{page.pageid}" bbox="{page.bbox}">')
        xml_parts.extend(
            f'<textbox bbox="{item.bbox}">{escape(item.get_text())}</textbox>'
            for item in page
            if isinstance(item, LTText) and isinstance(item, LTComponent)
        )
        xml_parts.append("</page>")
    return "<pages>" + "".join(xml_parts) + "</pages>"

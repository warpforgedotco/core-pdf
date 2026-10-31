from __future__ import annotations

from collections.abc import Iterable, Iterator
from html import escape
from io import TextIOBase
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import numpy

from core_pdf import PdfDocument, PdfPage
from core_pdf.impl.exceptions import PdfError
from core_pdf.impl.types import PdfName

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
    PDFMINER_POLICY,
    ProjectionPolicy,
    project_page,
)


def extract_pages(
    pdf_file: PdfInput,
    password: str = "",
    page_numbers: Iterable[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    laparams: LAParams | None = None,
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
        yield from extract_document_pages(document, params, selected, maxpages)
    finally:
        document.close()


def extract_document_pages(
    document: PdfDocument,
    params: LAParams,
    selected: set[int] | None = None,
    maxpages: int = 0,
    *,
    policy: ProjectionPolicy = PDFMINER_POLICY,
) -> Iterator[LTPage]:
    yielded = 0
    page_source: Iterable[tuple[int, PdfPage]]
    if policy.tolerant_pages:
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
        yield project_page(page, params, policy=policy)
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
    output_dir = kwargs.get("output_dir")
    if output_type == "text":
        try:
            output = extract_text(inf, password, page_numbers, maxpages, True, codec, laparams)
        except PdfError as error:
            # pdfminer yields no pages for a file whose xref chain never reaches a catalog.
            if "No /Root object" not in str(error):
                raise
            output = ""
    elif output_type in {"xml", "html", "hocr"}:
        output = _structured_output(inf, output_type, password, page_numbers, maxpages, laparams)
    else:
        raise ValueError(f"unsupported pdfminer output_type: {output_type}")
    if output_dir:
        write_images(inf, output_dir, password, page_numbers, maxpages)
    target: Any = outfp
    target.write(output if isinstance(outfp, TextIOBase) else output.encode(codec))


def _page_text(page: Any) -> str:
    return "".join(item.get_text() for item in page if isinstance(item, LTText))


def write_images(
    source: BinaryIO | PdfInput,
    output_dir: str,
    password: str,
    page_numbers: Iterable[int] | None,
    maxpages: int,
) -> None:
    stream: Any = source
    if hasattr(stream, "seek"):
        stream.seek(0)
    selected = None if page_numbers is None else set(page_numbers)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with PdfDocument.open(stream, password=password) as document:
        for page_index, page in enumerate(document.pages):
            if selected is not None and page_index not in selected:
                continue
            if maxpages and page_index >= maxpages:
                break
            for image_index, image in enumerate(page.extract_images()):
                name = f"page{page_index + 1}-{image_index}"
                (directory / f"{name}{image_extension(image)}").write_bytes(image_bytes(image))


def image_filters(image: Any) -> list[str]:
    value = (image.dictionary or {}).get(PdfName(b"Filter"))
    values = value if isinstance(value, (list, tuple)) else [value]
    return [item.value for item in values if isinstance(item, PdfName)]


def image_extension(image: Any) -> str:
    filters = image_filters(image)
    for name, extension in (
        ("DCTDecode", ".jpg"),
        ("JPXDecode", ".jp2"),
        ("JBIG2Decode", ".jb2"),
    ):
        if name in filters:
            return extension
    return ".bmp"


def image_bytes(image: Any) -> bytes:
    if image_extension(image) != ".bmp":
        return bytes(image.raw_data or b"")
    raster = image.data
    if raster is None:
        return bytes(image.raw_data or b"")
    samples = numpy.asarray(raster.array)
    if samples.dtype != numpy.uint8:
        scale = 255 / max(float(samples.max()), 1.0)
        samples = numpy.rint(samples * scale).astype(numpy.uint8)
    if samples.ndim == 2:
        samples = samples[:, :, numpy.newaxis]
    height, width, channels = samples.shape
    if channels not in {1, 3}:
        samples = samples[:, :, :3] if channels > 3 else samples[:, :, :1]
        channels = samples.shape[2]
    row_size = width * channels
    padding = -row_size % 4
    # BMP stores rows bottom-up, and color as BGR.
    ordered = samples[::-1, :, ::-1]
    body = b"".join(row.tobytes() + bytes(padding) for row in ordered)
    header_size = 54 + (1024 if channels == 1 else 0)
    palette = b"".join(bytes((v, v, v, 0)) for v in range(256)) if channels == 1 else b""
    header = (
        b"BM"
        + (header_size + len(body)).to_bytes(4, "little")
        + bytes(4)
        + header_size.to_bytes(4, "little")
        + (40).to_bytes(4, "little")
        + width.to_bytes(4, "little")
        + height.to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (channels * 8).to_bytes(2, "little")
        + bytes(24)
    )
    return header + palette + body


def _structured_output(
    source: BinaryIO | PdfInput,
    output_type: str,
    password: str,
    page_numbers: Iterable[int] | None,
    maxpages: int,
    laparams: LAParams | None,
) -> str:
    pages = list(extract_pages(source, password, page_numbers, maxpages, True, laparams))
    if output_type == "hocr":
        hocr_parts = [
            f'<div class="ocr_page" id="page_{page.pageid}">{escape(_page_text(page))}</div>'
            for page in pages
        ]
        return f"<html><body>{''.join(hocr_parts)}</body></html>"
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

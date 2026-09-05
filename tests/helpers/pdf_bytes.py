# SPDX-License-Identifier: AGPL-3.0-only
"""Build small PDFs in memory and read them back through the public document."""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager

from core_pdf import PdfDocument
from core_pdf.impl._impl.output.model import DiagnosticTextRun

HELVETICA = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
LETTER = (0, 0, 612, 792)


def assemble_pdf(objects: Sequence[bytes], *, version: str = "1.4") -> bytes:
    """Serialise ``objects`` as ``1 0 obj`` … ``N 0 obj`` with a classic xref table.

    Object 1 must be the catalog; the trailer points ``/Root`` at it.
    """
    pdf = bytearray(f"%PDF-{version}\n".encode() + b"%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(pdf)


def stream_obj(data: bytes, extra: bytes = b"") -> bytes:
    """A stream object body with ``/Length`` filled in; ``extra`` adds dictionary keys."""
    return f"<< /Length {len(data)} {extra.decode()} >>\nstream\n".encode() + data + b"\nendstream"


def one_page_pdf(
    content: bytes,
    *,
    page_extra: bytes = b"",
    resources: bytes = b"<< /Font << /F1 5 0 R >> >>",
    media_box: tuple[int, int, int, int] = LETTER,
    font: bytes = HELVETICA,
    extra_objects: Iterable[bytes] = (),
) -> bytes:
    """One page whose content stream is ``content``.

    Object numbers are fixed so callers can reference them: 1 catalog, 2 pages,
    3 page, 4 content, 5 font (``/F1``), then ``extra_objects`` from 6 upward.
    """
    x0, y0, x1, y1 = media_box
    return assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [{x0} {y0} {x1} {y1}] ".encode()
                + b"/Resources "
                + resources
                + b" /Contents 4 0 R "
                + page_extra
                + b" >>"
            ),
            stream_obj(content),
            font,
            *extra_objects,
        ]
    )


def text_pages_pdf(texts: Sequence[str], *, version: str = "1.7") -> bytes:
    """One page per entry in ``texts``, each showing that text in Helvetica."""
    kids = " ".join(f"{4 + page_index * 2} 0 R" for page_index in range(len(texts)))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(texts)} >>".encode(),
        HELVETICA,
    ]
    for page_index, text in enumerate(texts):
        content_object = 5 + page_index * 2
        content = f"BT /F1 10 Tf 36 750 Td ({text}) Tj ET".encode()
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
            + f"{content_object} 0 R >>".encode()
        )
        objects.append(stream_obj(content))
    return assemble_pdf(objects, version=version)


@contextmanager
def open_pdf(data: bytes) -> Iterator[PdfDocument]:
    with PdfDocument.open(io.BytesIO(data)) as document:
        yield document


def first_page_runs(data: bytes) -> tuple[DiagnosticTextRun, ...]:
    """The diagnostic text runs of page one, before layout."""
    with open_pdf(data) as document:
        return tuple(document.pages[0].text_diagnostics().runs)


def first_page_text(data: bytes) -> str:
    """The extracted text of page one."""
    with open_pdf(data) as document:
        return document.pages[0].extract().text

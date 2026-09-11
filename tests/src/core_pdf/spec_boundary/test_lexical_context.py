# SPDX-License-Identifier: AGPL-3.0-only
"""Actual page resources and content streams use the same selected name identities."""

import pytest

from core_pdf import PdfDocument


def internal_resource_document(version: str, component_values: bytes) -> bytes:
    content = b"/C#31 cs " + component_values + b" sc 0 0 4 4 re f"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 4 4] "
        b"/Resources << /ColorSpace << /C#31 /DeviceRGB /C1 /DeviceGray >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]
    data = f"%PDF-{version}\n".encode()
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    start = len(data)
    data += b"xref\n0 5\n0000000000 65535 f \n"
    data += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets)
    data += b"trailer\n<< /Size 5 /Root 1 0 R >>\n"
    return data + f"startxref\n{start}\n%%EOF\n".encode()


@pytest.mark.parametrize(
    ("version", "values", "pixel"),
    [("1.1", b"1 0 0", (255, 0, 0, 255)), ("1.2", b"1", (255, 255, 255, 255))],
)
def test_document_resource_name_collisions_follow_the_selected_grammar(
    version: str, values: bytes, pixel: tuple[int, int, int, int]
) -> None:
    # In 1.1 C#31 and C1 are distinct resources. From 1.2 both spell C1, so
    # core's existing duplicate-key recovery keeps the last entry (Gray).
    # The modern case is malformed-input recovery, not a strict valid PDF.
    # Capture must use the same name rules as resource parsing.
    with PdfDocument(internal_resource_document(version, values)) as document:
        image = document.pages[0].render().rasterize(background=(0, 0, 0, 255)).array()
        assert tuple(image[1, 1]) == pixel

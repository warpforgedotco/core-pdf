"""Regenerate authored OCR fixtures with Pillow 12.3.0 and the bundled font."""

import zlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
FONT = ROOT / "src/core_pdf/impl/_impl/fonts/data/raster_fonts/LiberationSans-Regular.ttf"


def write_pdf(image: Image.Image, destination: Path, matrix: str) -> None:
    content = f"q {matrix} cm /Im0 Do Q".encode()
    compressed = zlib.compress(image.tobytes())
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 480 192] "
        b"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>",
        f"<< /Type /XObject /Subtype /Image /Width {image.width} /Height {image.height} "
        f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
        f"/Length {len(compressed)} >>\nstream\n".encode()
        + compressed
        + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    )
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, value in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + value + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    destination.write_bytes(data)


def main() -> None:
    font = ImageFont.truetype(str(FONT), 48)
    invoice = Image.new("L", (1000, 400), 255)
    draw = ImageDraw.Draw(invoice)
    draw.multiline_text(
        (50, 45), "INVOICE 2048\nBlue widgets 12\nTotal 360 dollars", font=font, fill=0, spacing=30
    )
    invoice.save(HERE / "invoice.png")
    write_pdf(invoice, HERE / "invoice.pdf", "480 0 0 192 0 0")
    # An image-only page stored sideways exercises direct-image rotation.
    sideways = invoice.transpose(Image.Transpose.ROTATE_90)
    write_pdf(sideways, HERE / "rotated-image.pdf", "0 -192 480 0 0 192")


if __name__ == "__main__":
    main()

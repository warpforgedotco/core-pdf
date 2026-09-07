"""Image records for the native PyMuPDF text facade."""

from __future__ import annotations

import struct
import zlib
from typing import Any, cast

from core_pdf.api.compat._shared import float32, png_chunk
from core_pdf.api.compat.pymupdf.geometry import Matrix, Point, Rect
from core_pdf.api.document import PdfPage
from core_pdf.impl._impl.graphics.color_spec import normalize_image_color_spec
from core_pdf.impl._impl.graphics.images import ImageRaster, decode_image
from core_pdf.impl.spec.s_08_graphics.image_spec import ImageSource


def internal_png(raster: ImageRaster) -> bytes:
    pixels = raster.array
    height, width = pixels.shape[:2]
    channels = pixels.shape[2] if pixels.ndim == 3 else 1
    color_type = {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    data = pixels.tobytes()
    stride = width * channels
    rows = b"".join(b"\0" + data[i : i + stride] for i in range(0, len(data), stride))
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        + png_chunk(b"pHYs", struct.pack(">IIB", 3780, 3780, 1))
        + png_chunk(b"IDAT", zlib.compress(rows))
        + png_chunk(b"IEND", b"")
    )


def capture_images(
    page: PdfPage, *, clip: Rect, flags: int, matrix: Matrix | None = None
) -> list[tuple[int, dict[str, Any]]]:
    crop = Rect(page.crop_box or page.media_box)
    unit_value = page.document.resolver.resolve(page.page_dict.get("UserUnit"))
    unit = float32(unit_value) if isinstance(unit_value, (int, float)) else 1.0
    result = []
    for image in page.extract_images():
        source = image.image_source
        if not isinstance(source, ImageSource) or image.rect is None:
            continue
        quad = cast(
            tuple[tuple[float, float], ...] | None,
            next(
                (
                    item[1]
                    for item in image.items
                    if isinstance(item, tuple) and len(item) >= 2 and item[0] == "quad"
                ),
                None,
            ),
        )
        if quad is None:
            x0, y0, x1, y1 = image.rect
            quad = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
        points = [
            Point(
                float32((float32(x) - float32(crop.x0)) * unit),
                float32((float32(crop.y1) - float32(y)) * unit),
            )
            for x, y in quad
        ]
        if matrix is not None:
            points = [point * matrix for point in points]
        origin, right, bottom = points[2], points[3], points[0]
        transform = (
            float32(right.x - origin.x),
            float32(right.y - origin.y),
            float32(bottom.x - origin.x),
            float32(bottom.y - origin.y),
            origin.x,
            origin.y,
        )
        bbox = Rect(
            min(p.x for p in points),
            min(p.y for p in points),
            max(p.x for p in points),
            max(p.y for p in points),
        )
        if not (clip.intersects(bbox) if flags & 64 else clip.contains(bbox)):
            continue
        if flags & 64:
            bbox &= clip
        raster = decode_image(ImageSource(source.raw, source.dictionary))
        if raster is None:
            continue
        height, width = raster.array.shape[:2]
        color_spec = normalize_image_color_spec(source.dictionary)
        channels = {
            "DeviceRGB": 3,
            "DeviceCMYK": 4,
            "DeviceGray": 1,
            "CalRGB": 3,
            "Lab": 3,
            "CalGray": 1,
        }.get(color_spec.kind or "", color_spec.channels)
        encoded = internal_png(raster)
        ext = "png"
        if str(source.dictionary.get("Filter")) == "DCTDecode":
            encoded, ext = bytes(source.raw), "jpeg"
        mask = None
        if source.soft_mask is not None:
            smask = source.soft_mask
            mask_raster = decode_image(ImageSource(smask.raw, smask.dictionary))
            if mask_raster is not None:
                mask = internal_png(mask_raster)
        result.append(
            (
                image.seqno,
                {
                    "type": 1,
                    "number": 0,
                    "bbox": tuple(bbox),
                    "width": width,
                    "height": height,
                    "ext": ext,
                    "colorspace": channels,
                    "xres": 96,
                    "yres": 96,
                    "bpc": int(source.dictionary.get("BitsPerComponent", 8)),
                    "transform": transform,
                    "size": len(encoded),
                    "image": encoded,
                    "mask": mask,
                },
            )
        )
    return result


__all__ = ("capture_images",)

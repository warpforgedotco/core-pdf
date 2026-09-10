"""Image records for the native PyMuPDF text facade."""

from __future__ import annotations

import struct
import zlib
from dataclasses import replace
from typing import Any, cast

import numpy

from core_pdf.api.compat._shared import float32, png_chunk
from core_pdf.api.compat.pymupdf.colors import cmyk_bytes_to_rgb, indexed_bytes_to_rgb
from core_pdf.api.compat.pymupdf.geometry import Matrix, Point, Rect
from core_pdf.api.compat.pymupdf.paint_scopes import ImageScope, image_scopes, transformed_box
from core_pdf.api.compat.pymupdf.projection import page_space_matrix
from core_pdf.api.compat.pymupdf.shadings import ShadingRaster, rasterize_shading
from core_pdf.api.document import PdfPage
from core_pdf.impl._impl.graphics.color import internal_normalize_image_samples
from core_pdf.impl._impl.graphics.color_spec import normalize_image_color_spec
from core_pdf.impl._impl.graphics.decode_compat import normalize_stream_decode_spec
from core_pdf.impl._impl.graphics.icc_profiles import IccProfileError, parse_icc_transform
from core_pdf.impl._impl.graphics.images import ImageRaster, decode_image
from core_pdf.impl._impl.graphics.stream_decoding import decode_stream_data
from core_pdf.impl._impl.model.geometry import intersect_bbox
from core_pdf.impl._impl.runtime.scalars import parse_int
from core_pdf.impl.spec.s_07_filters.decode_spec import StreamDecodeSpec
from core_pdf.impl.spec.s_07_filters.errors import FilterError
from core_pdf.impl.spec.s_08_graphics.image_spec import ImageSource
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf.impl.types import ImageRecord, PdfName


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


def internal_unmatte(raster: ImageRaster, mask: ImageRaster, matte: object) -> ImageRaster:
    """Undo preblending while retaining the separately exported soft mask."""
    if not isinstance(matte, (list, tuple)) or len(matte) != raster.channels:
        return raster
    if raster.array.shape[:2] != mask.array.shape[:2]:
        return raster
    try:
        background = numpy.asarray(matte, dtype=numpy.float64) * 255
    except (TypeError, ValueError):
        return raster
    if not numpy.isfinite(background).all():
        return raster
    background = numpy.clip(background, 0, 255).astype(numpy.uint8).astype(numpy.float64)
    # MuPDF's image matte table repeats its first component when expanding
    # the interleaved color and alpha samples used by this export path.
    background = numpy.concatenate((background[:1], background[:-1]))
    alpha = mask.array[:, :, :1].astype(numpy.float64)
    samples = raster.array.astype(numpy.float64)
    unblended = background + numpy.trunc(
        numpy.divide(
            (samples - background) * 255,
            alpha,
            out=numpy.zeros_like(samples),
            where=alpha != 0,
        )
    )
    pixels = numpy.empty((*raster.array.shape[:2], raster.channels + 1), dtype=numpy.uint8)
    pixels[:, :, :-1] = numpy.clip(unblended, 0, 255).astype(numpy.uint8)
    pixels[:, :, -1] = 255
    return ImageRaster(pixels, raster.color_model, has_alpha=True)


def internal_image_geometry(
    image: ImageRecord, crop: Rect, unit: float, clip: Rect, flags: int, matrix: Matrix | None
) -> tuple[Rect, tuple[float, ...]] | None:
    if image.rect is None:
        return None
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
    # Preserve vector precision before translating large page coordinates.
    a, b = (float32(quad[1][i] - quad[0][i]) for i in (0, 1))
    c, d = (float32(quad[2][i] - quad[0][i]) for i in (0, 1))
    image_matrix = Matrix(
        float32(a * unit),
        float32(-b * unit),
        float32(-c * unit),
        float32(d * unit),
        float32(float32(float32(quad[0][0]) - float32(crop.x0) + c) * unit),
        float32(float32(float32(float32(crop.y1) - float32(quad[0][1])) - d) * unit),
    )
    if image.matrix_trace:
        image_matrix = Matrix(1, 0, 0, -1, 0, 1) * page_space_matrix(crop, unit, image.matrix_trace)
    if matrix is not None:
        image_matrix *= matrix
    transform = tuple(image_matrix)
    bbox = Rect(0, 0, 1, 1) * image_matrix
    image_clip = (
        image.control_point_clip if image.control_point_clip is not None else image.image_clip
    )
    if flags & 64 and image_clip is not None:
        x0, y0, x1, y1 = image_clip
        content_clip = Rect(
            float32((float32(x0) - float32(crop.x0)) * unit),
            float32((float32(crop.y1) - float32(y1)) * unit),
            float32((float32(x1) - float32(crop.x0)) * unit),
            float32((float32(crop.y1) - float32(y0)) * unit),
        )
        if matrix is not None:
            content_clip *= matrix
        clip = Rect(
            max(clip.x0, content_clip.x0),
            max(clip.y0, content_clip.y0),
            min(clip.x1, content_clip.x1),
            min(clip.y1, content_clip.y1),
        )
    if flags & 64:
        # Structured text retains image records even when their clipped bounds
        # are empty or inverted. Rect.intersect intentionally treats empty
        # rectangles differently, so clamp each bound directly here.
        bbox = Rect(
            max(bbox.x0, clip.x0),
            max(bbox.y0, clip.y0),
            min(bbox.x1, clip.x1),
            min(bbox.y1, clip.y1),
        )
    elif not clip.contains(bbox):
        return None
    return bbox, transform


def internal_image_payload(source: ImageSource) -> dict[str, Any] | None:
    color_spec = normalize_image_color_spec(source.dictionary)
    image_dictionary = source.dictionary
    if color_spec.kind == "ICCBased" and color_spec.channels in (1, 3):
        preserve_samples = color_spec.channels == 3
        if color_spec.channels == 1:
            try:
                icc_transform = parse_icc_transform(color_spec.icc_profile or b"")
            except IccProfileError:
                preserve_samples = True
            else:
                preserve_samples = icc_transform.input_channels != 1
        if preserve_samples:
            # Exports keep RGB samples and the DeviceGray fallback of an
            # invalid gray profile. Valid ICC gray profiles still convert
            # through their profile and export an RGB PNG.
            image_dictionary = {
                **source.dictionary,
                "ColorSpace": PdfName.of("DeviceGray" if color_spec.channels == 1 else "DeviceRGB"),
            }
    raster = None
    if color_spec.kind in {"DeviceCMYK", "Indexed"}:
        try:
            decoded = decode_stream_data(source.raw, source.dictionary)
        except FilterError:
            return None
        samples = internal_normalize_image_samples(decoded, color_spec, source.dictionary)
        if samples is not None:
            width = parse_int(source.dictionary.get("Width"), 0)
            height = parse_int(source.dictionary.get("Height"), 0)
            if width <= 0 or height <= 0:
                return None
            values = numpy.frombuffer(samples, dtype=numpy.uint8)
            if color_spec.kind == "Indexed" and values.size == width * height:
                rgb = indexed_bytes_to_rgb(values.reshape(height, width), color_spec)
                if rgb is not None:
                    raster = ImageRaster(rgb, "rgb")
            elif color_spec.kind == "DeviceCMYK" and values.size == width * height * 4:
                inks = values.reshape(height, width, 4)
                raster = ImageRaster(cmyk_bytes_to_rgb(inks), "rgb")
    if raster is None:
        raster = decode_image(ImageSource(source.raw, image_dictionary))
    if raster is None:
        return None
    export_color = normalize_image_color_spec(image_dictionary)
    is_stencil = source.dictionary.get("ImageMask") is True
    if is_stencil:
        raster = ImageRaster(raster.array[:, :, -1:], "gray")
    elif (
        export_color.kind == "DeviceGray"
        or (
            export_color.kind == "Indexed"
            and export_color.base_spec is not None
            and export_color.base_spec.kind == "DeviceGray"
        )
    ) and raster.channels == 3:
        raster = ImageRaster(raster.array[:, :, :1], "gray")
    height, width = raster.array.shape[:2]
    channels = {
        "DeviceRGB": 3,
        "DeviceCMYK": 4,
        "DeviceGray": 1,
        "CalRGB": 3,
        "Lab": 3,
        "CalGray": 1,
    }.get(color_spec.kind or "", color_spec.channels)
    if is_stencil:
        channels = 0
    mask = None
    if source.soft_mask is not None:
        smask = source.soft_mask
        mask_raster = decode_image(ImageSource(smask.raw, smask.dictionary))
        if mask_raster is not None:
            if mask_raster.channels == 3:
                mask_raster = ImageRaster(mask_raster.array[:, :, :1], "gray")
            mask = internal_png(mask_raster)
            raster = internal_unmatte(raster, mask_raster, smask.dictionary.get("Matte"))
    decode_spec = normalize_stream_decode_spec(source.dictionary)
    if decode_spec.filters and decode_spec.filters[-1] in ("DCTDecode", "JPXDecode"):
        # Preserve compressed image formats after removing transport filters.
        prefix = StreamDecodeSpec(decode_spec.filters[:-1], decode_spec.params[:-1])
        encoded = decode_stream_data(source.raw, prefix, parent_dictionary=source.dictionary)
        ext = "jpeg" if decode_spec.filters[-1] == "DCTDecode" else "jpx"
    else:
        encoded, ext = internal_png(raster), "png"
    return {
        "type": 1,
        "number": 0,
        "width": width,
        "height": height,
        "ext": ext,
        "colorspace": channels,
        "xres": 72 if ext == "jpx" else 96,
        "yres": 72 if ext == "jpx" else 96,
        "bpc": int(source.dictionary.get("BitsPerComponent", 8)),
        "size": len(encoded),
        "image": encoded,
        "mask": mask,
    }


def capture_images(
    page: PdfPage, *, clip: Rect, flags: int, matrix: Matrix | None = None
) -> list[tuple[int, dict[str, Any]]]:
    crop = Rect(page.crop_box or page.media_box)
    unit_value = page.document.resolver.resolve(page.page_dict.get("UserUnit"))
    unit = float32(unit_value) if isinstance(unit_value, (int, float)) else 1.0
    paints: list[tuple[tuple[int, ...], dict[str, Any]]] = []
    page_matrix = Matrix(unit, 0, 0, -unit, float32(-crop.x0 * unit), float32(crop.y1 * unit))
    if matrix is not None:
        page_matrix *= matrix
    payloads: dict[int, dict[str, Any] | None] = {}
    for scope in image_scopes(page, ImageScope(page.get_page_program()), page_matrix):
        order = scope.order
        for image in page.internal_image_records(scope.program):
            source = image.image_source
            if not isinstance(source, ImageSource):
                continue
            geometry = internal_image_geometry(
                internal_place_image(image, scope), crop, unit, clip, flags, matrix
            )
            if geometry is None:
                continue
            if id(source) not in payloads:
                payloads[id(source)] = internal_image_payload(source)
            payload = payloads[id(source)]
            if payload is not None:
                payload = payload.copy()
                bbox, transform = geometry
                payload.update(bbox=tuple(bbox), transform=transform)
                paints.append(((*order, image.seqno, 1), payload))
        transform = Matrix(scope.transform)
        local_matrix = transform * page_matrix
        for drawing in scope.program.drawings:
            if drawing.kind != "shading":
                continue
            shading = rasterize_shading(
                replace(
                    drawing,
                    shading_clip=intersect_bbox(
                        drawing.shading_clip, transformed_box(scope.clip, ~transform)
                    ),
                    control_point_clip=intersect_bbox(
                        drawing.control_point_clip
                        if drawing.control_point_clip is not None
                        else drawing.shading_clip,
                        transformed_box(scope.clip, ~transform),
                    ),
                ),
                page_matrix=local_matrix,
                page_bounds=crop * page_matrix,
                clip=clip,
                flags=flags,
            )
            if shading is not None:
                paints.append(((*order, drawing.seqno, 1), internal_shading_payload(shading)))
    paints.sort(key=lambda item: item[0])
    return [(order[0], payload) for order, payload in paints]


def internal_place_image(image: ImageRecord, scope: ImageScope) -> ImageRecord:
    transform = Matrix(scope.transform)
    if scope.transform == IDENTITY_MATRIX and scope.clip is None:
        return image
    trace = image.matrix_trace
    if trace and scope.initial_matrix is not None and trace[0] == scope.initial_matrix[0]:
        trace = (scope.initial_matrix[1], *trace[1:])
    elif scope.transform != IDENTITY_MATRIX:
        trace = (scope.transform, *trace) if trace else ()
    items = tuple(
        (
            "quad",
            tuple(
                tuple(Point(point) * transform)
                for point in cast(tuple[tuple[float, float], ...], item[1])
            ),
        )
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "quad"
        else item
        for item in image.items
    )
    return replace(
        image,
        rect=transformed_box(image.rect, transform),
        items=items,
        matrix_trace=trace,
        image_clip=intersect_bbox(transformed_box(image.image_clip, transform), scope.clip),
        control_point_clip=intersect_bbox(
            transformed_box(
                image.control_point_clip
                if image.control_point_clip is not None
                else image.image_clip,
                transform,
            ),
            scope.clip,
        ),
    )


def internal_shading_payload(shading: ShadingRaster) -> dict[str, Any]:
    encoded = internal_png(shading.raster)
    return {
        "type": 1,
        "number": 0,
        "bbox": tuple(shading.bbox),
        "width": shading.raster.width,
        "height": shading.raster.height,
        "ext": "png",
        "colorspace": 3,
        "xres": 96,
        "yres": 96,
        "bpc": 8,
        "transform": tuple(shading.transform),
        "size": len(encoded),
        "image": encoded,
        "mask": None,
    }


__all__ = ("capture_images",)

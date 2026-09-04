# SPDX-License-Identifier: AGPL-3.0-only
"""Renderer value records and raster storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy

from core_pdf.impl.runtime.array_views import uint8_image_view
from core_pdf.impl.spec.s_08_graphics.image_decode import ImageSource


@dataclass(slots=True)
class RenderOptions:
    page_number: int | None = None
    rotate: int = 0
    crop: tuple[float, float, float, float] | None = None
    include_annotations: bool = True
    include_layers: bool = True
    include_text: bool = True

    def __post_init__(self) -> None:
        if self.rotate % 90:
            raise ValueError("render rotation must be a multiple of 90 degrees")
        self.rotate %= 360


@dataclass(slots=True)
class DisplayListItem:
    kind: str
    seqno: int
    data: dict[str, Any] = field(default_factory=dict)


class PathPaintKind(IntEnum):
    FILL = 0
    STROKE = 1
    FILL_STROKE = 2


class LineCap(IntEnum):
    BUTT = 0
    ROUND = 1
    PROJECTING_SQUARE = 2


class LineJoin(IntEnum):
    MITER = 0
    ROUND = 1
    BEVEL = 2


PATH_PAINT_NAMES = ("fill", "stroke", "fillstroke")


@dataclass(slots=True)
class PathPaintItem:
    """Typed, allocation-light record for the common unpatterned path hot path."""

    paint_kind: PathPaintKind
    seqno: int
    bbox: Any
    path: Any
    fill: Any
    fill_opacity: Any
    stroke_color: Any
    stroke_opacity: Any
    line_width: Any
    line_cap: Any
    line_join: Any
    dash_pattern: Any
    fill_rule: Any
    blend_mode: Any
    soft_mask_alpha: Any
    coalesced_path: bool = False

    @property
    def kind(self) -> str:
        return PATH_PAINT_NAMES[int(self.paint_kind)]

    def to_data(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox,
            "path": self.path,
            "fill": self.fill,
            "fill_opacity": self.fill_opacity,
            "stroke_color": self.stroke_color,
            "stroke_opacity": self.stroke_opacity,
            "line_width": self.line_width,
            "line_cap": self.line_cap,
            "line_join": self.line_join,
            "dash_pattern": self.dash_pattern,
            "fill_rule": self.fill_rule,
            "blend_mode": self.blend_mode,
            "soft_mask_alpha": self.soft_mask_alpha,
        }


@dataclass(slots=True)
class ImagePaintItem:
    """Typed image paint command whose source owns all PDF image preparation."""

    paint_kind: str
    seqno: int
    bbox: Any
    source: ImageSource | None
    quad: tuple[tuple[float, float], ...] | None
    fill: Any
    fill_opacity: Any
    blend_mode: Any
    soft_mask_alpha: Any
    image_clip: Any
    source_metadata: dict[str, Any]
    ctm: Any = None
    xobject_depth: Any = None

    @property
    def kind(self) -> str:
        return self.paint_kind

    def to_data(self) -> dict[str, Any]:
        """Return the legacy diagnostic mapping without duplicating paint ownership."""
        source = self.source
        return {
            "bbox": self.bbox,
            "raw_data": source.raw if source is not None else None,
            "dictionary": source.dictionary if source is not None else None,
            "soft_mask": source.soft_mask if source is not None else None,
            "image_source": source,
            "items": [("quad", self.quad)] if self.quad is not None else [],
            "fill": self.fill,
            "fill_opacity": self.fill_opacity,
            "blend_mode": self.blend_mode,
            "soft_mask_alpha": self.soft_mask_alpha,
            "image_clip": self.image_clip,
            "source_metadata": self.source_metadata,
            "ctm": self.ctm,
            "xobject_depth": self.xobject_depth,
        }


DisplayItem = DisplayListItem | ImagePaintItem | PathPaintItem


@dataclass(frozen=True, slots=True)
class RasterImage:
    """A contiguous interleaved pixel buffer with its physical layout.

    ``pixels`` is normalized to a read-only byte view. The owner may be a
    ``bytearray`` or NumPy array, so constructing a raster does not copy the
    backing storage.
    """

    pixels: bytes | bytearray | memoryview | numpy.ndarray[Any, Any]
    width: int
    height: int
    channels: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.channels <= 0:
            raise ValueError("raster dimensions and channel count must be positive")
        try:
            pixels = memoryview(self.pixels)
            if not pixels.c_contiguous or pixels.itemsize != 1:
                raise ValueError
            pixels = pixels.cast("B").toreadonly()
        except (TypeError, ValueError) as exc:
            raise ValueError("raster pixels must be a contiguous byte buffer") from exc
        if pixels.nbytes != self.width * self.height * self.channels:
            raise ValueError("raster byte length does not match its dimensions")
        object.__setattr__(self, "pixels", pixels)

    def tesseract_bytes(self) -> bytes:
        """Return the immutable bytes object required by tesserocr."""
        return bytes(self.pixels)

    @property
    def stride(self) -> int:
        return self.width * self.channels

    @property
    def nbytes(self) -> int:
        return memoryview(self.pixels).nbytes

    def array(self) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
        """Return a read-only zero-copy ``(height, width, channels)`` view."""
        return uint8_image_view(
            self.pixels,
            (self.height, self.width, self.channels),
        )


__all__ = (
    "DisplayItem",
    "DisplayListItem",
    "ImagePaintItem",
    "LineCap",
    "LineJoin",
    "PathPaintItem",
    "PathPaintKind",
    "RasterImage",
    "RenderOptions",
)

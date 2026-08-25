# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable raster image storage."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy

from core_pdf.impl.engine.array_views import uint8_image_view


@dataclass(frozen=True, slots=True)
class RasterImage:
    """A contiguous interleaved pixel buffer with its physical layout.

    ``pixels`` is normalized to a read-only byte view.  The owner may be a
    ``bytearray`` or NumPy array, so constructing a raster does not copy the
    backing storage.
    """

    pixels: bytes | bytearray | memoryview | numpy.ndarray[Any, Any]
    width: int
    height: int
    channels: int
    internal_tesseract_pixels: bytes | None = field(
        default=None, init=False, repr=False, compare=False
    )
    internal_tesseract_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

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
        """Return the cached immutable bytes object required by tesserocr."""
        cached = self.internal_tesseract_pixels
        if cached is None:
            with self.internal_tesseract_lock:
                cached = self.internal_tesseract_pixels
                if cached is None:
                    cached = self.pixels if isinstance(self.pixels, bytes) else bytes(self.pixels)
                    object.__setattr__(self, "internal_tesseract_pixels", cached)
        return cached

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


__all__ = ("RasterImage",)

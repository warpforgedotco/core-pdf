# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy

from core_pdf.impl.array_views import UInt8Array, uint8_image_view
from core_pdf.impl.capture_records import CapturedPath
from core_pdf.impl.render_clipping import ClipState
from core_pdf.impl.render_model import DisplayItem, RasterGroup
from core_pdf.impl.render_resources import RenderResources
from core_pdf_spec.standards import SemanticContext

type PixelBox = tuple[int, int, int, int]
EMPTY_PIXEL_BOX: PixelBox = (0, 0, 0, 0)
type FloatPlane = numpy.ndarray[Any, numpy.dtype[numpy.float32]]


class ElementaryScratch:
    __slots__ = ("buffer", "dirty", "source_alpha", "source_shape", "synced_parent", "view")

    def __init__(self, size: int, height: int, width: int) -> None:
        self.buffer = bytearray(size)
        self.view = uint8_image_view(self.buffer, (height, width, 4))
        self.source_alpha = numpy.zeros((height, width), dtype=numpy.float32)
        self.source_shape = numpy.zeros((height, width), dtype=numpy.float32)
        self.synced_parent: RasterGroup | None = None
        self.dirty: list[int] | None = None


class RasterState:
    __slots__ = ()

    pixels: bytearray
    pixel_array: UInt8Array
    semantic_context: SemanticContext
    buffer_stack: list[RasterGroup]
    group_source_alpha: FloatPlane | None
    group_source_shape: FloatPlane | None
    paint_window: list[int] | None
    paint_alpha_is_shape: bool
    shape_alpha: float
    clip: ClipState
    width: int
    height: int
    scale: float
    crop_x0: float
    crop_y1: float
    page_pixels: UInt8Array
    page_buffer: bytearray
    crop_y0: float
    clip_stack: list[int]
    clip_floor: int
    group_floor: int
    scope_stack: list[tuple[int, list[int], int, int, int]]
    resources: RenderResources
    elementary_scratch: dict[int, ElementaryScratch]
    group_member_boxes: dict[int, tuple[float, float, float, float] | None] | None
    stroke_scratch: bytearray | None

    def paint_items(
        self,
        items: Iterable[DisplayItem],
        *,
        translation: tuple[float, float] | None = None,
        parent_blend_mode: str | None = None,
        clip_path: CapturedPath | None = None,
    ) -> None:
        raise NotImplementedError

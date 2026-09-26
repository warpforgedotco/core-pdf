# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import ClassVar

from core_pdf.impl.render.model import RasterImage
from core_pdf.impl.types import Record

frozen_setattr = object.__setattr__


class Raster(Record):
    __slots__ = ("image", "resolution")

    image: RasterImage
    resolution: int

    __fields__: ClassVar[tuple[str, ...]] = ("image", "resolution")
    __match_args__ = ("image", "resolution")

    def __init__(self, image: RasterImage, resolution: int) -> None:
        frozen_setattr(self, "image", image)
        frozen_setattr(self, "resolution", resolution)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.image == other.image and self.resolution == other.resolution

    def __hash__(self) -> int:
        return hash((self.image, self.resolution))

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


class RasterRegion(Record):
    __slots__ = ("raster", "page_box")

    raster: Raster
    page_box: tuple[float, float, float, float]

    __fields__: ClassVar[tuple[str, ...]] = ("raster", "page_box")
    __match_args__ = ("raster", "page_box")

    def __init__(
        self,
        raster: Raster,
        page_box: tuple[float, float, float, float],
    ) -> None:
        frozen_setattr(self, "raster", raster)
        frozen_setattr(self, "page_box", page_box)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.raster == other.raster and self.page_box == other.page_box

    def __hash__(self) -> int:
        return hash((self.raster, self.page_box))


class StrokedTextCell(Record):
    __slots__ = ("source_box", "packed_box", "drawing_indexes")

    source_box: tuple[float, float, float, float]
    packed_box: tuple[float, float, float, float]
    drawing_indexes: tuple[int, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("source_box", "packed_box", "drawing_indexes")
    __match_args__ = ("source_box", "packed_box", "drawing_indexes")

    def __init__(
        self,
        source_box: tuple[float, float, float, float],
        packed_box: tuple[float, float, float, float],
        drawing_indexes: tuple[int, ...],
    ) -> None:
        frozen_setattr(self, "source_box", source_box)
        frozen_setattr(self, "packed_box", packed_box)
        frozen_setattr(self, "drawing_indexes", drawing_indexes)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.source_box == other.source_box
            and self.packed_box == other.packed_box
            and self.drawing_indexes == other.drawing_indexes
        )

    def __hash__(self) -> int:
        return hash((self.source_box, self.packed_box, self.drawing_indexes))


class PackedStrokedTextRaster(Record):
    __slots__ = ("raster", "packed_box", "cells")

    raster: Raster
    packed_box: tuple[float, float, float, float]
    cells: tuple[StrokedTextCell, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("raster", "packed_box", "cells")
    __match_args__ = ("raster", "packed_box", "cells")

    def __init__(
        self,
        raster: Raster,
        packed_box: tuple[float, float, float, float],
        cells: tuple[StrokedTextCell, ...],
    ) -> None:
        frozen_setattr(self, "raster", raster)
        frozen_setattr(self, "packed_box", packed_box)
        frozen_setattr(self, "cells", cells)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.raster == other.raster
            and self.packed_box == other.packed_box
            and self.cells == other.cells
        )

    def __hash__(self) -> int:
        return hash((self.raster, self.packed_box, self.cells))


class OcrRegion(Record):
    __slots__ = ("page_box", "score", "reasons")

    page_box: tuple[float, float, float, float]
    score: float
    reasons: tuple[str, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("page_box", "score", "reasons")
    __match_args__ = ("page_box", "score", "reasons")

    def __init__(
        self,
        page_box: tuple[float, float, float, float],
        score: float,
        reasons: tuple[str, ...],
    ) -> None:
        frozen_setattr(self, "page_box", page_box)
        frozen_setattr(self, "score", score)
        frozen_setattr(self, "reasons", reasons)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.page_box == other.page_box
            and self.score == other.score
            and self.reasons == other.reasons
        )

    def __hash__(self) -> int:
        return hash((self.page_box, self.score, self.reasons))

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.page_box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


class OcrTask(Record):
    __slots__ = (
        "mode",
        "image",
        "rectangle",
        "page_box",
        "resolution",
        "minimum_confidence",
        "character_confidence_threshold",
        "recognize_words",
        "collect_symbols",
    )

    mode: int
    image: RasterImage
    rectangle: tuple[int, int, int, int]
    page_box: tuple[float, float, float, float]
    resolution: int
    minimum_confidence: float
    character_confidence_threshold: float | None
    recognize_words: bool
    collect_symbols: bool

    __fields__: ClassVar[tuple[str, ...]] = (
        "mode",
        "image",
        "rectangle",
        "page_box",
        "resolution",
        "minimum_confidence",
        "character_confidence_threshold",
        "recognize_words",
        "collect_symbols",
    )
    __match_args__ = (
        "mode",
        "image",
        "rectangle",
        "page_box",
        "resolution",
        "minimum_confidence",
        "character_confidence_threshold",
        "recognize_words",
        "collect_symbols",
    )

    def __init__(
        self,
        mode: int,
        image: RasterImage,
        rectangle: tuple[int, int, int, int],
        page_box: tuple[float, float, float, float],
        resolution: int,
        minimum_confidence: float = 20.0,
        character_confidence_threshold: float | None = None,
        recognize_words: bool = False,
        collect_symbols: bool = False,
    ) -> None:
        frozen_setattr(self, "mode", mode)
        frozen_setattr(self, "image", image)
        frozen_setattr(self, "rectangle", rectangle)
        frozen_setattr(self, "page_box", page_box)
        frozen_setattr(self, "resolution", resolution)
        frozen_setattr(self, "minimum_confidence", minimum_confidence)
        frozen_setattr(self, "character_confidence_threshold", character_confidence_threshold)
        frozen_setattr(self, "recognize_words", recognize_words)
        frozen_setattr(self, "collect_symbols", collect_symbols)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.mode == other.mode
            and self.image == other.image
            and self.rectangle == other.rectangle
            and self.page_box == other.page_box
            and self.resolution == other.resolution
            and self.minimum_confidence == other.minimum_confidence
            and self.character_confidence_threshold == other.character_confidence_threshold
            and self.recognize_words == other.recognize_words
            and self.collect_symbols == other.collect_symbols
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.mode,
                self.image,
                self.rectangle,
                self.page_box,
                self.resolution,
                self.minimum_confidence,
                self.character_confidence_threshold,
                self.recognize_words,
                self.collect_symbols,
            )
        )


def pixel_box_to_page_box(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    page_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    page_x0, page_y0, page_x1, page_y1 = page_box
    page_width = page_x1 - page_x0
    page_height = page_y1 - page_y0
    return (
        page_x0 + x0 * page_width / image_width,
        page_y1 - y1 * page_height / image_height,
        page_x0 + x1 * page_width / image_width,
        page_y1 - y0 * page_height / image_height,
    )


def map_ocr_box(
    task: OcrTask,
    bbox: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    return pixel_box_to_page_box(bbox, task.image.width, task.image.height, task.page_box)


def ocr_region_box(
    box: tuple[float, float, float, float],
    *,
    page_width: float,
    page_height: float,
    padding: float,
) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = box
    clipped = (
        max(0.0, x0 - padding),
        max(0.0, y0 - padding),
        min(page_width, x1 + padding),
        min(page_height, y1 + padding),
    )
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def raster_rectangle_page_box(
    raster: Raster,
    page_box: tuple[float, float, float, float],
    rectangle: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    x, y, width, height = rectangle
    return pixel_box_to_page_box(
        (x, y, x + width, y + height), raster.width, raster.height, page_box
    )

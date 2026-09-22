# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, ClassVar, NoReturn, Self

from core_pdf.impl.render.model import RasterImage

internal_frozen_setattr = object.__setattr__


class internal_Raster:
    __slots__ = ("image", "resolution")

    image: RasterImage
    resolution: int

    __fields__: ClassVar[tuple[str, ...]] = ("image", "resolution")
    __match_args__ = ("image", "resolution")

    def __init__(self, image: RasterImage, resolution: int) -> None:
        internal_frozen_setattr(self, "image", image)
        internal_frozen_setattr(self, "resolution", resolution)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(image={self.image!r}, resolution={self.resolution!r})"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.image == other.image and self.resolution == other.resolution

    def __hash__(self) -> int:
        return hash((self.image, self.resolution))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        image = changes.pop("image", self.image)
        resolution = changes.pop("resolution", self.resolution)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(image, resolution)

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


class internal_RasterRegion:
    __slots__ = ("raster", "page_box")

    raster: internal_Raster
    page_box: tuple[float, float, float, float]

    __fields__: ClassVar[tuple[str, ...]] = ("raster", "page_box")
    __match_args__ = ("raster", "page_box")

    def __init__(
        self,
        raster: internal_Raster,
        page_box: tuple[float, float, float, float],
    ) -> None:
        internal_frozen_setattr(self, "raster", raster)
        internal_frozen_setattr(self, "page_box", page_box)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(raster={self.raster!r}, page_box={self.page_box!r})"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.raster == other.raster and self.page_box == other.page_box

    def __hash__(self) -> int:
        return hash((self.raster, self.page_box))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        raster = changes.pop("raster", self.raster)
        page_box = changes.pop("page_box", self.page_box)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(raster, page_box)


class internal_StrokedTextCell:
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
        internal_frozen_setattr(self, "source_box", source_box)
        internal_frozen_setattr(self, "packed_box", packed_box)
        internal_frozen_setattr(self, "drawing_indexes", drawing_indexes)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"source_box={self.source_box!r}, "
            f"packed_box={self.packed_box!r}, "
            f"drawing_indexes={self.drawing_indexes!r}"
            ")"
        )

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

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        source_box = changes.pop("source_box", self.source_box)
        packed_box = changes.pop("packed_box", self.packed_box)
        drawing_indexes = changes.pop("drawing_indexes", self.drawing_indexes)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(source_box, packed_box, drawing_indexes)


class internal_PackedStrokedTextRaster:
    __slots__ = ("raster", "packed_box", "cells")

    raster: internal_Raster
    packed_box: tuple[float, float, float, float]
    cells: tuple[internal_StrokedTextCell, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("raster", "packed_box", "cells")
    __match_args__ = ("raster", "packed_box", "cells")

    def __init__(
        self,
        raster: internal_Raster,
        packed_box: tuple[float, float, float, float],
        cells: tuple[internal_StrokedTextCell, ...],
    ) -> None:
        internal_frozen_setattr(self, "raster", raster)
        internal_frozen_setattr(self, "packed_box", packed_box)
        internal_frozen_setattr(self, "cells", cells)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"raster={self.raster!r}, "
            f"packed_box={self.packed_box!r}, "
            f"cells={self.cells!r}"
            ")"
        )

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

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        raster = changes.pop("raster", self.raster)
        packed_box = changes.pop("packed_box", self.packed_box)
        cells = changes.pop("cells", self.cells)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(raster, packed_box, cells)


class internal_OcrRegion:
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
        internal_frozen_setattr(self, "page_box", page_box)
        internal_frozen_setattr(self, "score", score)
        internal_frozen_setattr(self, "reasons", reasons)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"page_box={self.page_box!r}, "
            f"score={self.score!r}, "
            f"reasons={self.reasons!r}"
            ")"
        )

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

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        page_box = changes.pop("page_box", self.page_box)
        score = changes.pop("score", self.score)
        reasons = changes.pop("reasons", self.reasons)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(page_box, score, reasons)

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.page_box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


class internal_OcrTask:
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
        internal_frozen_setattr(self, "mode", mode)
        internal_frozen_setattr(self, "image", image)
        internal_frozen_setattr(self, "rectangle", rectangle)
        internal_frozen_setattr(self, "page_box", page_box)
        internal_frozen_setattr(self, "resolution", resolution)
        internal_frozen_setattr(self, "minimum_confidence", minimum_confidence)
        internal_frozen_setattr(
            self, "character_confidence_threshold", character_confidence_threshold
        )
        internal_frozen_setattr(self, "recognize_words", recognize_words)
        internal_frozen_setattr(self, "collect_symbols", collect_symbols)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"mode={self.mode!r}, "
            f"image={self.image!r}, "
            f"rectangle={self.rectangle!r}, "
            f"page_box={self.page_box!r}, "
            f"resolution={self.resolution!r}, "
            f"minimum_confidence={self.minimum_confidence!r}, "
            f"character_confidence_threshold={self.character_confidence_threshold!r}, "
            f"recognize_words={self.recognize_words!r}, "
            f"collect_symbols={self.collect_symbols!r}"
            ")"
        )

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

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        mode = changes.pop("mode", self.mode)
        image = changes.pop("image", self.image)
        rectangle = changes.pop("rectangle", self.rectangle)
        page_box = changes.pop("page_box", self.page_box)
        resolution = changes.pop("resolution", self.resolution)
        minimum_confidence = changes.pop("minimum_confidence", self.minimum_confidence)
        character_confidence_threshold = changes.pop(
            "character_confidence_threshold", self.character_confidence_threshold
        )
        recognize_words = changes.pop("recognize_words", self.recognize_words)
        collect_symbols = changes.pop("collect_symbols", self.collect_symbols)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            mode,
            image,
            rectangle,
            page_box,
            resolution,
            minimum_confidence,
            character_confidence_threshold,
            recognize_words,
            collect_symbols,
        )


def internal_pixel_box_to_page_box(
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


def internal_map_ocr_box(
    task: internal_OcrTask,
    bbox: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    return internal_pixel_box_to_page_box(bbox, task.image.width, task.image.height, task.page_box)


def internal_ocr_region_box(
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


def internal_raster_rectangle_page_box(
    raster: internal_Raster,
    page_box: tuple[float, float, float, float],
    rectangle: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    x, y, width, height = rectangle
    return internal_pixel_box_to_page_box(
        (x, y, x + width, y + height), raster.width, raster.height, page_box
    )

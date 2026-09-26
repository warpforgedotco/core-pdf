# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, ClassVar, NoReturn, Self

from core_jbig2.bitmap import (
    compose_packed_bitmap_data,
    uint8_matrix_view,
)

frozen_setattr = object.__setattr__


JBIG2_PAGE_INFO = 48
JBIG2_END_OF_FILE = 51
JBIG2_IMMEDIATE_GENERIC_REGION = 38
JBIG2_IMMEDIATE_LOSSLESS_GENERIC_REGION = 39


class Jbig2Error(Exception):
    pass


class Jbig2ParseError(Jbig2Error):
    pass


class Jbig2UnsupportedError(Jbig2Error):
    pass


GENERIC_TEMPLATE_0_DEFAULT_AT = ((3, -1), (-3, -1), (2, -2), (-2, -2))


class JBIG2Segment:
    __slots__ = ("number", "flags", "retention_flags", "page_association", "data")

    number: int
    flags: int
    retention_flags: int
    page_association: int
    data: bytes

    __fields__: ClassVar[tuple[str, ...]] = (
        "number",
        "flags",
        "retention_flags",
        "page_association",
        "data",
    )
    __match_args__ = ("number", "flags", "retention_flags", "page_association", "data")

    def __init__(
        self,
        number: int,
        flags: int,
        retention_flags: int,
        page_association: int,
        data: bytes,
    ) -> None:
        self.number = number
        self.flags = flags
        self.retention_flags = retention_flags
        self.page_association = page_association
        self.data = data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"number={self.number!r}, "
            f"flags={self.flags!r}, "
            f"retention_flags={self.retention_flags!r}, "
            f"page_association={self.page_association!r}, "
            f"data={self.data!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.number == other.number
            and self.flags == other.flags
            and self.retention_flags == other.retention_flags
            and self.page_association == other.page_association
            and self.data == other.data
        )

    __hash__ = None  # type: ignore[assignment]

    def __replace__(self, /, **changes: Any) -> Self:
        number = changes.pop("number", self.number)
        flags = changes.pop("flags", self.flags)
        retention_flags = changes.pop("retention_flags", self.retention_flags)
        page_association = changes.pop("page_association", self.page_association)
        data = changes.pop("data", self.data)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(number, flags, retention_flags, page_association, data)

    @property
    def segment_type(self) -> int:
        return self.flags & 0x3F


class JBIG2PageInfo:
    __slots__ = ("width", "height", "x_resolution", "y_resolution", "flags")

    width: int
    height: int
    x_resolution: int
    y_resolution: int
    flags: int

    __fields__: ClassVar[tuple[str, ...]] = (
        "width",
        "height",
        "x_resolution",
        "y_resolution",
        "flags",
    )
    __match_args__ = ("width", "height", "x_resolution", "y_resolution", "flags")

    def __init__(
        self,
        width: int,
        height: int,
        x_resolution: int,
        y_resolution: int,
        flags: int,
    ) -> None:
        self.width = width
        self.height = height
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.flags = flags

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"width={self.width!r}, "
            f"height={self.height!r}, "
            f"x_resolution={self.x_resolution!r}, "
            f"y_resolution={self.y_resolution!r}, "
            f"flags={self.flags!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.width == other.width
            and self.height == other.height
            and self.x_resolution == other.x_resolution
            and self.y_resolution == other.y_resolution
            and self.flags == other.flags
        )

    __hash__ = None  # type: ignore[assignment]

    def __replace__(self, /, **changes: Any) -> Self:
        width = changes.pop("width", self.width)
        height = changes.pop("height", self.height)
        x_resolution = changes.pop("x_resolution", self.x_resolution)
        y_resolution = changes.pop("y_resolution", self.y_resolution)
        flags = changes.pop("flags", self.flags)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(width, height, x_resolution, y_resolution, flags)


class JBIG2Region:
    __slots__ = ("width", "height", "x", "y", "flags", "raw")

    width: int
    height: int
    x: int
    y: int
    flags: int
    raw: bytes

    __fields__: ClassVar[tuple[str, ...]] = ("width", "height", "x", "y", "flags", "raw")
    __match_args__ = ("width", "height", "x", "y", "flags", "raw")

    def __init__(self, width: int, height: int, x: int, y: int, flags: int, raw: bytes) -> None:
        frozen_setattr(self, "width", width)
        frozen_setattr(self, "height", height)
        frozen_setattr(self, "x", x)
        frozen_setattr(self, "y", y)
        frozen_setattr(self, "flags", flags)
        frozen_setattr(self, "raw", raw)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"width={self.width!r}, "
            f"height={self.height!r}, "
            f"x={self.x!r}, "
            f"y={self.y!r}, "
            f"flags={self.flags!r}, "
            f"raw={self.raw!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.width == other.width
            and self.height == other.height
            and self.x == other.x
            and self.y == other.y
            and self.flags == other.flags
            and self.raw == other.raw
        )

    def __hash__(self) -> int:
        return hash((self.width, self.height, self.x, self.y, self.flags, self.raw))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        width = changes.pop("width", self.width)
        height = changes.pop("height", self.height)
        x = changes.pop("x", self.x)
        y = changes.pop("y", self.y)
        flags = changes.pop("flags", self.flags)
        raw = changes.pop("raw", self.raw)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(width, height, x, y, flags, raw)


class JBIG2GenericRegionHeader:
    __slots__ = ("region", "mmr", "template", "prediction", "adaptive_pixels", "bitmap_start")

    region: JBIG2Region
    mmr: bool
    template: int
    prediction: bool
    adaptive_pixels: tuple[tuple[int, int], ...]
    bitmap_start: int

    __fields__: ClassVar[tuple[str, ...]] = (
        "region",
        "mmr",
        "template",
        "prediction",
        "adaptive_pixels",
        "bitmap_start",
    )
    __match_args__ = ("region", "mmr", "template", "prediction", "adaptive_pixels", "bitmap_start")

    def __init__(
        self,
        region: JBIG2Region,
        mmr: bool,
        template: int,
        prediction: bool,
        adaptive_pixels: tuple[tuple[int, int], ...],
        bitmap_start: int,
    ) -> None:
        frozen_setattr(self, "region", region)
        frozen_setattr(self, "mmr", mmr)
        frozen_setattr(self, "template", template)
        frozen_setattr(self, "prediction", prediction)
        frozen_setattr(self, "adaptive_pixels", adaptive_pixels)
        frozen_setattr(self, "bitmap_start", bitmap_start)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"region={self.region!r}, "
            f"mmr={self.mmr!r}, "
            f"template={self.template!r}, "
            f"prediction={self.prediction!r}, "
            f"adaptive_pixels={self.adaptive_pixels!r}, "
            f"bitmap_start={self.bitmap_start!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.region == other.region
            and self.mmr == other.mmr
            and self.template == other.template
            and self.prediction == other.prediction
            and self.adaptive_pixels == other.adaptive_pixels
            and self.bitmap_start == other.bitmap_start
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.region,
                self.mmr,
                self.template,
                self.prediction,
                self.adaptive_pixels,
                self.bitmap_start,
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
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        region = changes.pop("region", self.region)
        mmr = changes.pop("mmr", self.mmr)
        template = changes.pop("template", self.template)
        prediction = changes.pop("prediction", self.prediction)
        adaptive_pixels = changes.pop("adaptive_pixels", self.adaptive_pixels)
        bitmap_start = changes.pop("bitmap_start", self.bitmap_start)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(region, mmr, template, prediction, adaptive_pixels, bitmap_start)


class JBIG2SegmentHeader:
    __slots__ = (
        "number",
        "flags",
        "retention_flags",
        "referred_to_count",
        "referred_to_segments",
        "page_association",
        "data_length",
        "header_length",
    )

    number: int
    flags: int
    retention_flags: int
    referred_to_count: int
    referred_to_segments: list[int]
    page_association: int
    data_length: int
    header_length: int

    __fields__: ClassVar[tuple[str, ...]] = (
        "number",
        "flags",
        "retention_flags",
        "referred_to_count",
        "referred_to_segments",
        "page_association",
        "data_length",
        "header_length",
    )
    __match_args__ = (
        "number",
        "flags",
        "retention_flags",
        "referred_to_count",
        "referred_to_segments",
        "page_association",
        "data_length",
        "header_length",
    )

    def __init__(
        self,
        number: int,
        flags: int,
        retention_flags: int,
        referred_to_count: int,
        referred_to_segments: list[int],
        page_association: int,
        data_length: int,
        header_length: int,
    ) -> None:
        self.number = number
        self.flags = flags
        self.retention_flags = retention_flags
        self.referred_to_count = referred_to_count
        self.referred_to_segments = referred_to_segments
        self.page_association = page_association
        self.data_length = data_length
        self.header_length = header_length

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"number={self.number!r}, "
            f"flags={self.flags!r}, "
            f"retention_flags={self.retention_flags!r}, "
            f"referred_to_count={self.referred_to_count!r}, "
            f"referred_to_segments={self.referred_to_segments!r}, "
            f"page_association={self.page_association!r}, "
            f"data_length={self.data_length!r}, "
            f"header_length={self.header_length!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.number == other.number
            and self.flags == other.flags
            and self.retention_flags == other.retention_flags
            and self.referred_to_count == other.referred_to_count
            and self.referred_to_segments == other.referred_to_segments
            and self.page_association == other.page_association
            and self.data_length == other.data_length
            and self.header_length == other.header_length
        )

    __hash__ = None  # type: ignore[assignment]

    def __replace__(self, /, **changes: Any) -> Self:
        number = changes.pop("number", self.number)
        flags = changes.pop("flags", self.flags)
        retention_flags = changes.pop("retention_flags", self.retention_flags)
        referred_to_count = changes.pop("referred_to_count", self.referred_to_count)
        referred_to_segments = changes.pop("referred_to_segments", self.referred_to_segments)
        page_association = changes.pop("page_association", self.page_association)
        data_length = changes.pop("data_length", self.data_length)
        header_length = changes.pop("header_length", self.header_length)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            number,
            flags,
            retention_flags,
            referred_to_count,
            referred_to_segments,
            page_association,
            data_length,
            header_length,
        )


class JBIG2Image:
    __slots__ = ("width", "height", "stride", "data")

    width: int
    height: int
    stride: int
    data: bytearray

    __fields__: ClassVar[tuple[str, ...]] = ("width", "height", "stride", "data")
    __match_args__ = ("width", "height", "stride", "data")

    def __init__(self, width: int, height: int, stride: int, data: bytearray) -> None:
        self.width = width
        self.height = height
        self.stride = stride
        self.data = data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"width={self.width!r}, "
            f"height={self.height!r}, "
            f"stride={self.stride!r}, "
            f"data={self.data!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.width == other.width
            and self.height == other.height
            and self.stride == other.stride
            and self.data == other.data
        )

    __hash__ = None  # type: ignore[assignment]

    def __replace__(self, /, **changes: Any) -> Self:
        width = changes.pop("width", self.width)
        height = changes.pop("height", self.height)
        stride = changes.pop("stride", self.stride)
        data = changes.pop("data", self.data)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(width, height, stride, data)

    @classmethod
    def create(cls, width: int, height: int) -> JBIG2Image:
        if width <= 0 or height <= 0:
            raise Jbig2ParseError("invalid JBIG2 image dimensions")
        stride = (width + 7) // 8
        return cls(width=width, height=height, stride=stride, data=bytearray(stride * height))

    def fill(self, value: int) -> None:
        fill_byte = 0xFF if value else 0x00
        self.data[:] = bytes([fill_byte]) * len(self.data)


def read_be_u32(data: bytes, pos: int) -> int:
    chunk = data[pos : pos + 4]
    if len(chunk) != 4:
        raise Jbig2ParseError("truncated JBIG2 data")
    return int.from_bytes(chunk, "big")


def read_be_i32(data: bytes, pos: int) -> int:
    chunk = data[pos : pos + 4]
    if len(chunk) != 4:
        raise Jbig2ParseError("truncated JBIG2 data")
    return int.from_bytes(chunk, "big", signed=True)


def read_be_i8(data: bytes, pos: int) -> int:
    value = data[pos]
    return value - 0x100 if value & 0x80 else value


def parse_page_info(data: bytes) -> JBIG2PageInfo:
    if len(data) < 19:
        raise Jbig2ParseError("truncated JBIG2 page info")
    width = read_be_u32(data, 0)
    height = read_be_u32(data, 4)
    x_resolution = read_be_u32(data, 8)
    y_resolution = read_be_u32(data, 12)
    flags = data[16]
    return JBIG2PageInfo(
        width=width,
        height=height,
        x_resolution=x_resolution,
        y_resolution=y_resolution,
        flags=flags,
    )


def parse_region(data: bytes, kind: str) -> JBIG2Region:
    if len(data) < 17:
        raise Jbig2ParseError(f"truncated JBIG2 {kind} region")
    return JBIG2Region(
        width=read_be_u32(data, 0),
        height=read_be_u32(data, 4),
        x=read_be_i32(data, 8),
        y=read_be_i32(data, 12),
        flags=data[16],
        raw=data,
    )


def parse_generic_region_header(region: JBIG2Region) -> JBIG2GenericRegionHeader:
    data = region.raw
    if len(data) < 18:
        raise Jbig2ParseError("truncated JBIG2 generic region")
    flags = data[17]
    mmr = bool(flags & 1)
    template = (flags >> 1) & 3
    prediction = bool(flags & 8)
    pos = 18
    at: list[tuple[int, int]] = []
    if not mmr:
        at_count = 4 if template == 0 else 1
        if len(data) < pos + at_count * 2:
            raise Jbig2ParseError("truncated JBIG2 generic region")
        for ignored in range(at_count):
            x = read_be_i8(data, pos)
            y = read_be_i8(data, pos + 1)
            at.append((x, y))
            pos += 2
    return JBIG2GenericRegionHeader(region, mmr, template, prediction, tuple(at), pos)


def read_u8(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 1 > len(data):
        raise Jbig2ParseError("truncated JBIG2 data")
    return data[pos], pos + 1


def read_u32(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 4 > len(data):
        raise Jbig2ParseError("truncated JBIG2 data")
    return read_be_u32(data, pos), pos + 4


def read_u24(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 3 > len(data):
        raise Jbig2ParseError("truncated JBIG2 data")
    return int.from_bytes(data[pos : pos + 3], "big"), pos + 3


def parse_page_association(data: bytes, pos: int, long_form: bool) -> tuple[int, int]:
    if long_form:
        return read_u32(data, pos)
    return read_u8(data, pos)


def parse_referred_to_segments(
    data: bytes, pos: int, count: int, long_form: bool
) -> tuple[list[int], int]:
    return read_segment_references(data, pos, count, 4 if long_form else 1)


def read_segment_references(data: bytes, pos: int, count: int, width: int) -> tuple[list[int], int]:
    end = pos + count * width
    if end > len(data):
        raise Jbig2ParseError("truncated JBIG2 segment references")
    return [
        int.from_bytes(data[index : index + width], "big") for index in range(pos, end, width)
    ], end


def parse_segment_header(data: bytes, pos: int) -> tuple[JBIG2SegmentHeader, int]:
    start = pos
    if pos + 11 > len(data):
        raise Jbig2ParseError("truncated JBIG2 segment header")
    number, pos = read_u32(data, pos)
    flags, pos = read_u8(data, pos)
    retention_flags, pos = read_u8(data, pos)
    referred_to_count = retention_flags >> 5
    if referred_to_count == 7:
        ref_count, pos = read_u24(data, pos)
        referred_to_count = ((retention_flags & 0x1F) << 24) | ref_count
        bit_bytes = (referred_to_count + 1 + 7) // 8
        if pos + bit_bytes > len(data):
            raise Jbig2ParseError("truncated JBIG2 segment header")
        pos += bit_bytes
    elif referred_to_count in (5, 6):
        raise Jbig2ParseError("invalid JBIG2 referred-to segment count")
    reference_width = 1 if number <= 256 else 2 if number <= 65536 else 4
    referred_to_segments, pos = read_segment_references(
        data, pos, referred_to_count, reference_width
    )
    if pos + (4 if (flags & 0x40) else 1) + 4 > len(data):
        raise Jbig2ParseError("truncated JBIG2 segment header")
    page_association, pos = parse_page_association(data, pos, bool(flags & 0x40))
    data_length, pos = read_u32(data, pos)
    return (
        JBIG2SegmentHeader(
            number=number,
            flags=flags,
            retention_flags=retention_flags,
            referred_to_count=referred_to_count,
            referred_to_segments=referred_to_segments,
            page_association=page_association,
            data_length=data_length,
            header_length=pos - start,
        ),
        pos,
    )


def parse_embedded_segments(data: bytes) -> list[JBIG2Segment]:
    pos = 0
    segments: list[JBIG2Segment] = []
    while pos + 11 <= len(data):
        header, pos = parse_segment_header(data, pos)
        if header.data_length == 0xFFFFFFFF or pos + header.data_length > len(data):
            payload = data[pos:]
            pos = len(data)
        else:
            payload = data[pos : pos + header.data_length]
            pos += header.data_length
        segments.append(
            JBIG2Segment(
                number=header.number,
                flags=header.flags,
                retention_flags=header.retention_flags,
                page_association=header.page_association,
                data=payload,
            )
        )
        if header.flags & 0x3F == JBIG2_END_OF_FILE:
            break
    return segments


class JBIG2PageDecoder:
    def __init__(self) -> None:
        self.page_info: JBIG2PageInfo | None = None
        self.image: JBIG2Image | None = None
        self.max_x = 0
        self.max_y = 0

    def ensure_image(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        image = self.image
        if image is None:
            self.image = JBIG2Image.create(width, height)
            return
        if width <= image.width and height <= image.height:
            return
        new_image = JBIG2Image.create(max(width, image.width), max(height, image.height))
        source = uint8_matrix_view(image.data, image.height, image.stride)
        destination = uint8_matrix_view(new_image.data, new_image.height, new_image.stride)
        destination[: image.height, : image.stride] = source
        self.image = new_image

    def include_region(self, region: JBIG2Region) -> None:
        self.max_x = max(self.max_x, region.x + region.width)
        self.max_y = max(self.max_y, region.y + region.height)
        self.ensure_image(self.max_x, self.max_y)

    def decode_segment(self, segment: JBIG2Segment) -> None:
        kind = segment.segment_type
        if kind == JBIG2_PAGE_INFO:
            self.page_info = parse_page_info(segment.data)
            self.image = JBIG2Image.create(self.page_info.width, self.page_info.height)
            self.image.fill(jbig2_page_default_pixel(self.page_info))
        elif kind in (4, 6, 7):
            region = parse_region(segment.data, "text")
            self.include_region(region)
            self.decode_text_region(region)
        elif kind in (JBIG2_IMMEDIATE_GENERIC_REGION, JBIG2_IMMEDIATE_LOSSLESS_GENERIC_REGION):
            region = parse_region(segment.data, "generic")
            self.include_region(region)
            if self.image is not None:
                self.decode_generic_region(parse_generic_region_header(region))
        elif kind in (49, JBIG2_END_OF_FILE):
            if segment.data:
                raise Jbig2ParseError("JBIG2 end marker has segment data")
        elif kind == 52:
            if len(segment.data) < 4:
                raise Jbig2ParseError("truncated JBIG2 profiles segment")
            count = read_be_u32(segment.data, 0)
            if len(segment.data) != 4 + count * 4:
                raise Jbig2ParseError("invalid JBIG2 profiles segment length")
        elif kind == 62:
            if len(segment.data) < 4:
                raise Jbig2ParseError("truncated JBIG2 extension segment")
            extension = read_be_u32(segment.data, 0)
            if extension & (1 << 31):
                if not extension & (1 << 29):
                    raise Jbig2ParseError("necessary JBIG2 extension requires reserved bit 29")
                raise Jbig2UnsupportedError("unsupported necessary JBIG2 extension")
        elif kind in (0, 16, 20, 22, 23, 36, 40, 42, 43, 50, 53):
            raise Jbig2UnsupportedError(f"unsupported JBIG2 segment type {kind}")
        else:
            raise Jbig2ParseError(f"reserved JBIG2 segment type {kind}")

    def decode_text_region(self, region: JBIG2Region) -> None:
        if len(region.raw) < 20:
            raise Jbig2ParseError("truncated JBIG2 text region")
        raise Jbig2UnsupportedError("JBIG2 text region decoding is not implemented")

    def decode_generic_region(self, header: JBIG2GenericRegionHeader) -> None:
        if header.mmr:
            raise Jbig2UnsupportedError("JBIG2 MMR region decoding is not implemented")
        raise Jbig2UnsupportedError("JBIG2 arithmetic region decoding is not implemented")

    def finish(self) -> bytes:
        if self.image is None:
            raise Jbig2UnsupportedError("JBIG2Decode produced no image")
        return bytes(self.image.data)


def jbig2_page_default_pixel(page_info: JBIG2PageInfo) -> int:
    return (page_info.flags >> 2) & 1


def jbig2_page_combination_operator(page_info: JBIG2PageInfo | None) -> int:
    if page_info is None:
        return 0
    return (page_info.flags >> 3) & 3


def jbig2_page_allows_region_operator(page_info: JBIG2PageInfo | None) -> bool:
    return page_info is not None and bool(page_info.flags & 64)


def compose_packed_bitmap_region(
    region: JBIG2Region,
    packed_bitmap: bytes | bytearray,
    image: JBIG2Image,
    page_info: JBIG2PageInfo | None,
) -> None:
    operator = region_operator(region, page_info)
    if operator not in (0, 2):
        raise Jbig2UnsupportedError(f"unsupported JBIG2 combination operator {operator}")
    row_bytes = max(1, (region.width + 7) // 8)
    compose_packed_bitmap_data(
        packed_bitmap,
        min(region.height, len(packed_bitmap) // row_bytes),
        region.width,
        region.x,
        region.y,
        image.width,
        image.height,
        image.stride,
        image.data,
        operator,
    )


def region_operator(region: JBIG2Region, page_info: JBIG2PageInfo | None) -> int:
    if jbig2_page_allows_region_operator(page_info):
        return region.flags & 7
    return jbig2_page_combination_operator(page_info)


__all__ = (
    "JBIG2Region",
    "JBIG2GenericRegionHeader",
    "JBIG2PageDecoder",
    "parse_region",
    "parse_embedded_segments",
    "JBIG2_PAGE_INFO",
    "JBIG2_END_OF_FILE",
    "JBIG2_IMMEDIATE_GENERIC_REGION",
    "JBIG2_IMMEDIATE_LOSSLESS_GENERIC_REGION",
    "Jbig2Error",
    "Jbig2ParseError",
    "Jbig2UnsupportedError",
    "GENERIC_TEMPLATE_0_DEFAULT_AT",
    "JBIG2Segment",
    "JBIG2PageInfo",
    "JBIG2SegmentHeader",
    "JBIG2Image",
    "read_be_u32",
    "read_be_i32",
    "read_be_i8",
    "parse_page_info",
    "parse_generic_region_header",
    "read_u8",
    "read_u32",
    "read_u24",
    "parse_page_association",
    "parse_referred_to_segments",
    "parse_segment_header",
    "jbig2_page_default_pixel",
    "jbig2_page_combination_operator",
    "jbig2_page_allows_region_operator",
    "compose_packed_bitmap_region",
    "region_operator",
)

"""Private helpers shared by the compat facades."""

from __future__ import annotations

import struct
import zlib
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, cast

from core_pdf.impl._impl.model.geometry import rect_tuple

BBox = tuple[float, float, float, float]

LIGATURES = {"ff": "ﬀ", "fi": "ﬁ", "fl": "ﬂ", "ffi": "ﬃ", "ffl": "ﬄ"}


class ClosingMixin:
    def close(self) -> None:
        return None

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def coerce_bbox(value: object) -> BBox:
    box = rect_tuple(value)
    if box is None:
        raise ValueError(f"value does not describe a rectangle: {value!r}")
    return box


def write_bytes(target: str | PathLike[str] | BinaryIO, data: bytes) -> None:
    if isinstance(target, (str, PathLike)):
        Path(cast(str | PathLike[str], target)).write_bytes(data)
    else:
        target.write(data)


def float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def encode_png(width: int, height: int, channels: int, pixels: bytes | bytearray) -> bytes:
    if channels not in (3, 4):
        raise ValueError("PNG output requires RGB or RGBA pixels")
    stride = width * channels
    scanlines = b"".join(
        b"\x00" + pixels[row * stride : (row + 1) * stride] for row in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 6 if channels == 4 else 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(scanlines))
        + png_chunk(b"IEND", b"")
    )

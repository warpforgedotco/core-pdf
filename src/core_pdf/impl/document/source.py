# SPDX-License-Identifier: AGPL-3.0-only

"""Reading a PDF's bytes: the source as a buffer or memory map, and probes of its xref
streams and object headers.
"""

from __future__ import annotations

import contextlib
import mmap
from array import array
from collections.abc import Iterable, Sequence
from os import PathLike
from typing import Any, BinaryIO

lazy import numpy

from core_pdf.impl.base.exceptions import PdfEmptySourceError, PdfSourceError
from core_pdf.impl.base.types import PdfByteBuffer, PdfSource
from core_pdf.impl.base.values import recover_pdf_name
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
lazy from core_pdf_cythonized import object_headers_match, regions_with_markers

XREF_STREAM_MARKERS = (b"#", b"XRef", b"/W", b"/Size")


def is_xref_stream_dictionary(dictionary: PdfDict) -> bool:
    return recover_pdf_name(dictionary.get("Type")) == "XRef" or (
        dictionary.get("W") is not None and dictionary.get("Size") is not None
    )


def xref_stream_candidates(
    data: bytes | mmap.mmap, starts: Sequence[int], data_len: int
) -> list[int]:
    """Indexes of the regions a may_be_xref_stream check would accept, in ascending order."""
    hashes, xrefs, widths, sizes = regions_with_markers(data, XREF_STREAM_MARKERS, starts, data_len)
    return sorted(hashes | xrefs | (widths & sizes))


def load_source(source: PdfSource) -> tuple[PdfByteBuffer, BinaryIO | None]:
    if isinstance(source, (str, PathLike)):
        if isinstance(source, str) and source.startswith("%PDF"):
            return source.encode("latin-1"), None
        # The mapping holds its own reference, so the file need not stay open. Keeping it
        # open leaked a handle from every document that was never closed explicitly.
        with open(source, "rb") as file_handle:
            try:
                return mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ), None
            except (OSError, ValueError) as exc:
                try:
                    is_empty = file_handle.seek(0, 2) == 0
                except OSError:
                    is_empty = False
                if is_empty:
                    raise PdfEmptySourceError("PDF source is empty") from exc
                raise PdfSourceError(str(exc)) from exc
    if isinstance(source, bytes):
        return source, None
    if isinstance(source, (memoryview, bytearray)):
        return bytes(source), None

    mapped = try_mmap_reader(source)
    if mapped is not None:
        return mapped, None

    read = getattr(source, "read", None)
    if not callable(read):
        raise PdfSourceError(f"PDF source type {type(source).__name__} is not supported")
    reader = source
    tell = getattr(source, "tell", None)
    seek = getattr(source, "seek", None)
    position: int | None = None
    if callable(tell) and callable(seek):
        try:
            position = tell()
            seek(0)
        except OSError, TypeError, ValueError:
            position = None
    try:
        raw = reader.read()
    except OSError as exc:
        raise PdfSourceError(str(exc)) from exc
    finally:
        if position is not None and callable(seek):
            seek(position)
    return (raw if isinstance(raw, bytes) else bytes(raw)), None


def try_mmap_reader(source: object) -> mmap.mmap | None:
    fileno = getattr(source, "fileno", None)
    if not callable(fileno):
        return None
    try:
        fd = fileno()
    except OSError, TypeError, ValueError:
        return None
    try:
        return mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    except ValueError as error:
        raise PdfEmptySourceError("PDF source is empty") from error
    except OSError:
        return None


HUGE_OBJECT_NUMBER = 1 << 63


def object_headers_present(data: Any, keys: list[int], offsets: list[int]) -> list[bool]:
    data_len = len(data)
    try:
        key_column = numpy.array(keys, dtype=numpy.int64)
        offset_column = numpy.array(offsets, dtype=numpy.int64)
    except OverflowError:
        return object_headers_present_by_entry(data, keys, offsets)
    numbers = key_column >> 16
    generations = key_column & 0xFFFF
    offset_column[(offset_column < 0) | (offset_column >= data_len)] = -1
    states = object_headers_match(data, numbers, generations, offset_column)
    return (states == 1).tolist()


def object_headers_present_by_entry(data: Any, keys: list[int], offsets: list[int]) -> list[bool]:
    data_len = len(data)
    numbers = array("q")
    generations = array("q")
    clamped = array("q")
    for key, offset in zip(keys, offsets, strict=True):
        number = key >> 16
        numbers.append(number if number < HUGE_OBJECT_NUMBER else -2)
        generations.append(key & 0xFFFF)
        clamped.append(offset if 0 <= offset < data_len else -1)
    states = object_headers_match(data, numbers, generations, clamped)
    present = [state == 1 for state in states.tolist()]
    for index, state in enumerate(states.tolist()):
        if state == 2:
            offset = clamped[index]
            end = offset
            while end < data_len and 0x30 <= data[end] <= 0x39:
                end += 1
            present[index] = int(bytes(data[offset:end])) == keys[index] >> 16
    return present


TRAILER_METADATA_KEYS = ("Info", "ID", "Encrypt", "AuthCode")


def first_indexes(values: Iterable[object]) -> dict[object, int]:
    indexes: dict[object, int] = {}
    for index, value in enumerate(values):
        with contextlib.suppress(TypeError):
            indexes.setdefault(value, index)
    return indexes

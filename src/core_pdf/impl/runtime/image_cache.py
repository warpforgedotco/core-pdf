# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded, document-owned cache for decoded images and raster variants."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

DEFAULT_IMAGE_CACHE_BYTES = 1 << 30
IMAGE_CACHE_BYTES_ENV = "CORE_PDF_IMAGE_CACHE_BYTES"


@dataclass(frozen=True, slots=True)
class ImageCacheKey:
    """Stable namespace and options for one immutable image-derived value."""

    kind: str
    identity: tuple[Hashable, ...]
    options: tuple[Hashable, ...] = ()


@dataclass(frozen=True, slots=True)
class ImageCacheStats:
    hits: int
    misses: int
    evictions: int
    bypasses: int
    entries: int
    bytes: int
    peak_bytes: int


@dataclass(slots=True)
class _ImageCacheEntry:
    value: Any
    size: int


def image_cache_value_size(value: object) -> int:
    """Return the pixel-buffer footprint of a cache value when available."""
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int) and nbytes >= 0:
        return nbytes
    if isinstance(value, (bytes, bytearray, memoryview)):
        return memoryview(value).nbytes
    return 0


def image_cache_default_bytes() -> int:
    raw = os.environ.get(IMAGE_CACHE_BYTES_ENV)
    if raw is None:
        return DEFAULT_IMAGE_CACHE_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_IMAGE_CACHE_BYTES


class ImageCache:
    """Thread-safe LRU with byte-budgeted immutable values and single-flight fills."""

    __slots__ = (
        "max_bytes",
        "_entries",
        "_inflight",
        "_lock",
        "_bytes",
        "_peak_bytes",
        "_hits",
        "_misses",
        "_evictions",
        "_bypasses",
    )

    def __init__(self, max_bytes: int | None = None) -> None:
        self.max_bytes = image_cache_default_bytes() if max_bytes is None else max(0, max_bytes)
        self._entries: OrderedDict[ImageCacheKey, _ImageCacheEntry] = OrderedDict()
        self._inflight: dict[ImageCacheKey, threading.Event] = {}
        self._lock = threading.RLock()
        self._bytes = 0
        self._peak_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._bypasses = 0

    def get(self, key: ImageCacheKey) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, key: ImageCacheKey, value: Any, *, size: int | None = None) -> Any:
        entry_size = image_cache_value_size(value) if size is None else max(0, size)
        with self._lock:
            if self.max_bytes <= 0 or entry_size > self.max_bytes:
                self._bypasses += 1
                return value
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous.size
            self._entries[key] = _ImageCacheEntry(value, entry_size)
            self._bytes += entry_size
            self._peak_bytes = max(self._peak_bytes, self._bytes)
            while self._entries and self._bytes > self.max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted.size
                self._evictions += 1
        return value

    def get_or_create(self, key: ImageCacheKey, factory: Callable[[], Any]) -> Any:
        """Return a cached value, allowing only one concurrent factory call per key."""
        while True:
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None:
                    self._entries.move_to_end(key)
                    self._hits += 1
                    return entry.value
                event = self._inflight.get(key)
                if event is None:
                    event = threading.Event()
                    self._inflight[key] = event
                    owner = True
                else:
                    owner = False
                self._misses += 1
            if owner:
                break
            event.wait()

        try:
            value = factory()
            if value is not None:
                self.put(key, value)
            return value
        finally:
            with self._lock:
                completed = self._inflight.pop(key, None)
                if completed is not None:
                    completed.set()

    def invalidate(self, prefix: tuple[Hashable, ...] | None = None) -> None:
        with self._lock:
            if prefix is None:
                self._entries.clear()
                self._bytes = 0
                return
            for key in tuple(self._entries):
                flattened = (key.kind, *key.identity, *key.options)
                if flattened[: len(prefix)] == prefix:
                    self._bytes -= self._entries.pop(key).size

    def clear(self) -> None:
        self.invalidate()

    def stats(self) -> ImageCacheStats:
        with self._lock:
            return ImageCacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                bypasses=self._bypasses,
                entries=len(self._entries),
                bytes=self._bytes,
                peak_bytes=self._peak_bytes,
            )


__all__ = (
    "DEFAULT_IMAGE_CACHE_BYTES",
    "IMAGE_CACHE_BYTES_ENV",
    "ImageCache",
    "ImageCacheKey",
    "ImageCacheStats",
    "image_cache_default_bytes",
    "image_cache_value_size",
)

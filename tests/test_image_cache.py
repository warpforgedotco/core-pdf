from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from core_pdf.impl.extract.ocr.types import internal_Raster
from core_pdf.impl.render.model import RasterImage
from core_pdf.impl.runtime.image_cache import ImageCache, ImageCacheKey


def key(name: str) -> ImageCacheKey:
    return ImageCacheKey("test", (name,))


def test_image_cache_evicts_by_bytes_and_tracks_stats() -> None:
    cache = ImageCache(max_bytes=8)
    cache.put(key("a"), b"1234")
    cache.put(key("b"), b"5678")
    assert cache.get(key("a")) == b"1234"

    cache.put(key("c"), b"9012")

    assert cache.get(key("a")) == b"1234"
    assert cache.get(key("b")) is None
    assert cache.get(key("c")) == b"9012"
    stats = cache.stats()
    assert stats.evictions == 1
    assert stats.entries == 2
    assert stats.bytes == 8


def test_image_cache_bypasses_values_larger_than_budget() -> None:
    cache = ImageCache(max_bytes=3)
    cache.put(key("large"), b"1234")

    assert cache.get(key("large")) is None
    assert cache.stats().bypasses == 1


def test_image_cache_budgets_internal_rasters_by_pixel_bytes() -> None:
    cache = ImageCache(max_bytes=6)
    first = internal_Raster(RasterImage(b"1234", 2, 2, 1), 72)
    second = internal_Raster(RasterImage(b"5678", 2, 2, 1), 72)

    assert first.nbytes == 4
    cache.put(key("first-raster"), first)
    cache.put(key("second-raster"), second)

    assert cache.get(key("first-raster")) is None
    assert cache.get(key("second-raster")) is second
    assert cache.stats().bytes == 4
    assert cache.stats().evictions == 1

    oversized = internal_Raster(RasterImage(bytes(8), 2, 2, 2), 72)
    cache.put(key("oversized-raster"), oversized)
    assert cache.get(key("oversized-raster")) is None
    assert cache.stats().bypasses == 1


def test_image_cache_single_flights_concurrent_factory() -> None:
    cache = ImageCache(max_bytes=100)
    workers = 4
    calls = 0
    calls_lock = threading.Lock()
    start = threading.Barrier(workers + 1)
    release_factory = threading.Event()

    def factory() -> bytes:
        nonlocal calls
        with calls_lock:
            calls += 1
        assert release_factory.wait(timeout=2)
        return b"shared"

    def load() -> bytes:
        start.wait()
        return cache.get_or_create(key("x"), factory)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(load) for _ in range(workers)]
        start.wait()
        deadline = time.monotonic() + 2.0
        while cache.stats().misses < workers and time.monotonic() < deadline:
            time.sleep(0.001)
        try:
            assert cache.stats().misses == workers
        finally:
            release_factory.set()
        values = [future.result(timeout=2) for future in futures]

    assert values == [b"shared"] * 4
    assert calls == 1
    assert cache.stats().hits == workers - 1


def test_image_cache_invalidate_prefix() -> None:
    cache = ImageCache(max_bytes=100)
    first = ImageCacheKey("image", ("source", 1), ("variant",))
    second = ImageCacheKey("image", ("source", 2), ("variant",))
    cache.put(first, b"one")
    cache.put(second, b"two")

    cache.invalidate(("image", "source", 1))

    assert cache.get(first) is None
    assert cache.get(second) == b"two"

from __future__ import annotations

import threading
import time

from core_pdf.impl.engine.image_cache import ImageCache, ImageCacheKey


def key(name: str) -> ImageCacheKey:
    return ImageCacheKey("test", (name,))


def test_image_cache_evicts_by_bytes_and_tracks_stats() -> None:
    cache = ImageCache(max_bytes=6)
    cache.put(key("a"), b"1234")
    cache.put(key("b"), b"5678")

    assert cache.get(key("a")) is None
    assert cache.get(key("b")) == b"5678"
    stats = cache.stats()
    assert stats.evictions == 1
    assert stats.bytes == 4


def test_image_cache_bypasses_values_larger_than_budget() -> None:
    cache = ImageCache(max_bytes=3)
    cache.put(key("large"), b"1234")

    assert cache.get(key("large")) is None
    assert cache.stats().bypasses == 1


def test_image_cache_single_flights_concurrent_factory() -> None:
    cache = ImageCache(max_bytes=100)
    calls = 0
    calls_lock = threading.Lock()

    def factory() -> bytes:
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.01)
        return b"shared"

    values: list[bytes] = []
    threads = [
        threading.Thread(target=lambda: values.append(cache.get_or_create(key("x"), factory)))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert values == [b"shared"] * 4
    assert calls == 1


def test_image_cache_invalidate_prefix() -> None:
    cache = ImageCache(max_bytes=100)
    first = ImageCacheKey("image", ("source", 1), ("variant",))
    second = ImageCacheKey("image", ("source", 2), ("variant",))
    cache.put(first, b"one")
    cache.put(second, b"two")

    cache.invalidate(("image", "source", 1))

    assert cache.get(first) is None
    assert cache.get(second) == b"two"

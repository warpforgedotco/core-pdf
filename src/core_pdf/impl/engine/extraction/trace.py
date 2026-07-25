# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import os
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator, MutableMapping

TRACE_CACHE_KEY = "extraction_stage_trace"


def extraction_tracing_enabled() -> bool:
    return os.environ.get("CORE_PDF_TRACE", "").strip().casefold() in {"1", "true", "yes", "on"}


@contextmanager
def extraction_stage(
    cache: MutableMapping[Any, object],
    name: str,
) -> Iterator[None]:
    """Record an additive stage duration in the page cache when tracing is enabled."""
    if not extraction_tracing_enabled():
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - started
        existing = cache.get(TRACE_CACHE_KEY)
        stages = dict(existing) if isinstance(existing, dict) else {}
        stages[name] = float(stages.get(name, 0.0)) + elapsed
        cache[TRACE_CACHE_KEY] = stages


__all__ = ("TRACE_CACHE_KEY", "extraction_stage", "extraction_tracing_enabled")

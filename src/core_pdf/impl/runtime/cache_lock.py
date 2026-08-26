# SPDX-License-Identifier: AGPL-3.0-only
"""Shared cache-lock access for documents and lightweight document doubles."""

from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

internal_FALLBACK_CACHE_LOCK = threading.RLock()

internal_T = TypeVar("internal_T")


def document_cache_lock(document: Any) -> Any:
    return getattr(document, "internal_cache_lock", internal_FALLBACK_CACHE_LOCK)


def get_or_compute(document: Any, cache_attr: str, compute: Callable[[], internal_T]) -> internal_T:
    """Return ``document.<cache_attr>``, computing and storing it once under the
    document's cache lock if it is still unset. Only fits caches where a
    present value is never falsy-but-valid None and where computing has no
    early-exit/side-effect branches -- callers with that shape should keep
    their own inline handling instead of forcing this helper."""
    with document_cache_lock(document):
        value = getattr(document, cache_attr)
        if value is None:
            value = compute()
            setattr(document, cache_attr, value)
        return value

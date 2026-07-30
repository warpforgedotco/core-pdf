# SPDX-License-Identifier: AGPL-3.0-only
"""Shared lock access for document mixins and lightweight document doubles."""

from __future__ import annotations

import threading
from typing import Any

internal_FALLBACK_CACHE_LOCK = threading.RLock()


def document_cache_lock(document: Any) -> Any:
    return getattr(document, "internal_cache_lock", internal_FALLBACK_CACHE_LOCK)


def document_recovery_enabled(document: Any) -> bool:
    return bool(
        getattr(document, "xref_was_recovered", False)
        or getattr(document, "page_tree_was_recovered", False)
    )

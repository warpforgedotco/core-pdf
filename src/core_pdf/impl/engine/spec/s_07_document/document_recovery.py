# SPDX-License-Identifier: AGPL-3.0-only
"""Whether a document was reconstructed and therefore needs lenient traversal."""

from __future__ import annotations

from typing import Any


def document_recovery_enabled(document: Any) -> bool:
    return bool(
        getattr(document, "xref_was_recovered", False)
        or getattr(document, "page_tree_was_recovered", False)
    )

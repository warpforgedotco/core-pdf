# SPDX-License-Identifier: AGPL-3.0-only
"""Compatibility imports for PDF name and number tree traversal."""

from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_objects.trees import (
    iter_name_tree_items,
    iter_number_tree_items,
)

__all__ = ("iter_name_tree_items", "iter_number_tree_items")

# SPDX-License-Identifier: AGPL-3.0-only
"""Legacy import surface for table extraction internals."""

from core_pdf.impl.extract.table_detection import (
    internal_compact_stream_table,
    internal_stream_table,
    internal_stream_tables,
)
from core_pdf.impl.extract.table_pipeline import extract_tables

__all__ = (
    "extract_tables",
    "internal_compact_stream_table",
    "internal_stream_table",
    "internal_stream_tables",
)

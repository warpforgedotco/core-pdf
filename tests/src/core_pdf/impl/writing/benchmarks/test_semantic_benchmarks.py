# SPDX-License-Identifier: AGPL-3.0-only
"""Performance coverage for semantic PDF construction."""

from __future__ import annotations

import pytest

from core_pdf.impl.structured.model import Block, BlockKind, Document, Page, TextLine
from core_pdf.impl.writing.semantic import serialize_document_to_pdf


def internal_document(page_count: int = 16) -> Document:
    return Document(
        pages=tuple(
            Page(
                page_number=page_number,
                width=612.0,
                height=792.0,
                blocks=(
                    Block(
                        0,
                        BlockKind.HEADING,
                        (TextLine(f"Page {page_number}", bbox=(36.0, 740.0, 180.0, 758.0)),),
                        level=1,
                    ),
                    Block(
                        1,
                        BlockKind.PARAGRAPH,
                        tuple(
                            TextLine(
                                f"Semantic writer benchmark line {index}",
                                bbox=(36.0, 700.0 - index * 14.0, 360.0, 712.0 - index * 14.0),
                            )
                            for index in range(24)
                        ),
                    ),
                ),
            )
            for page_number in range(1, page_count + 1)
        ),
        metadata={"Title": "Semantic writer benchmark", "Lang": "en-US"},
    )


@pytest.mark.benchmark_high_impact
def test_tagged_semantic_writer_benchmark(benchmark) -> None:
    document = internal_document()

    result = benchmark(serialize_document_to_pdf, document)

    assert result.startswith(b"%PDF-1.7")
    assert len(result) > 10_000

# SPDX-License-Identifier: AGPL-3.0-only
"""Serialization of the normalized structured document.

Every caller that asks for JSON pays this after extraction has finished, over
whatever the extractor produced. The input is therefore a real extracted
document rather than a synthesized one: the node, line and table mix that
``document_to_json_dict`` walks is the mix a real page produces, and it changes
when extraction changes.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.serialize import document_to_json_dict
from tests.helpers.benchmark_pages import DENSE_PDF
from tests.helpers.paths import require_fixture


@pytest.fixture(scope="module")
def extracted_document() -> Iterator[Any]:
    with PdfDocument.open(require_fixture(DENSE_PDF)) as document:
        yield document.extract()


def test_normalized_json_graph_benchmark(benchmark, extracted_document) -> None:
    record = benchmark(document_to_json_dict, extracted_document)

    assert record["schema_version"]
    assert len(record["pages"]) == 1
    assert record["lines"]

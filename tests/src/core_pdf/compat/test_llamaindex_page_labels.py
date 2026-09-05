# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from core_pdf import PdfDocument
from core_pdf.api.compat.llamaindex import load_data
from core_pdf.impl.spec.s_07_document.document import internal_PageNode
from tests.helpers.pdf_bytes import HELVETICA, assemble_pdf, stream_obj


def labeled_reader_pdf(labels: bytes) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R " + labels + b" >>",
        b"<< /Type /Pages /Kids [4 0 R 6 0 R 8 0 R 10 0 R 12 0 R] /Count 5 "
        b"/MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> >>",
        HELVETICA,
    ]
    for number in range(5):
        objects.extend(
            [
                f"<< /Type /Page /Parent 2 0 R /Contents {5 + number * 2} 0 R >>".encode(),
                stream_obj(f"BT /F1 12 Tf 50 700 Td (Page {number + 1}) Tj ET".encode()),
            ]
        )
    return assemble_pdf(objects)


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (b"", ["1", "2", "3", "4", "5"]),
        (b"/PageLabels << /Nums [0 << >>] >>", ["1", "2", "3", "4", "5"]),
        (
            b"/PageLabels << /Nums [0 << /S /r >> 2 << /S /D /St 7 /P (A-) >>] >>",
            ["i", "ii", "A-7", "A-8", "A-9"],
        ),
    ],
    ids=["absent", "empty-label-fallback", "mixed-ranges"],
)
def test_reader_reuses_pages_and_labels_without_changing_metadata(
    labels: bytes, expected: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.pdf"
    path.write_bytes(labeled_reader_pdf(labels))
    with PdfDocument(path) as document:
        assert [page.label or str(page.page_number) for page in document.pages] == expected

    walks = 0
    original = PdfDocument.internal_iter_page_nodes

    def count_walks(document: PdfDocument) -> Iterator[internal_PageNode]:
        nonlocal walks
        walks += 1
        yield from original(document)

    monkeypatch.setattr(PdfDocument, "internal_iter_page_nodes", count_walks)
    results = load_data(path, extra_info={"origin": "fixture", "page_label": "overridden"})

    assert [result.text for result in results] == [f"Page {number}" for number in range(1, 6)]
    assert [result.metadata for result in results] == [
        {"origin": "fixture", "page_label": label, "file_name": "labels.pdf"} for label in expected
    ]
    assert walks == 1


def test_page_labels_are_fresh_after_the_number_tree_changes() -> None:
    with PdfDocument(labeled_reader_pdf(b"/PageLabels << /Nums [0 << /S /D >>] >>")) as document:
        assert document.page_labels == ["1", "2", "3", "4", "5"]
        labels = document.resolver.resolve_dict(document.catalog()["PageLabels"])
        assert labels is not None
        labels["Nums"] = [0, {"P": "Revised-"}]

        assert document.page_labels == ["Revised-"] * 5
        assert document.pages[0].label == "Revised-"

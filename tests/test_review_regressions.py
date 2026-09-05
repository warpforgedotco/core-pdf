# SPDX-License-Identifier: AGPL-3.0-only
"""Document lifecycle regressions that need a real corpus document."""

from __future__ import annotations

import mmap
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest

from core_pdf import PdfDocument as PublicPdfDocument
from core_pdf import PdfSourceError
from core_pdf.impl.runtime.execution import internal_ExtractionCancelled
from core_pdf.impl.spec.s_07_document.document import PdfDocument
from tests.helpers.paths import score_bench_pdf


def test_explicit_crypt_metadata_stream_uses_document_security_handler() -> None:
    with PublicPdfDocument(simple_pdf_fixture()) as document:
        xmp = document.get_metadata()["xmp"]

    assert xmp is not None
    assert "parse_error" not in xmp
    assert xmp["tag"].endswith("xmpmeta")


def simple_pdf_fixture() -> Path:
    return score_bench_pdf("g-325a.pdf")


def test_nested_form_rebinds_same_named_font_resource() -> None:
    fixture = score_bench_pdf("korean_power_system_challenges-p003.pdf")

    with PublicPdfDocument(fixture) as document:
        result = cast(Any, document.pages[0]).extract()
        text = "\n".join(line.text for block in result.blocks for line in block.lines)

    assert "This document was prepared as an account of work" in text
    assert "5Iis document was prepared as an account of worL" not in text


def test_document_close_releases_owned_path_resources() -> None:
    document: PdfDocument[Any] = PdfDocument(simple_pdf_fixture())
    mapping = document.raw_data
    file_handle = document.file_handle

    assert isinstance(mapping, mmap.mmap)
    assert file_handle is not None
    document.close()

    assert document.closed
    assert mapping.closed
    assert file_handle.closed
    assert document.raw_data == b""
    document.close()


def test_document_close_preserves_caller_owned_reader() -> None:
    reader = BytesIO(simple_pdf_fixture().read_bytes())
    document: PdfDocument[Any] = PdfDocument(reader)

    document.close()

    assert not reader.closed
    assert document.closed


@pytest.mark.parametrize("page_only", [False, True])
def test_close_cancels_extraction_before_releasing_owned_resources(
    monkeypatch: pytest.MonkeyPatch, page_only: bool
) -> None:
    with PublicPdfDocument(simple_pdf_fixture()) as document:
        mapping = document.raw_data
        file_handle = document.file_handle
        assert isinstance(mapping, mmap.mmap)
        assert file_handle is not None

        def close_during_extraction(*args: Any) -> None:
            context = args[1]
            context.raise_if_cancelled()
            document.close()
            assert document.closed
            assert not mapping.closed
            assert not file_handle.closed
            context.raise_if_cancelled()

        entrypoint = "extract_page" if page_only else "extract_document"
        monkeypatch.setattr(f"core_pdf.api.document.{entrypoint}", close_during_extraction)
        with pytest.raises(internal_ExtractionCancelled, match="extraction was cancelled"):
            if page_only:
                document.pages[0].extract()
            else:
                document.extract()

        assert mapping.closed
        assert file_handle.closed


def test_document_close_defers_unmap_for_external_view() -> None:
    document: PdfDocument[Any] = PdfDocument(simple_pdf_fixture())
    mapping = document.raw_data
    assert isinstance(mapping, mmap.mmap)
    external_view = memoryview(mapping)

    document.close()

    assert document.closed
    assert not mapping.closed
    external_view.release()
    mapping.close()
    assert mapping.closed


def test_empty_path_closes_opened_file(tmp_path: Path) -> None:
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    document = object.__new__(PdfDocument)
    document.file_handle = None

    with pytest.raises(PdfSourceError, match="empty"):
        document.load_data(empty_pdf)

    assert document.file_handle is None


class internal_FailingDocument(PdfDocument):
    mapping_at_failure: mmap.mmap | None = None
    handle_at_failure: Any = None

    def scan_xref(self) -> None:
        type(self).mapping_at_failure = cast(mmap.mmap, self.raw_data)
        type(self).handle_at_failure = self.file_handle
        raise RuntimeError("scan failed")


def test_construction_failure_releases_acquired_resources() -> None:
    with pytest.raises(RuntimeError, match="scan failed"):
        internal_FailingDocument(simple_pdf_fixture())

    mapping = internal_FailingDocument.mapping_at_failure
    handle = internal_FailingDocument.handle_at_failure
    assert mapping is not None
    assert mapping.closed
    assert handle is not None
    assert handle.closed

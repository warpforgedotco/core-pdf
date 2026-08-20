from pathlib import Path

from core_pdf import PdfDocument
from core_pdf.impl.engine.spec.s_07_document.document_xref import DocumentXRefMixin
from core_pdf.impl.engine.spec.s_07_syntax.xref import PdfXRefEntry, key_for
from core_pdf.impl.objects import PdfReference


class internal_MetadataRecoveryDocument(DocumentXRefMixin):
    def __init__(self, raw_data: bytes, *, recovered: bool) -> None:
        self.raw_data = raw_data
        self.xref_was_recovered = recovered
        self.inference_calls = 0

    def infer_trailer_metadata(self) -> dict[str, object]:
        self.inference_calls += 1
        return {"Info": PdfReference(7, 0)}


def test_healthy_xref_skips_absent_optional_metadata_scan() -> None:
    document = internal_MetadataRecoveryDocument(b"%PDF-1.7\n/Root 1 0 R", recovered=False)

    trailer = document.merge_recovered_trailer_metadata({"Root": PdfReference(1, 0)})

    assert trailer == {"Root": PdfReference(1, 0)}
    assert document.inference_calls == 0


def test_healthy_xref_scans_when_optional_metadata_marker_exists() -> None:
    document = internal_MetadataRecoveryDocument(
        b"%PDF-1.7\n/Root 1 0 R /Info 7 0 R", recovered=False
    )

    trailer = document.merge_recovered_trailer_metadata({"Root": PdfReference(1, 0)})

    assert trailer["Info"] == PdfReference(7, 0)
    assert document.inference_calls == 1


def test_recovered_xref_keeps_exhaustive_metadata_scan() -> None:
    document = internal_MetadataRecoveryDocument(b"%PDF-1.7\n", recovered=True)

    trailer = document.merge_recovered_trailer_metadata({})

    assert trailer["Info"] == PdfReference(7, 0)
    assert document.inference_calls == 1


def test_repairs_xref_offsets_measured_from_prefixed_pdf_header() -> None:
    prefix = b"producer diagnostics\n"
    body = b"%PDF-1.7\n5 0 obj\n<< /Length 6 0 R >>\nstream\ntext\nendstream\nendobj\n"
    document = internal_MetadataRecoveryDocument(prefix + body, recovered=False)
    relative_offset = body.index(b"5 0 obj")
    document.xref = {key_for(5): PdfXRefEntry(relative_offset)}

    document.repair_stale_xref_offsets()

    assert document.xref[key_for(5)].offset == len(prefix) + relative_offset
    assert document.xref_was_recovered is True


def test_preserves_absolute_xref_offsets_in_prefixed_pdf() -> None:
    prefix = b"producer diagnostics\n"
    body = b"%PDF-1.7\n5 0 obj\nnull\nendobj\n"
    document = internal_MetadataRecoveryDocument(prefix + body, recovered=False)
    absolute_offset = len(prefix) + body.index(b"5 0 obj")
    document.xref = {key_for(5): PdfXRefEntry(absolute_offset)}

    document.repair_stale_xref_offsets()

    assert document.xref[key_for(5)].offset == absolute_offset
    assert document.xref_was_recovered is False


def test_repairs_header_relative_xref_with_accumulated_producer_drift() -> None:
    prefix = b"producer diagnostics\n"
    body = b"%PDF-1.7\n" + b"padding\n" * 40 + b"5 0 obj\nnull\nendobj\n"
    document = internal_MetadataRecoveryDocument(prefix + body, recovered=False)
    actual_offset = len(prefix) + body.index(b"5 0 obj")
    document.xref = {key_for(5): PdfXRefEntry(actual_offset - len(prefix) - 7)}

    document.repair_stale_xref_offsets()

    assert document.xref[key_for(5)].offset == actual_offset
    assert document.xref_was_recovered is True


def test_repaired_xref_keeps_later_objects_with_limited_recovery() -> None:
    pdf_path = Path("tests/fixtures/pdfplumber/tests/pdfs/issue-848.pdf")

    with PdfDocument.open(pdf_path, recovery_scan_all_revisions=False) as document:
        assert document.xref_was_recovered
        assert len(document.pages[0].content_streams) == 1
        assert document.pages[0].get_page_program().products.glyphs

from core_pdf.impl.engine.spec.s_07_document.document_xref import DocumentXRefMixin
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

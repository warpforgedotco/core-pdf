"""Annotation validation and recovery preserve usable records and source identity."""

from typing import Any

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.types import PdfName


def internal_annotation() -> dict[str, Any]:
    return {"Subtype": PdfName(b"Link"), "Rect": [10, 20, 30, 40], "Contents": b"note"}


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("malformation", ["array", "entry", "rectangle"])
def test_annotation_validation_and_recovery(
    text_pdf_bytes: bytes, recover: bool, malformation: str
) -> None:
    with PdfDocument(text_pdf_bytes) as document:
        document.xref_was_recovered = recover
        valid = internal_annotation()
        malformed = internal_annotation()
        malformed["Rect"] = b"invalid"
        annots = (
            valid
            if malformation == "array"
            else [None, valid]
            if malformation == "entry"
            else [malformed, valid]
        )
        document.build_page_dicts()[0]["Annots"] = annots
        page = document.pages[0]
        # Tolerant dictionary discovery accepts a singleton and skips non-dictionaries.
        discovered = page.annotation_dicts()
        assert discovered[-1] is valid
        if recover:
            records = page.get_annotations()
            if malformation == "array":
                assert records == []
            else:
                assert len(records) == 1
                assert records[0].dict is valid
                assert records[0].rect == (10, 20, 30, 40)
                assert records[0].contents == "note"
        else:
            message = {
                "array": "Annots array",
                "entry": "annotation entry",
                "rectangle": "annotation rectangle",
            }[malformation]
            with pytest.raises(ValueError, match=message):
                page.get_annotations()
        assert page.annotation_dicts() == discovered


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("action_kind", ["GoTo", "URI", None])
def test_annotation_destination_precedence_and_action_identity(
    text_pdf_bytes: bytes, direct: bool, action_kind: str | None
) -> None:
    with PdfDocument(text_pdf_bytes) as document:
        annot = internal_annotation()
        action = (
            {"S": PdfName(action_kind.encode()), "D": b"action-target"} if action_kind else None
        )
        annot["A"] = action
        if direct:
            annot["Dest"] = b"direct-target"
        document.build_page_dicts()[0]["Annots"] = [annot]
        (record,) = document.pages[0].get_annotations()
        assert record.dest == (
            b"direct-target" if direct else b"action-target" if action_kind == "GoTo" else None
        )
        assert record.action is action
        assert record.dict is annot
        assert record.subtype == "Link"

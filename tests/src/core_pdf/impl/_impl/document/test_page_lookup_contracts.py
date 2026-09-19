"""Navigation lookup recognizes page identities before recovery signatures."""

import pytest

from core_pdf.impl._impl.document.document import PdfDocument, internal_PageLookup
from core_pdf.impl.types import PdfReference


@pytest.mark.parametrize("mode", ["identity", "copy", "structure", "contents"])
def test_page_lookup_recognizes_supported_aliases(text_pdf_bytes, mode):
    with PdfDocument(text_pdf_bytes) as document:
        original = document.build_page_dicts()[0]
        original["StructParents"] = 7
        if mode == "identity":
            target = original
        elif mode == "copy":
            target = original.copy()
        elif mode == "structure":
            target = {"StructParents": 7}
        else:
            target = {"Contents": original["Contents"]}
        lookup = internal_PageLookup(document)
        assert lookup.page_index_for(target) == 0
        assert lookup.page_index_for(target) == 0
        assert lookup.nodes is lookup.nodes


@pytest.mark.parametrize("target", [None, 0, [], {}, {"Contents": PdfReference(9999, 0)}])
def test_page_lookup_does_not_invent_a_target(text_pdf_bytes, target):
    with PdfDocument(text_pdf_bytes) as document:
        assert document.page_index_for(target) is None

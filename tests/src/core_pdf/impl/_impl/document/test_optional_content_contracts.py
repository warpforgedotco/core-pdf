"""Optional-content defaults, overrides, and reader recovery use real resolution."""

from typing import Any

import pytest

from core_pdf.impl._impl.document.document import PdfDocument
from core_pdf.impl.types import PdfName


@pytest.mark.parametrize("base", [None, "ON", "OFF", "Unchanged"])
@pytest.mark.parametrize("override", [None, "ON", "OFF", "both"])
def test_layer_visibility_applies_overrides_in_order(text_pdf_bytes, base, override):
    with PdfDocument(text_pdf_bytes) as document:
        layer = {"Name": b"Layer"}
        config: dict[str, Any] = {}
        if base is not None:
            config["BaseState"] = PdfName(base.encode())
        if override in {"ON", "both"}:
            config["ON"] = [layer]
        if override in {"OFF", "both"}:
            config["OFF"] = [layer]
        document.catalog()["OCProperties"] = {"OCGs": [layer], "D": config}
        hidden = override in {"OFF", "both"} or (base == "OFF" and override != "ON")
        assert document.oc_hidden_layers() == (frozenset({"Layer"}) if hidden else frozenset())


@pytest.mark.parametrize(
    ("properties", "message"),
    [
        (1, "dictionary"),
        ({"OCGs": 1}, "OCGs array"),
        ({"OCGs": [], "D": 1}, "D dictionary"),
        ({"OCGs": [], "D": {"BaseState": 1}}, "BaseState value"),
        ({"OCGs": [], "D": {"BaseState": PdfName(b"invalid")}}, "BaseState value"),
        ({"OCGs": [], "D": {"ON": [1]}}, "ON entry"),
        ({"OCGs": [], "D": {"OFF": [1]}}, "OFF entry"),
        ({"OCGs": [1]}, "OCG entry"),
        ({"OCGs": [{}]}, "OCG name"),
    ],
)
@pytest.mark.parametrize("recover", [False, True])
def test_malformed_layer_configuration_is_rejected_or_recovered(
    text_pdf_bytes, properties, message, recover
):
    with PdfDocument(text_pdf_bytes) as document:
        document.catalog()["OCProperties"] = properties
        document.xref_was_recovered = recover
        if recover:
            assert document.oc_hidden_layers() == frozenset()
        else:
            with pytest.raises(ValueError, match=message):
                document.oc_hidden_layers()

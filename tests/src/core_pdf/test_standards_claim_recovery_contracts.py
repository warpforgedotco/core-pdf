"""Declaration discovery preserves claims and diagnoses malformed metadata."""

import pytest

from core_pdf.impl._impl.document.standards import (
    internal_discover_extensions,
    internal_profile_claim,
    internal_xmp_claims,
)
from core_pdf_spec.types import PdfName


@pytest.mark.parametrize(
    ("family", "properties", "identifier"),
    [
        ("PDF/A", (("part", "4"), ("rev", "2020")), "pdfa-4"),
        ("PDF/A", (("part", "4"), ("rev", "2099")), None),
        ("PDF/UA", (("part", "2"), ("rev", "2024")), "pdfua-2"),
        ("PDF/UA", (("part", "2"), ("rev", "2099")), None),
        ("PDF/X", (("GTS_PDFXVersion", "PDF/X-4"),), "pdfx-4"),
        ("PDF/VT", (("GTS_PDFVTVersion", "PDF/VT-3"), ("rev", "2020")), "pdfvt-3"),
        ("PDF/VT", (("GTS_PDFVTVersion", "PDF/VT-3"), ("rev", "2099")), None),
        ("PDF/E", (("ISO_PDFEVersion", "PDF/E-1"),), "pdfe-1"),
        ("PDF/E", (("ISO_PDFEVersion", "PDF/E-99"),), None),
        ("Unknown", (), None),
    ],
)
def test_profile_identification_keeps_raw_properties_and_rejects_unknown_revisions(
    family, properties, identifier
):
    diagnostics = []
    claim = internal_profile_claim(family, "test/source", properties, diagnostics)
    assert claim.identifier == identifier
    assert claim.properties == properties
    assert (claim.family, claim.source) == (family, "test/source")
    assert [item.code for item in diagnostics] == ([] if identifier else ["unknown-profile-claim"])


@pytest.mark.parametrize("conflicting", [False, True])
def test_repeated_profile_properties_remain_visible_and_conflicts_disable_identification(
    conflicting,
):
    properties = (("part", "1"), ("part", "2" if conflicting else "1"), ("conformance", "B"))
    diagnostics = []
    claim = internal_profile_claim("PDF/A", "metadata", properties, diagnostics)
    assert claim.properties == properties
    assert claim.identifier == (None if conflicting else "pdfa-1b")
    assert [item.code for item in diagnostics] == (
        ["conflicting-profile-claim"] if conflicting else []
    )


@pytest.mark.parametrize("wrapper", ["rdf", "xmp"])
@pytest.mark.parametrize("prefix", ["id", "arbitrary"])
@pytest.mark.parametrize("attribute", [False, True])
def test_xmp_identification_uses_expanded_namespace_names(wrapper, prefix, attribute):
    declaration = f'{prefix}:part="1" {prefix}:conformance="B"' if attribute else ""
    children = (
        ""
        if attribute
        else f"<{prefix}:part>1</{prefix}:part><{prefix}:conformance>B</{prefix}:conformance>"
    )
    rdf = (
        f"<r:RDF"
        f' xmlns:r="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
        f' xmlns:{prefix}="http://www.aiim.org/pdfa/ns/id/">'
        f"<r:Description {declaration}>{children}</r:Description>"
        f"</r:RDF>"
    )
    xml = rdf if wrapper == "rdf" else f'<x:xmpmeta xmlns:x="adobe:ns:meta/">{rdf}</x:xmpmeta>'
    diagnostics = []
    claims = internal_xmp_claims(xml.encode(), diagnostics)
    assert [claim.identifier for claim in claims] == ["pdfa-1b"]
    assert all(
        key.startswith("{http://www.aiim.org/pdfa/ns/id/}") for key, _ in claims[0].properties
    )
    assert diagnostics == []


@pytest.mark.parametrize(
    "description",
    [
        '<r:Description r:about="other" a:part="1" a:conformance="B"/>',
        '<r:Description r:nodeID="node" a:part="1" a:conformance="B"/>',
        '<r:Bag><r:Description a:part="1" a:conformance="B"/></r:Bag>',
        "<r:Description><r:Bag>"
        '<r:Description a:part="1" a:conformance="B"/>'
        "</r:Bag></r:Description>",
    ],
)
def test_nested_or_other_resource_descriptions_do_not_claim_document_conformance(description):
    xml = (
        f"<r:RDF"
        f' xmlns:r="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
        f' xmlns:a="http://www.aiim.org/pdfa/ns/id/">{description}</r:RDF>'
    )
    assert internal_xmp_claims(xml.encode(), []) == []


@pytest.mark.parametrize(
    "namespace", ["http://pdfa.org/declarations/", "https://pdfa.org/declarations/"]
)
@pytest.mark.parametrize(
    ("target", "identifier", "unknown"),
    [
        ("http://pdfa.org/declarations/wtpdf#reuse1.0-validated", "wtpdf-1.0-reuse", False),
        ("http://pdfa.org/declarations/wtpdf#unknown", None, True),
        ("https://example.com/profile", None, False),
    ],
)
def test_wtpdf_targets_remain_unverified_claims_and_unknown_targets_are_diagnosed(
    namespace, target, identifier, unknown
):
    xml = (
        f"<r:RDF"
        f' xmlns:r="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
        f' xmlns:d="{namespace}">'
        f"<r:Description>"
        f"<d:declarations>"
        f"<d:conformsTo>{target}</d:conformsTo>"
        f"</d:declarations>"
        f"</r:Description>"
        f"</r:RDF>"
    )
    diagnostics = []
    claims = internal_xmp_claims(xml.encode(), diagnostics)
    assert [claim.identifier for claim in claims] == ([identifier] if identifier or unknown else [])
    if claims:
        assert claims[0].properties == ((f"{{{namespace}}}conformsTo", target),)
    assert [item.code for item in diagnostics] == (["unknown-profile-claim"] if unknown else [])


@pytest.mark.parametrize("raw", [42, [], "bad"])
def test_malformed_extension_container_produces_a_diagnostic(raw):
    diagnostics = []
    assert internal_discover_extensions(raw, lambda value: value, diagnostics) == ()
    assert [item.code for item in diagnostics] == ["invalid-extensions"]


@pytest.mark.parametrize(
    "extension_type", [PdfName(b"Extensions"), PdfName(b"Wrong"), "Extensions"]
)
def test_valid_extensions_survive_malformed_neighbors(extension_type):
    diagnostics = []
    value = {
        "Type": extension_type,
        "GOOD": {"BaseVersion": PdfName(b"1.7"), "ExtensionLevel": 3},
        "BAD": [None, {}],
        "EMPTY": [],
    }
    extensions = internal_discover_extensions(value, lambda item: item, diagnostics)
    assert len(extensions) == 1
    assert extensions[0].prefix == "GOOD"
    expected = ["invalid-extension"] * 3
    if not isinstance(extension_type, PdfName) or extension_type.value != "Extensions":
        expected.insert(0, "invalid-extensions-type")
    assert [item.code for item in diagnostics] == expected

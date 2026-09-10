# SPDX-License-Identifier: AGPL-3.0-only
"""Reader security scalar recovery preserves authenticated dictionaries."""

import json
from pathlib import Path
from typing import cast

import pytest

from core_pdf.impl._impl.document.document import PdfDocument
from core_pdf.impl._impl.document.recovery import security
from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.s_07_security.standard import StandardSecurityHandler
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfObject
from core_pdf_spec.types import PdfName, PdfString

FIXTURES = Path(__file__).resolve().parents[4] / "tests/fixtures/security_interop"


@pytest.fixture
def security_values() -> tuple[list[object], PdfDict]:
    with PdfDocument(FIXTURES / "aes-128-r4-cleartext-metadata.pdf", password="user-aes128") as doc:
        params = doc.resolver.resolve_dict(doc.trailer_dict["Encrypt"])
        assert params is not None
        return cast(list[object], doc.trailer_dict["ID"]), params


@pytest.mark.parametrize("field", ["V", "R", "P", "Length", "CF/Length"])
@pytest.mark.parametrize("representation", [str, bytes, bytearray, memoryview])
def test_reader_accepts_legacy_integer_tokens_without_modifying_sources(
    security_values: tuple[list[object], PdfDict], field: str, representation: type
) -> None:
    document_id, params = security_values
    target = cast(PdfDict, cast(PdfDict, params["CF"])["StdCF"]) if field == "CF/Length" else params
    key = "Length" if field == "CF/Length" else field
    value = target[key]
    encoded = str(value).encode("ascii")
    replacement = str(value) if representation is str else representation(encoded)
    # Inject a legacy token outside the strict PDF object type for reader recovery.
    target[key] = cast(PdfObject, replacement)
    handler = security.create_recovered_security_handler(document_id, params, "user-aes128")
    assert handler.config.version == 4
    assert target[key] is replacement


@pytest.mark.parametrize("value", [True, 4.0, " 4 ", "4_0", "٤", "four", PdfString(b"4"), None])
def test_reader_does_not_broaden_security_integer_recovery(
    security_values: tuple[list[object], PdfDict], value: object
) -> None:
    document_id, params = security_values
    params["V"] = cast(int, value)
    with pytest.raises(PdfUnsupportedError, match="^Invalid encryption dictionary$"):
        security.create_recovered_security_handler(document_id, params, "user-aes128")


def test_reader_reuses_valid_dictionary_and_copies_only_changed_branches(
    security_values: tuple[list[object], PdfDict], monkeypatch: pytest.MonkeyPatch
) -> None:
    document_id, params = security_values
    received: list[PdfDict] = []
    original_factory = security.create_standard_security_handler

    def capture(ids: object, values: PdfDict, password: str) -> StandardSecurityHandler:
        received.append(values)
        return original_factory(cast(list[object], ids), values, password)

    monkeypatch.setattr(security, "create_standard_security_handler", capture)
    security.create_recovered_security_handler(document_id, params, "user-aes128")
    assert received[-1] is params
    filters = cast(PdfDict, params["CF"])
    standard = cast(PdfDict, filters["StdCF"])
    standard["Length"] = b"16"
    ignored: PdfDict = {"Length": PdfString(b"not a number")}
    filters["Identity"] = ignored
    security.create_recovered_security_handler(document_id, params, "user-aes128")
    normalized = received[-1]
    normalized_filters = cast(PdfDict, normalized["CF"])
    assert normalized is not params
    assert normalized_filters is not filters
    assert normalized_filters["StdCF"] is not standard
    assert normalized_filters["Identity"] is ignored
    assert normalized["O"] is params["O"]
    assert standard["Length"] == b"16"


def test_reader_security_names_recover_host_slashes_but_not_decoded_name_slashes(
    security_values: tuple[list[object], PdfDict],
) -> None:
    document_id, params = security_values
    params["Filter"] = b"/Standard"
    assert (
        security.create_recovered_security_handler(
            document_id, params, "user-aes128"
        ).config.version
        == 4
    )
    assert params["Filter"] == b"/Standard"
    params["Filter"] = PdfName.of("/Standard")
    with pytest.raises(PdfUnsupportedError, match="Unsupported encryption filter: /Standard"):
        security.create_recovered_security_handler(document_id, params, "user-aes128")


@pytest.mark.parametrize(
    "manifest", ["manifest.json", "aes_gcm/manifest.json", "pdf_mac/manifest.json"]
)
def test_security_interoperability_documents_still_authenticate(manifest: str) -> None:
    path = FIXTURES / manifest
    data = json.loads(path.read_text())
    records = data["fixtures"] if "fixtures" in data else [data["fixture"]]
    for record in records:
        with PdfDocument(path.parent / record["filename"], password=record["user_password"]) as doc:
            assert doc.decipher is not None
            assert doc.page_count() == 1

# SPDX-License-Identifier: AGPL-3.0-only


import pytest

from core_pdf_spec.exceptions import PdfDecryptionError, PdfUnsupportedError
from core_pdf_spec.s_07_security import document as security_document
from core_pdf_spec.s_07_security.standard import (
    StandardSecurityHandler,
    create_standard_security_handler,
)
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.types import PdfName


def security_dictionary() -> PdfDict:
    return {
        "Filter": PdfName.of("Standard"),
        "V": 4,
        "R": 4,
        "P": -4,
        "Length": 128,
        "O": bytes.fromhex("fb2d1f37f7759b388730604ff5ce8e7c9aa8f97defc007bc15f3bf9b134734df"),
        "U": bytes.fromhex("6800576c738a44382046b7847b5a78ae0021446990b9e4114071a4d9104984c1"),
        "EncryptMetadata": False,
        "CF": {"StdCF": {"CFM": PdfName.of("AESV2"), "Length": 16}},
        "StmF": PdfName.of("StdCF"),
        "StrF": PdfName.of("StdCF"),
    }


@pytest.mark.parametrize("field", ["V", "R", "P", "Length", "CF/Length"])
@pytest.mark.parametrize("representation", [str, bytes, bytearray, memoryview, float, bool])
def test_security_integer_fields_reject_noninteger_representations(
    field: str, representation: type
) -> None:
    params = security_dictionary()
    target = params["CF"]["StdCF"] if field == "CF/Length" else params
    key = "Length" if field == "CF/Length" else field
    value = target[key]
    encoded = str(value).encode("ascii")
    replacement = (
        representation(encoded)
        if representation in (bytes, bytearray, memoryview)
        else representation(value)
    )
    target[key] = replacement
    with pytest.raises(PdfUnsupportedError, match="^Invalid encryption dictionary$"):
        create_standard_security_handler([b"core-pdf-security"], params, "user-aes128")


def test_real_integer_security_dictionary_authenticates_and_keeps_defaults() -> None:
    params = security_dictionary()
    handler = create_standard_security_handler([b"core-pdf-security"], params, "user-aes128")
    assert (handler.config.version, handler.config.revision, handler.config.length_bits) == (
        4,
        4,
        128,
    )
    assert handler.config.permissions == 0xFFFFFFFC
    assert len(handler.file_key) == 16
    del params["Length"]
    del params["CF"]["StdCF"]["Length"]
    assert (
        create_standard_security_handler([b"core-pdf-security"], params, "user-aes128").file_key
        == handler.file_key
    )
    with pytest.raises(PdfUnsupportedError, match="^Incorrect password$"):
        create_standard_security_handler([b"core-pdf-security"], params, "incorrect")


@pytest.mark.parametrize("field", ["V", "R", "P", "Length", "CF/Length"])
def test_explicit_null_security_integer_is_not_a_default(field: str) -> None:
    params = security_dictionary()
    target = params["CF"]["StdCF"] if field == "CF/Length" else params
    target["Length" if field == "CF/Length" else field] = None
    with pytest.raises(PdfUnsupportedError, match="^Invalid encryption dictionary$"):
        create_standard_security_handler([b"core-pdf-security"], params, "user-aes128")


def test_handler_factory_runs_after_id_validation() -> None:
    trailer: PdfDict = {"Encrypt": security_dictionary(), "ID": []}
    resolver = ObjectResolver(b"", {})

    def unexpected_factory(*args: object) -> StandardSecurityHandler:
        raise AssertionError("authentication preceded ID validation")

    with pytest.raises(PdfUnsupportedError, match="Invalid trailer ID array"):
        security_document.initialize_document_security(
            b"", trailer, resolver, "user-aes128", handler_factory=unexpected_factory
        )
    assert resolver.decipher is None


def test_mac_failure_after_factory_never_installs_the_decipher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trailer: PdfDict = {"Encrypt": security_dictionary(), "ID": [b"core-pdf-security"]}
    resolver = ObjectResolver(b"", {})
    events: list[str] = []

    def factory(document_id: object, params: PdfDict, password: str) -> StandardSecurityHandler:
        assert resolver.decipher is None
        events.append("authenticate")
        return create_standard_security_handler(document_id, params, password)  # ty: ignore[invalid-argument-type]

    def failed_mac(*args: object) -> bool:
        assert resolver.decipher is None
        events.append("mac")
        raise PdfDecryptionError("Invalid PDF MAC")

    monkeypatch.setattr(security_document, "validate_pdf_mac_if_present", failed_mac)
    with pytest.raises(PdfDecryptionError, match="Invalid PDF MAC"):
        security_document.initialize_document_security(
            b"", trailer, resolver, "user-aes128", handler_factory=factory
        )
    assert events == ["authenticate", "mac"]
    assert resolver.decipher is None

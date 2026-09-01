from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

import pytest
from asn1crypto import algos, cms, core
from cryptography.hazmat.primitives import hashes

from core_pdf import PdfDocument
from core_pdf.impl.exceptions import PdfDecryptionError, PdfUnsupportedError
from core_pdf.impl.primitives import PdfName, PdfReference, PdfString
from core_pdf.impl.spec.s_07_document import document as document_module
from core_pdf.impl.spec.s_07_security.pdf_mac import (
    internal_CMS_ALGORITHM_PROTECTION_ATTRIBUTE_OID,
    internal_CONTENT_TYPE_ATTRIBUTE_OID,
    internal_digest_algorithm,
    internal_digest_byte_range,
    internal_encapsulated_content,
    internal_extract_standalone_token,
    internal_parse_der,
    internal_PDF_MAC_INTEGRITY_INFO_OID,
    internal_PdfMacIntegrityInfo,
    internal_SHA3_256_OID,
    internal_SHA3_384_OID,
    internal_SHA3_512_OID,
    internal_SHA256_OID,
    internal_SHA384_OID,
    internal_SHA512_OID,
    internal_validate_authenticated_attributes,
    internal_validate_authenticated_data,
    internal_validate_integrity_info,
    internal_validate_mac_algorithm,
    validate_pdf_mac_extension,
    validate_pdf_mac_if_present,
)
from core_pdf.impl.spec.s_07_security.standard import internal_StandardSecurityHandler
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.types import PdfByteBuffer
from tests.helpers.paths import FIXTURES

internal_FIXTURE = FIXTURES / "security_interop" / "pdf_mac" / "aes-256-r6-cbc-mac.pdf"
internal_PASSWORD = "user-mac-cbc"


@dataclass(frozen=True, slots=True)
class internal_PdfMacMaterial:
    raw_data: bytes
    trailer: PdfDict
    handler: internal_StandardSecurityHandler


@pytest.fixture()
def pdf_mac_material(monkeypatch: pytest.MonkeyPatch) -> internal_PdfMacMaterial:
    captured: list[internal_PdfMacMaterial] = []

    def internal_capture(
        raw_data: PdfByteBuffer,
        trailer: PdfDict,
        handler: internal_StandardSecurityHandler,
    ) -> None:
        captured.append(internal_PdfMacMaterial(bytes(raw_data), trailer, handler))

    # Capture the already-authenticated file key at the document integration
    # boundary. Individual tests then exercise the ISO/TS 32004:2024 validator
    # directly without reproducing the Standard handler's password algorithm.
    monkeypatch.setattr(document_module, "validate_pdf_mac_if_present", internal_capture)
    with PdfDocument.open(internal_FIXTURE, password=internal_PASSWORD):
        pass
    assert len(captured) == 1
    return captured[0]


def internal_auth_code(material: internal_PdfMacMaterial) -> PdfDict:
    value = material.trailer["AuthCode"]
    assert isinstance(value, dict)
    return cast(PdfDict, value)


def internal_auth_data(
    material: internal_PdfMacMaterial,
) -> tuple[tuple[int, int, int, int], cms.AuthenticatedData]:
    byte_range, token = internal_extract_standalone_token(
        material.raw_data,
        internal_auth_code(material),
    )
    content_info = internal_parse_der(token, cms.ContentInfo)
    auth_data = content_info["content"]
    assert isinstance(auth_data, cms.AuthenticatedData)
    return byte_range, auth_data


def internal_validate_auth_data(
    material: internal_PdfMacMaterial,
    byte_range: tuple[int, int, int, int],
    auth_data: cms.AuthenticatedData,
) -> None:
    kdf_salt = material.handler.config.kdf_salt
    assert kdf_salt is not None
    internal_validate_authenticated_data(
        material.raw_data,
        byte_range,
        auth_data,
        material.handler.file_key,
        kdf_salt,
    )


def internal_copy_attributes(attributes: cms.CMSAttributes) -> cms.CMSAttributes:
    return cms.CMSAttributes.load(attributes.untag().dump(force=True), strict=True)


def test_pdf_mac_validator_accepts_pristine_fixture(
    pdf_mac_material: internal_PdfMacMaterial,
) -> None:
    assert validate_pdf_mac_if_present(
        pdf_mac_material.raw_data,
        pdf_mac_material.trailer,
        pdf_mac_material.handler,
    )


def test_pdf_mac_requires_exact_iso_32004_extension_declaration() -> None:
    declaration: PdfDict = {
        "Type": PdfName.of("DeveloperExtensions"),
        "BaseVersion": PdfName.of("2.0"),
        "ExtensionLevel": 32004,
        "ExtensionRevision": PdfString(b":2024"),
        "URL": PdfString(b"https://www.iso.org/standard/45877.html"),
    }

    validate_pdf_mac_extension([declaration])
    for key in declaration:
        invalid = dict(declaration)
        invalid.pop(key)
        with pytest.raises(PdfDecryptionError, match="extension declaration"):
            validate_pdf_mac_extension([invalid])


def test_pdf_mac_payload_bytes_do_not_depend_on_global_oid_registration(
    pdf_mac_material: internal_PdfMacMaterial,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered_name = "core_pdf_test_pdf_mac_integrity_info"
    monkeypatch.setitem(
        cms.ContentType._map,
        internal_PDF_MAC_INTEGRITY_INFO_OID,
        registered_name,
    )
    raw_reverse_map = cms.ContentType._reverse_map
    if raw_reverse_map is None:
        reverse_map = {name: oid for oid, name in cms.ContentType._map.items()}
        monkeypatch.setattr(cms.ContentType, "_reverse_map", reverse_map)
    else:
        reverse_map = cast(dict[str, str], raw_reverse_map)
    monkeypatch.setitem(
        reverse_map,
        registered_name,
        internal_PDF_MAC_INTEGRITY_INFO_OID,
    )
    monkeypatch.setitem(
        cms.EncapsulatedContentInfo._oid_specs,
        registered_name,
        internal_PdfMacIntegrityInfo,
    )

    byte_range, auth_data = internal_auth_data(pdf_mac_material)
    assert isinstance(
        auth_data["encap_content_info"]["content"].parsed,
        internal_PdfMacIntegrityInfo,
    )
    internal_validate_auth_data(pdf_mac_material, byte_range, auth_data)


@pytest.mark.parametrize("missing_signal", ["auth-code", "kdf-salt"])
def test_pdf_mac_required_signal_cannot_be_removed(
    pdf_mac_material: internal_PdfMacMaterial,
    missing_signal: str,
) -> None:
    trailer = dict(pdf_mac_material.trailer)
    handler = pdf_mac_material.handler
    if missing_signal == "auth-code":
        trailer.pop("AuthCode")
    else:
        handler = replace(
            handler,
            config=replace(handler.config, kdf_salt=None),
        )

    with pytest.raises(PdfDecryptionError, match="Invalid PDF MAC"):
        validate_pdf_mac_if_present(pdf_mac_material.raw_data, trailer, handler)


def test_pdf_mac_auth_code_must_be_a_direct_dictionary(
    pdf_mac_material: internal_PdfMacMaterial,
) -> None:
    trailer = dict(pdf_mac_material.trailer)
    trailer["AuthCode"] = PdfReference(9)

    with pytest.raises(PdfDecryptionError, match="Invalid PDF MAC"):
        validate_pdf_mac_if_present(
            pdf_mac_material.raw_data,
            trailer,
            pdf_mac_material.handler,
        )


@pytest.mark.parametrize(
    ("location", "exception", "message"),
    [
        (PdfName.of("AttachedToSig"), PdfUnsupportedError, "Attached-to-signature"),
        (PdfName.of("VendorLocation"), PdfUnsupportedError, "Unsupported PDF MAC location"),
        ("Standalone", PdfDecryptionError, "Invalid PDF MAC"),
    ],
)
def test_pdf_mac_location_is_direct_and_supported(
    pdf_mac_material: internal_PdfMacMaterial,
    location: object,
    exception: type[Exception],
    message: str,
) -> None:
    # Deliberately holds a value the format does not allow, so it is not a
    # well-typed PdfDict at this point -- that is what the test is checking.
    auth_code: dict[Any, Any] = dict(internal_auth_code(pdf_mac_material))
    auth_code["MACLocation"] = location
    trailer: dict[Any, Any] = dict(pdf_mac_material.trailer)
    trailer["AuthCode"] = auth_code

    with pytest.raises(exception, match=message):
        validate_pdf_mac_if_present(
            pdf_mac_material.raw_data,
            trailer,
            pdf_mac_material.handler,
        )


def test_standalone_pdf_mac_forbids_signature_reference(
    pdf_mac_material: internal_PdfMacMaterial,
) -> None:
    auth_code = dict(internal_auth_code(pdf_mac_material))
    auth_code["SigObjRef"] = PdfReference(9)
    trailer = dict(pdf_mac_material.trailer)
    trailer["AuthCode"] = auth_code

    with pytest.raises(PdfDecryptionError, match="Invalid PDF MAC"):
        validate_pdf_mac_if_present(
            pdf_mac_material.raw_data,
            trailer,
            pdf_mac_material.handler,
        )


@pytest.mark.parametrize(
    "invalid_range",
    [
        [0, 1, 2],
        [False, 1, 2, 3],
        [1, 10, 20, 30],
        [0, 20, 20, 30],
        [0, 10, PdfReference(2), 30],
    ],
)
def test_pdf_mac_rejects_invalid_byte_range_shape(
    pdf_mac_material: internal_PdfMacMaterial,
    invalid_range: list[object],
) -> None:
    auth_code = dict(internal_auth_code(pdf_mac_material))
    auth_code["ByteRange"] = invalid_range
    trailer = dict(pdf_mac_material.trailer)
    trailer["AuthCode"] = auth_code

    with pytest.raises(PdfDecryptionError, match="Invalid PDF MAC"):
        validate_pdf_mac_if_present(
            pdf_mac_material.raw_data,
            trailer,
            pdf_mac_material.handler,
        )


@pytest.mark.parametrize(
    "invalid_mac",
    [
        b"not a PDF string",
        PdfString(b"not hexadecimal", is_literal=True),
        PdfReference(7),
    ],
)
def test_standalone_pdf_mac_requires_direct_hexadecimal_token(
    pdf_mac_material: internal_PdfMacMaterial,
    invalid_mac: object,
) -> None:
    auth_code: dict[Any, Any] = dict(internal_auth_code(pdf_mac_material))
    auth_code["MAC"] = invalid_mac
    trailer: dict[Any, Any] = dict(pdf_mac_material.trailer)
    trailer["AuthCode"] = auth_code

    with pytest.raises(PdfDecryptionError, match="Invalid PDF MAC"):
        validate_pdf_mac_if_present(
            pdf_mac_material.raw_data,
            trailer,
            pdf_mac_material.handler,
        )


def test_standalone_pdf_mac_rejects_whitespace_inside_hex_token(
    pdf_mac_material: internal_PdfMacMaterial,
) -> None:
    auth_code = internal_auth_code(pdf_mac_material)
    byte_range = cast(list[int], auth_code["ByteRange"])
    corrupted = bytearray(pdf_mac_material.raw_data)
    corrupted[byte_range[1] + 1] = ord(" ")

    with pytest.raises(PdfDecryptionError, match="Invalid PDF MAC"):
        validate_pdf_mac_if_present(
            bytes(corrupted),
            pdf_mac_material.trailer,
            pdf_mac_material.handler,
        )


def test_pdf_mac_der_parser_rejects_trailing_data(
    pdf_mac_material: internal_PdfMacMaterial,
) -> None:
    _, token = internal_extract_standalone_token(
        pdf_mac_material.raw_data,
        internal_auth_code(pdf_mac_material),
    )

    with pytest.raises(ValueError):
        internal_parse_der(token + b"\x00", cms.ContentInfo)


@pytest.mark.parametrize(
    "defect",
    ["version", "originator-info", "unauth-attrs", "mac-algorithm"],
)
def test_pdf_mac_rejects_invalid_authenticated_data_structure(
    pdf_mac_material: internal_PdfMacMaterial,
    defect: str,
) -> None:
    byte_range, auth_data = internal_auth_data(pdf_mac_material)
    if defect == "version":
        auth_data["version"] = 1
    elif defect == "originator-info":
        auth_data["originator_info"] = cms.OriginatorInfo({"certs": []})
    elif defect == "unauth-attrs":
        auth_data["unauth_attrs"] = cms.CMSAttributes([])
    else:
        auth_data["mac_algorithm"] = algos.HmacAlgorithm({"algorithm": "sha1"})

    with pytest.raises(ValueError):
        internal_validate_auth_data(pdf_mac_material, byte_range, auth_data)


def test_pdf_mac_rejects_unsupported_digest_algorithm(
    pdf_mac_material: internal_PdfMacMaterial,
) -> None:
    byte_range, auth_data = internal_auth_data(pdf_mac_material)
    auth_data["digest_algorithm"] = algos.DigestAlgorithm({"algorithm": "md5"})

    with pytest.raises(PdfUnsupportedError, match="Unsupported PDF MAC digest algorithm"):
        internal_validate_auth_data(pdf_mac_material, byte_range, auth_data)


@pytest.mark.parametrize(
    ("oid", "algorithm_type"),
    [
        (internal_SHA256_OID, hashes.SHA256),
        (internal_SHA384_OID, hashes.SHA384),
        (internal_SHA512_OID, hashes.SHA512),
        (internal_SHA3_256_OID, hashes.SHA3_256),
        (internal_SHA3_384_OID, hashes.SHA3_384),
        (internal_SHA3_512_OID, hashes.SHA3_512),
    ],
)
def test_pdf_mac_supports_every_iso_ts_32004_digest(
    oid: str,
    algorithm_type: type[hashes.HashAlgorithm],
) -> None:
    identifier = algos.DigestAlgorithm({"algorithm": oid})

    assert isinstance(internal_digest_algorithm(identifier), algorithm_type)


def test_pdf_mac_sha3_algorithm_parameters_must_be_absent() -> None:
    identifier = algos.DigestAlgorithm(
        {
            "algorithm": internal_SHA3_256_OID,
            "parameters": core.Null(),
        }
    )

    with pytest.raises(ValueError, match="digest algorithm parameters"):
        internal_digest_algorithm(identifier)


@pytest.mark.parametrize("parameters", [None, core.Null()])
def test_pdf_mac_hmac_sha256_accepts_rfc_4231_parameter_encodings(
    parameters: core.Null | None,
) -> None:
    values: dict[str, object] = {"algorithm": "1.2.840.113549.2.9"}
    if parameters is not None:
        values["parameters"] = parameters

    internal_validate_mac_algorithm(algos.HmacAlgorithm(values))


def test_pdf_mac_hmac_sha256_rejects_other_parameters() -> None:
    identifier = algos.HmacAlgorithm(
        {
            "algorithm": "1.2.840.113549.2.9",
            "parameters": core.OctetString(b"parameters"),
        }
    )

    with pytest.raises(ValueError, match="MAC algorithm parameters"):
        internal_validate_mac_algorithm(identifier)


@pytest.mark.parametrize("defect", ["recipient-count", "kdf", "key-wrap"])
def test_pdf_mac_rejects_invalid_recipient_keying_structure(
    pdf_mac_material: internal_PdfMacMaterial,
    defect: str,
) -> None:
    byte_range, auth_data = internal_auth_data(pdf_mac_material)
    if defect == "recipient-count":
        auth_data["recipient_infos"] = cms.RecipientInfos([])
    else:
        password_info = auth_data["recipient_infos"][0].chosen
        assert isinstance(password_info, cms.PasswordRecipientInfo)
        if defect == "kdf":
            password_info["key_derivation_algorithm"] = algos.KdfAlgorithm({"algorithm": "1.2.3.4"})
        else:
            password_info["key_encryption_algorithm"] = cms.KeyEncryptionAlgorithm(
                {"algorithm": "aes128_wrap"}
            )

    with pytest.raises(ValueError):
        internal_validate_auth_data(pdf_mac_material, byte_range, auth_data)


def test_pdf_mac_authenticated_attributes_match_payload_and_algorithms(
    pdf_mac_material: internal_PdfMacMaterial,
) -> None:
    _, auth_data = internal_auth_data(pdf_mac_material)
    attributes = auth_data["auth_attrs"]
    assert isinstance(attributes, cms.CMSAttributes)
    encapsulated_content = internal_encapsulated_content(auth_data)
    digest_algorithm = internal_digest_algorithm(auth_data["digest_algorithm"])

    internal_validate_authenticated_attributes(
        attributes,
        auth_data,
        encapsulated_content,
        digest_algorithm,
    )

    without_content_type = cms.CMSAttributes(
        [
            cms.CMSAttribute.load(attribute.dump(force=True), strict=True)
            for attribute in attributes
            if attribute["type"].dotted != internal_CONTENT_TYPE_ATTRIBUTE_OID
        ]
    )
    with pytest.raises(ValueError, match="must occur exactly once"):
        internal_validate_authenticated_attributes(
            without_content_type,
            auth_data,
            encapsulated_content,
            digest_algorithm,
        )

    # ISO/TS 32004:2024, 6.3.6.4 only recommends this attribute. Tokens
    # without it remain valid; when present, its RFC 6211:2011 rules apply.
    without_algorithm_protection = cms.CMSAttributes(
        [
            cms.CMSAttribute.load(attribute.dump(force=True), strict=True)
            for attribute in attributes
            if attribute["type"].dotted != internal_CMS_ALGORITHM_PROTECTION_ATTRIBUTE_OID
        ]
    )
    internal_validate_authenticated_attributes(
        without_algorithm_protection,
        auth_data,
        encapsulated_content,
        digest_algorithm,
    )


def test_pdf_mac_rejects_incorrect_authenticated_message_digest(
    pdf_mac_material: internal_PdfMacMaterial,
) -> None:
    _, auth_data = internal_auth_data(pdf_mac_material)
    attributes = auth_data["auth_attrs"]
    assert isinstance(attributes, cms.CMSAttributes)
    attributes = internal_copy_attributes(attributes)
    message_digest = next(
        attribute for attribute in attributes if attribute["type"].native == "message_digest"
    )
    message_digest["values"][0] = core.OctetString(b"\x00" * 32)

    with pytest.raises(ValueError, match="incorrect PDF MAC message-digest"):
        internal_validate_authenticated_attributes(
            attributes,
            auth_data,
            internal_encapsulated_content(auth_data),
            internal_digest_algorithm(auth_data["digest_algorithm"]),
        )


def test_pdf_mac_rejects_unprotected_algorithm_change(
    pdf_mac_material: internal_PdfMacMaterial,
) -> None:
    _, auth_data = internal_auth_data(pdf_mac_material)
    attributes = auth_data["auth_attrs"]
    assert isinstance(attributes, cms.CMSAttributes)
    attributes = internal_copy_attributes(attributes)
    protection_attribute = next(
        attribute
        for attribute in attributes
        if attribute["type"].native == "cms_algorithm_protection"
    )
    protection = protection_attribute["values"][0]
    assert isinstance(protection, cms.CMSAlgorithmProtection)
    protection["digest_algorithm"] = algos.DigestAlgorithm({"algorithm": "sha384"})

    with pytest.raises(ValueError, match="digest algorithm is not protected"):
        internal_validate_authenticated_attributes(
            attributes,
            auth_data,
            internal_encapsulated_content(auth_data),
            internal_digest_algorithm(auth_data["digest_algorithm"]),
        )


def test_standalone_pdf_mac_integrity_info_binds_document_digest(
    pdf_mac_material: internal_PdfMacMaterial,
) -> None:
    byte_range, auth_data = internal_auth_data(pdf_mac_material)
    integrity_info = internal_parse_der(
        internal_encapsulated_content(auth_data),
        internal_PdfMacIntegrityInfo,
    )
    digest_algorithm = internal_digest_algorithm(auth_data["digest_algorithm"])
    document_digest = internal_digest_byte_range(
        pdf_mac_material.raw_data,
        byte_range,
        digest_algorithm,
    )

    internal_validate_integrity_info(integrity_info, document_digest)
    with pytest.raises(ValueError, match="document digest does not match"):
        internal_validate_integrity_info(integrity_info, b"\x00" * len(document_digest))

    integrity_info["signature_digest"] = core.OctetString(b"\x00" * len(document_digest))
    with pytest.raises(ValueError, match="cannot contain a signature digest"):
        internal_validate_integrity_info(integrity_info, document_digest)

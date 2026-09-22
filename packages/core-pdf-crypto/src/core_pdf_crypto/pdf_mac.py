# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Buffer
from typing import Any, cast

from asn1crypto import cms, core
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, hmac, keywrap
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from core_pdf_crypto.errors import UnsupportedAlgorithmError

AUTHENTICATED_DATA_OID = "1.2.840.113549.1.9.16.1.2"
CONTENT_TYPE_ATTRIBUTE_OID = "1.2.840.113549.1.9.3"
MESSAGE_DIGEST_ATTRIBUTE_OID = "1.2.840.113549.1.9.4"
CMS_ALGORITHM_PROTECTION_ATTRIBUTE_OID = "1.2.840.113549.1.9.52"
PDF_MAC_INTEGRITY_INFO_OID = "1.0.32004.1.0"
PDF_MAC_WRAP_KDF_OID = "1.0.32004.1.1"
AES_256_KEY_WRAP_OID = "2.16.840.1.101.3.4.1.45"
HMAC_SHA256_OID = "1.2.840.113549.2.9"
SHA256_OID = "2.16.840.1.101.3.4.2.1"
SHA384_OID = "2.16.840.1.101.3.4.2.2"
SHA512_OID = "2.16.840.1.101.3.4.2.3"
SHA3_256_OID = "2.16.840.1.101.3.4.2.8"
SHA3_384_OID = "2.16.840.1.101.3.4.2.9"
SHA3_512_OID = "2.16.840.1.101.3.4.2.10"
PDF_MAC_KDF_SALT_BYTES = 32
PDF_MAC_KEK_BYTES = 32
PDF_MAC_KEY_BYTES = 32
PDF_MAC_WRAPPED_KEY_BYTES = 40
PDF_MAC_HKDF_INFO = b"PDFMAC"


class PdfMacIntegrityInfo(core.Sequence):
    _fields = [
        ("version", core.Integer),
        ("data_digest", core.OctetString),
        (
            "signature_digest",
            core.OctetString,
            {"implicit": 0, "optional": True},
        ),
    ]


def validate_pdf_mac_token(
    raw_data: Buffer,
    byte_range: tuple[int, int, int, int],
    token: bytes,
    file_key: bytes,
    kdf_salt: bytes,
) -> None:
    content_info = parse_der(token, cms.ContentInfo)
    if content_info["content_type"].dotted != AUTHENTICATED_DATA_OID:
        raise ValueError("PDF MAC token is not CMS AuthenticatedData")
    auth_data = content_info["content"]
    if not isinstance(auth_data, cms.AuthenticatedData):
        raise ValueError("invalid CMS AuthenticatedData")
    validate_authenticated_data(raw_data, byte_range, auth_data, file_key, kdf_salt)


def validate_authenticated_data(
    raw_data: Buffer,
    byte_range: tuple[int, int, int, int],
    auth_data: cms.AuthenticatedData,
    file_key: bytes,
    kdf_salt: bytes,
) -> None:
    if auth_data["version"].native != "v0":
        raise ValueError("invalid PDF MAC AuthenticatedData version")
    if not isinstance(auth_data["originator_info"], core.Void):
        raise ValueError("PDF MAC cannot contain originator information")
    if not isinstance(auth_data["unauth_attrs"], core.Void):
        raise ValueError("PDF MAC cannot contain unauthenticated attributes")

    hash_algorithm = digest_algorithm(auth_data["digest_algorithm"])
    validate_mac_algorithm(auth_data["mac_algorithm"])
    encapsulated_content = internal_encapsulated_content(auth_data)
    integrity_info = parse_der(encapsulated_content, PdfMacIntegrityInfo)

    mac_key = unwrap_mac_key(
        auth_data["recipient_infos"],
        file_key,
        kdf_salt,
    )
    auth_attrs = auth_data["auth_attrs"]
    if not isinstance(auth_attrs, cms.CMSAttributes):
        raise ValueError("PDF MAC authenticated attributes are missing")
    received_mac = auth_data["mac"].native
    if not isinstance(received_mac, bytes) or len(received_mac) != PDF_MAC_KEY_BYTES:
        raise ValueError("invalid PDF MAC value")

    verifier = hmac.HMAC(mac_key, hashes.SHA256())
    verifier.update(auth_attrs.untag().dump(force=True))
    try:
        verifier.verify(received_mac)
    except InvalidSignature as exc:
        raise ValueError("PDF MAC verification failed") from exc

    validate_authenticated_attributes(
        auth_attrs,
        auth_data,
        encapsulated_content,
        hash_algorithm,
    )
    validate_integrity_info(
        integrity_info,
        digest_byte_range(raw_data, byte_range, hash_algorithm),
    )


def unwrap_mac_key(
    recipient_infos: cms.RecipientInfos,
    file_key: bytes,
    kdf_salt: bytes,
) -> bytes:
    if len(recipient_infos) != 1 or recipient_infos[0].name != "pwri":
        raise ValueError("PDF MAC requires one PasswordRecipientInfo")
    password_info = recipient_infos[0].chosen
    if not isinstance(password_info, cms.PasswordRecipientInfo):
        raise ValueError("invalid PDF MAC PasswordRecipientInfo")
    if password_info["version"].native != "v0":
        raise ValueError("invalid PDF MAC PasswordRecipientInfo version")

    kdf = password_info["key_derivation_algorithm"]
    if isinstance(kdf, core.Void):
        raise ValueError("PDF MAC key derivation algorithm is missing")
    require_algorithm(
        kdf,
        PDF_MAC_WRAP_KDF_OID,
        require_absent_parameters=True,
    )
    require_algorithm(
        password_info["key_encryption_algorithm"],
        AES_256_KEY_WRAP_OID,
        require_absent_parameters=True,
    )

    encrypted_key = password_info["encrypted_key"].native
    if not isinstance(encrypted_key, bytes) or len(encrypted_key) != PDF_MAC_WRAPPED_KEY_BYTES:
        raise ValueError("invalid wrapped PDF MAC key")
    if len(kdf_salt) != PDF_MAC_KDF_SALT_BYTES:
        raise ValueError("invalid PDF MAC KDFSalt")

    key_encryption_key = HKDF(
        algorithm=hashes.SHA256(),
        length=PDF_MAC_KEK_BYTES,
        salt=kdf_salt,
        info=PDF_MAC_HKDF_INFO,
    ).derive(file_key)
    try:
        mac_key = keywrap.aes_key_unwrap(key_encryption_key, encrypted_key)
    except keywrap.InvalidUnwrap as exc:
        raise ValueError("PDF MAC key unwrap failed") from exc
    if len(mac_key) != PDF_MAC_KEY_BYTES:
        raise ValueError("invalid unwrapped PDF MAC key")
    return mac_key


def validate_authenticated_attributes(
    attributes: cms.CMSAttributes,
    auth_data: cms.AuthenticatedData,
    encapsulated_content: bytes,
    hash_algorithm: hashes.HashAlgorithm,
) -> None:
    content_type = unique_attribute(attributes, CONTENT_TYPE_ATTRIBUTE_OID)
    if not isinstance(content_type, cms.ContentType):
        raise ValueError("invalid PDF MAC content-type attribute")
    if content_type.dotted != PDF_MAC_INTEGRITY_INFO_OID:
        raise ValueError("incorrect PDF MAC content-type attribute")

    message_digest = unique_attribute(attributes, MESSAGE_DIGEST_ATTRIBUTE_OID)
    if not isinstance(message_digest, core.OctetString):
        raise ValueError("invalid PDF MAC message-digest attribute")
    if message_digest.native != digest(encapsulated_content, hash_algorithm):
        raise ValueError("incorrect PDF MAC message-digest attribute")

    algorithm_protection = unique_attribute(
        attributes,
        CMS_ALGORITHM_PROTECTION_ATTRIBUTE_OID,
        required=False,
    )
    if algorithm_protection is None:
        return
    if not isinstance(algorithm_protection, cms.CMSAlgorithmProtection):
        raise ValueError("invalid PDF MAC algorithm-protection attribute")
    if not isinstance(algorithm_protection["signature_algorithm"], core.Void):
        raise ValueError("PDF MAC algorithm protection cannot name a signature algorithm")
    if isinstance(algorithm_protection["mac_algorithm"], core.Void):
        raise ValueError("PDF MAC algorithm protection is missing its MAC algorithm")
    if not algorithm_identifiers_match(
        algorithm_protection["digest_algorithm"],
        auth_data["digest_algorithm"],
    ):
        raise ValueError("PDF MAC digest algorithm is not protected")
    if not algorithm_identifiers_match(
        algorithm_protection["mac_algorithm"],
        auth_data["mac_algorithm"],
    ):
        raise ValueError("PDF MAC algorithm is not protected")


def validate_integrity_info(
    integrity_info: PdfMacIntegrityInfo,
    document_digest: bytes,
) -> None:
    if integrity_info["version"].native != 0:
        raise ValueError("invalid PdfMacIntegrityInfo version")
    if not isinstance(integrity_info["signature_digest"], core.Void):
        raise ValueError("standalone PDF MAC cannot contain a signature digest")
    if integrity_info["data_digest"].native != document_digest:
        raise ValueError("PDF MAC document digest does not match")


def internal_encapsulated_content(auth_data: cms.AuthenticatedData) -> bytes:
    content_info = auth_data["encap_content_info"]
    if content_info["content_type"].dotted != PDF_MAC_INTEGRITY_INFO_OID:
        raise ValueError("incorrect PDF MAC encapsulated content type")
    content = content_info["content"]
    if isinstance(content, core.Void):
        raise ValueError("PDF MAC encapsulated content is missing")
    return bytes(content)


def unique_attribute(
    attributes: cms.CMSAttributes,
    oid: str,
    *,
    required: bool = True,
) -> core.Asn1Value | None:
    matches = [attribute for attribute in attributes if attribute["type"].dotted == oid]
    if not matches and not required:
        return None
    if len(matches) != 1 or len(matches[0]["values"]) != 1:
        raise ValueError(f"PDF MAC attribute {oid} must occur exactly once")
    return cast(core.Asn1Value, matches[0]["values"][0])


def digest_algorithm(identifier: Any) -> hashes.HashAlgorithm:
    oid = require_algorithm(identifier, None, require_absent_parameters=False)
    match oid:
        case value if value == SHA256_OID:
            algorithm: hashes.HashAlgorithm = hashes.SHA256()
        case value if value == SHA384_OID:
            algorithm = hashes.SHA384()
        case value if value == SHA512_OID:
            algorithm = hashes.SHA512()
        case value if value == SHA3_256_OID:
            algorithm = hashes.SHA3_256()
        case value if value == SHA3_384_OID:
            algorithm = hashes.SHA3_384()
        case value if value == SHA3_512_OID:
            algorithm = hashes.SHA3_512()
        case _:
            raise UnsupportedAlgorithmError(f"Unsupported PDF MAC digest algorithm: {oid}")
    parameters = identifier["parameters"]
    if oid in {SHA3_256_OID, SHA3_384_OID, SHA3_512_OID}:
        valid_parameters = isinstance(parameters, core.Void)
    else:
        valid_parameters = isinstance(parameters, (core.Void, core.Null))
    if not valid_parameters:
        raise ValueError("invalid PDF MAC digest algorithm parameters")
    return algorithm


def validate_mac_algorithm(identifier: Any) -> None:
    require_algorithm(
        identifier,
        HMAC_SHA256_OID,
        require_absent_parameters=False,
    )
    parameters = identifier["parameters"]
    parameters_are_null = isinstance(parameters, core.Null) or (
        isinstance(parameters, core.Any) and isinstance(parameters.parsed, core.Null)
    )
    if not isinstance(parameters, core.Void) and not parameters_are_null:
        raise ValueError("invalid PDF MAC algorithm parameters")


def require_algorithm(
    identifier: Any,
    expected_oid: str | None,
    *,
    require_absent_parameters: bool,
) -> str:
    algorithm = identifier["algorithm"]
    oid = algorithm.dotted
    if expected_oid is not None and oid != expected_oid:
        raise ValueError(f"unexpected PDF MAC algorithm: {oid}")
    if require_absent_parameters and not isinstance(identifier["parameters"], core.Void):
        raise ValueError(f"PDF MAC algorithm {oid} cannot have parameters")
    return cast(str, oid)


def algorithm_identifiers_match(left: Any, right: Any) -> bool:
    if left["algorithm"].dotted != right["algorithm"].dotted:
        return False
    left_parameters = left["parameters"]
    right_parameters = right["parameters"]
    if isinstance(left_parameters, core.Void) or isinstance(right_parameters, core.Void):
        return isinstance(left_parameters, core.Void) and isinstance(right_parameters, core.Void)
    return left_parameters.dump(force=True) == right_parameters.dump(force=True)


def digest(data: bytes, algorithm: hashes.HashAlgorithm) -> bytes:
    digest = hashes.Hash(algorithm)
    digest.update(data)
    return digest.finalize()


def digest_byte_range(
    raw_data: Buffer,
    byte_range: tuple[int, int, int, int],
    algorithm: hashes.HashAlgorithm,
) -> bytes:
    first_start, first_length, second_start, second_length = byte_range
    digest = hashes.Hash(algorithm)
    view = memoryview(raw_data)
    try:
        digest.update(view[first_start : first_start + first_length])
        digest.update(view[second_start : second_start + second_length])
    finally:
        view.release()
    return digest.finalize()


def parse_der(data: bytes, asn1_type: Any) -> Any:
    value = asn1_type.load(data, strict=True)
    if value.dump(force=True) != data:
        raise ValueError("PDF MAC value is not canonical DER")
    return value


__all__ = (
    "PdfMacIntegrityInfo",
    "validate_pdf_mac_token",
    "validate_authenticated_data",
    "unwrap_mac_key",
    "digest",
    "digest_algorithm",
    "digest_byte_range",
    "parse_der",
)

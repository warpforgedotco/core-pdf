# SPDX-License-Identifier: AGPL-3.0-only
"""ISO/TS 32004:2024 standalone PDF MAC validation.

ASN.1 and CMS structure are parsed by asn1crypto. All cryptographic operations
use PyCA cryptography: HKDF-SHA-256, AES-256 key unwrap, document and content
digests, and HMAC-SHA-256 verification.
"""

from __future__ import annotations

from typing import Any, cast

from asn1crypto import cms, core
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, hmac, keywrap
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from core_pdf.impl.exceptions import PdfDecryptionError, PdfUnsupportedError
from core_pdf.impl.spec.s_07_security.standard import internal_StandardSecurityHandler
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.types import MISSING, PdfByteBuffer, PdfName, PdfString

internal_AUTHENTICATED_DATA_OID = "1.2.840.113549.1.9.16.1.2"
internal_CONTENT_TYPE_ATTRIBUTE_OID = "1.2.840.113549.1.9.3"
internal_MESSAGE_DIGEST_ATTRIBUTE_OID = "1.2.840.113549.1.9.4"
internal_CMS_ALGORITHM_PROTECTION_ATTRIBUTE_OID = "1.2.840.113549.1.9.52"
internal_PDF_MAC_INTEGRITY_INFO_OID = "1.0.32004.1.0"
internal_PDF_MAC_WRAP_KDF_OID = "1.0.32004.1.1"
internal_AES_256_KEY_WRAP_OID = "2.16.840.1.101.3.4.1.45"
internal_HMAC_SHA256_OID = "1.2.840.113549.2.9"
internal_SHA256_OID = "2.16.840.1.101.3.4.2.1"
internal_SHA384_OID = "2.16.840.1.101.3.4.2.2"
internal_SHA512_OID = "2.16.840.1.101.3.4.2.3"
internal_SHA3_256_OID = "2.16.840.1.101.3.4.2.8"
internal_SHA3_384_OID = "2.16.840.1.101.3.4.2.9"
internal_SHA3_512_OID = "2.16.840.1.101.3.4.2.10"
internal_PDF_MAC_KDF_SALT_BYTES = 32
internal_PDF_MAC_KEK_BYTES = 32
internal_PDF_MAC_KEY_BYTES = 32
internal_PDF_MAC_WRAPPED_KEY_BYTES = 40
internal_PDF_MAC_HKDF_INFO = b"PDFMAC"
internal_HEXADECIMAL_DIGITS = frozenset(b"0123456789ABCDEFabcdef")


class internal_PdfMacIntegrityInfo(core.Sequence):
    """ISO/TS 32004:2024, 6.2 PdfMacIntegrityInfo ASN.1 sequence."""

    _fields = [
        ("version", core.Integer),
        ("data_digest", core.OctetString),
        (
            "signature_digest",
            core.OctetString,
            {"implicit": 0, "optional": True},
        ),
    ]


def validate_pdf_mac_if_present(
    raw_data: PdfByteBuffer,
    trailer: PdfDict,
    handler: internal_StandardSecurityHandler,
) -> bool:
    """Validate ISO/TS 32004:2024 AuthCode and report whether PDF MAC is in use."""
    raw_auth_code = trailer.get("AuthCode", MISSING)
    has_auth_code = raw_auth_code is not MISSING
    has_kdf_salt = handler.config.kdf_salt is not None

    # ISO/TS 32004:2024, Table 2 requires KDFSalt in documents using PDF
    # MAC, and Tables 3 and 5 require AuthCode whenever permission bit 13 is
    # zero. Validate either trace even when bit 13 is one so stripping only
    # one of the two signals cannot silently disable integrity protection.
    if not (handler.config.pdf_mac_required or has_auth_code or has_kdf_salt):
        return False
    if handler.config.version < 5:
        raise PdfDecryptionError("Invalid PDF MAC")
    if not isinstance(raw_auth_code, dict) or handler.config.kdf_salt is None:
        raise PdfDecryptionError("Invalid PDF MAC")

    try:
        internal_validate_standalone_pdf_mac(
            raw_data,
            cast(PdfDict, raw_auth_code),
            handler.file_key,
            handler.config.kdf_salt,
        )
    except PdfUnsupportedError:
        raise
    except (InvalidSignature, ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        raise PdfDecryptionError("Invalid PDF MAC") from exc
    return True


def validate_pdf_mac_extension(declarations: object) -> None:
    """Require the developer-extension declaration from ISO/TS 32004:2024, Table 1."""
    if not isinstance(declarations, list):
        raise PdfDecryptionError("Invalid PDF MAC extension declaration")

    for declaration in declarations:
        if not isinstance(declaration, dict):
            continue
        extension_level = declaration.get("ExtensionLevel")
        revision = declaration.get("ExtensionRevision")
        url = declaration.get("URL")
        if (
            declaration.get("Type") == PdfName.of("DeveloperExtensions")
            and declaration.get("BaseVersion") == PdfName.of("2.0")
            and type(extension_level) is int
            and extension_level == 32004
            and isinstance(revision, PdfString)
            and revision.data == b":2024"
            and isinstance(url, PdfString)
            and url.data == b"https://www.iso.org/standard/45877.html"
        ):
            return

    raise PdfDecryptionError("Invalid PDF MAC extension declaration")


def internal_validate_standalone_pdf_mac(
    raw_data: PdfByteBuffer,
    auth_code: PdfDict,
    file_key: bytes,
    kdf_salt: bytes,
) -> None:
    """Validate an ISO/TS 32004:2024 standalone AuthCode dictionary and token."""
    # ISO/TS 32004:2024, Table 6 requires MACLocation to be a direct name.
    raw_location = auth_code.get("MACLocation", MISSING)
    if not isinstance(raw_location, PdfName):
        raise ValueError("invalid PDF MAC location")
    location = raw_location.value
    if location == "AttachedToSig":
        raise PdfUnsupportedError("Attached-to-signature PDF MAC is not supported")
    if location != "Standalone":
        raise PdfUnsupportedError(f"Unsupported PDF MAC location: {location}")
    if "SigObjRef" in auth_code:
        raise ValueError("standalone PDF MAC cannot contain SigObjRef")

    byte_range, token = internal_extract_standalone_token(raw_data, auth_code)
    content_info = internal_parse_der(token, cms.ContentInfo)
    if content_info["content_type"].dotted != internal_AUTHENTICATED_DATA_OID:
        raise ValueError("PDF MAC token is not CMS AuthenticatedData")
    auth_data = content_info["content"]
    if not isinstance(auth_data, cms.AuthenticatedData):
        raise ValueError("invalid CMS AuthenticatedData")

    internal_validate_authenticated_data(
        raw_data,
        byte_range,
        auth_data,
        file_key,
        kdf_salt,
    )


def internal_extract_standalone_token(
    raw_data: PdfByteBuffer,
    auth_code: PdfDict,
) -> tuple[tuple[int, int, int, int], bytes]:
    """Apply ISO/TS 32004:2024, Table 6 and 6.5.1 byte-coverage rules."""
    raw_byte_range = auth_code.get("ByteRange", MISSING)
    if not isinstance(raw_byte_range, list) or len(raw_byte_range) != 4:
        raise ValueError("invalid PDF MAC ByteRange")
    if any(type(value) is not int or value < 0 for value in raw_byte_range):
        raise ValueError("invalid PDF MAC ByteRange")
    first_start, first_length, second_start, second_length = cast(
        tuple[int, int, int, int],
        tuple(raw_byte_range),
    )

    # ISO/TS 32004:2024, Table 6 requires [0, L1, S, L2], and 6.5.1
    # requires the latest standalone token to cover the entire file except
    # the MAC string value itself.
    if (
        first_start != 0
        or second_start <= first_length
        or second_start > len(raw_data)
        or second_start + second_length != len(raw_data)
    ):
        raise ValueError("PDF MAC ByteRange does not cover the entire file")

    raw_mac = auth_code.get("MAC", MISSING)
    if not isinstance(raw_mac, PdfString) or raw_mac.is_literal is not False:
        raise ValueError("standalone PDF MAC must be a hexadecimal string")

    # ISO/TS 32004:2024, Table 6 is stricter than ordinary PDF hexadecimal
    # strings: the excluded region is exactly '<' + two hex digits per DER
    # byte + '>', with no whitespace, missing nibble, padding, or trailing data.
    serialized_mac = bytes(raw_data[first_length:second_start])
    encoded_token = serialized_mac[1:-1]
    if (
        len(serialized_mac) != (2 * len(raw_mac.data)) + 2
        or not serialized_mac.startswith(b"<")
        or not serialized_mac.endswith(b">")
        or any(byte not in internal_HEXADECIMAL_DIGITS for byte in encoded_token)
        or bytes.fromhex(encoded_token.decode("ascii")) != raw_mac.data
    ):
        raise ValueError("invalid serialized standalone PDF MAC")

    return (
        first_start,
        first_length,
        second_start,
        second_length,
    ), raw_mac.data


def internal_validate_authenticated_data(
    raw_data: PdfByteBuffer,
    byte_range: tuple[int, int, int, int],
    auth_data: cms.AuthenticatedData,
    file_key: bytes,
    kdf_salt: bytes,
) -> None:
    """Validate ISO/TS 32004:2024, 6.2-6.4 and 6.6 CMS requirements."""
    if auth_data["version"].native != "v0":
        raise ValueError("invalid PDF MAC AuthenticatedData version")
    if not isinstance(auth_data["originator_info"], core.Void):
        # RFC 5652:2009, 9.1 permits originatorInfo only when the key
        # management algorithm needs it. ISO/TS 32004:2024, 6.3.3 fixes the
        # sole recipient to PasswordRecipientInfo, which does not use it.
        raise ValueError("PDF MAC cannot contain originator information")
    if not isinstance(auth_data["unauth_attrs"], core.Void):
        # ISO/TS 32004:2024, 6.3.7 forbids unauthenticated attributes.
        raise ValueError("PDF MAC cannot contain unauthenticated attributes")

    digest_algorithm = internal_digest_algorithm(auth_data["digest_algorithm"])
    internal_validate_mac_algorithm(auth_data["mac_algorithm"])
    encapsulated_content = internal_encapsulated_content(auth_data)
    integrity_info = internal_parse_der(encapsulated_content, internal_PdfMacIntegrityInfo)

    mac_key = internal_unwrap_mac_key(
        auth_data["recipient_infos"],
        file_key,
        kdf_salt,
    )
    auth_attrs = auth_data["auth_attrs"]
    if not isinstance(auth_attrs, cms.CMSAttributes):
        raise ValueError("PDF MAC authenticated attributes are missing")
    received_mac = auth_data["mac"].native
    if not isinstance(received_mac, bytes) or len(received_mac) != internal_PDF_MAC_KEY_BYTES:
        raise ValueError("invalid PDF MAC value")

    # ISO/TS 32004:2024, 6.3.5 requires HMAC-SHA-256 with a 256-bit key.
    # RFC 5652:2009, 9.2 requires the DER SET OF encoding of authAttrs as
    # HMAC input, not its context-specific [2] wrapper.
    verifier = hmac.HMAC(mac_key, hashes.SHA256())
    verifier.update(auth_attrs.untag().dump(force=True))
    verifier.verify(received_mac)

    internal_validate_authenticated_attributes(
        auth_attrs,
        auth_data,
        encapsulated_content,
        digest_algorithm,
    )
    internal_validate_integrity_info(
        integrity_info,
        internal_digest_byte_range(raw_data, byte_range, digest_algorithm),
    )


def internal_unwrap_mac_key(
    recipient_infos: cms.RecipientInfos,
    file_key: bytes,
    kdf_salt: bytes,
) -> bytes:
    """Apply ISO/TS 32004:2024, 6.3.3 and 6.4 key derivation and unwrap."""
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
    internal_require_algorithm(
        kdf,
        internal_PDF_MAC_WRAP_KDF_OID,
        require_absent_parameters=True,
    )
    internal_require_algorithm(
        password_info["key_encryption_algorithm"],
        internal_AES_256_KEY_WRAP_OID,
        require_absent_parameters=True,
    )

    encrypted_key = password_info["encrypted_key"].native
    if (
        not isinstance(encrypted_key, bytes)
        or len(encrypted_key) != internal_PDF_MAC_WRAPPED_KEY_BYTES
    ):
        raise ValueError("invalid wrapped PDF MAC key")
    if len(kdf_salt) != internal_PDF_MAC_KDF_SALT_BYTES:
        raise ValueError("invalid PDF MAC KDFSalt")

    # ISO/TS 32004:2024, 6.4 fixes HKDF to SHA-256, 32-byte KDFSalt,
    # UTF-8 "PDFMAC" info, the file encryption key as input, and a 256-bit
    # output because Table 7 requires AES-256 key wrap without padding.
    key_encryption_key = HKDF(
        algorithm=hashes.SHA256(),
        length=internal_PDF_MAC_KEK_BYTES,
        salt=kdf_salt,
        info=internal_PDF_MAC_HKDF_INFO,
    ).derive(file_key)
    try:
        mac_key = keywrap.aes_key_unwrap(key_encryption_key, encrypted_key)
    except keywrap.InvalidUnwrap as exc:
        raise ValueError("PDF MAC key unwrap failed") from exc
    if len(mac_key) != internal_PDF_MAC_KEY_BYTES:
        raise ValueError("invalid unwrapped PDF MAC key")
    return mac_key


def internal_validate_authenticated_attributes(
    attributes: cms.CMSAttributes,
    auth_data: cms.AuthenticatedData,
    encapsulated_content: bytes,
    digest_algorithm: hashes.HashAlgorithm,
) -> None:
    """Apply ISO/TS 32004:2024, 6.3.6 authenticated-attribute rules."""
    content_type = internal_unique_attribute(attributes, internal_CONTENT_TYPE_ATTRIBUTE_OID)
    if not isinstance(content_type, cms.ContentType):
        raise ValueError("invalid PDF MAC content-type attribute")
    if content_type.dotted != internal_PDF_MAC_INTEGRITY_INFO_OID:
        raise ValueError("incorrect PDF MAC content-type attribute")

    message_digest = internal_unique_attribute(attributes, internal_MESSAGE_DIGEST_ATTRIBUTE_OID)
    if not isinstance(message_digest, core.OctetString):
        raise ValueError("invalid PDF MAC message-digest attribute")
    if message_digest.native != internal_digest(encapsulated_content, digest_algorithm):
        raise ValueError("incorrect PDF MAC message-digest attribute")

    algorithm_protection = internal_unique_attribute(
        attributes,
        internal_CMS_ALGORITHM_PROTECTION_ATTRIBUTE_OID,
        required=False,
    )
    # ISO/TS 32004:2024, 6.3.6.4 says this attribute SHOULD, rather than
    # SHALL, be present. RFC 6211:2011, 2-3 makes its contents mandatory
    # and unique when it is present.
    if algorithm_protection is None:
        return
    if not isinstance(algorithm_protection, cms.CMSAlgorithmProtection):
        raise ValueError("invalid PDF MAC algorithm-protection attribute")
    if not isinstance(algorithm_protection["signature_algorithm"], core.Void):
        raise ValueError("PDF MAC algorithm protection cannot name a signature algorithm")
    if isinstance(algorithm_protection["mac_algorithm"], core.Void):
        raise ValueError("PDF MAC algorithm protection is missing its MAC algorithm")
    if not internal_algorithm_identifiers_match(
        algorithm_protection["digest_algorithm"],
        auth_data["digest_algorithm"],
    ):
        raise ValueError("PDF MAC digest algorithm is not protected")
    if not internal_algorithm_identifiers_match(
        algorithm_protection["mac_algorithm"],
        auth_data["mac_algorithm"],
    ):
        raise ValueError("PDF MAC algorithm is not protected")


def internal_validate_integrity_info(
    integrity_info: internal_PdfMacIntegrityInfo,
    document_digest: bytes,
) -> None:
    """Apply ISO/TS 32004:2024, 6.2 and 6.6.2 to an unsigned revision."""
    if integrity_info["version"].native != 0:
        raise ValueError("invalid PdfMacIntegrityInfo version")
    if not isinstance(integrity_info["signature_digest"], core.Void):
        raise ValueError("standalone PDF MAC cannot contain a signature digest")
    if integrity_info["data_digest"].native != document_digest:
        raise ValueError("PDF MAC document digest does not match")


def internal_encapsulated_content(auth_data: cms.AuthenticatedData) -> bytes:
    """Read ISO/TS 32004:2024, 6.3.2 PdfMacIntegrityInfo content."""
    content_info = auth_data["encap_content_info"]
    if content_info["content_type"].dotted != internal_PDF_MAC_INTEGRITY_INFO_OID:
        raise ValueError("incorrect PDF MAC encapsulated content type")
    content = content_info["content"]
    if isinstance(content, core.Void):
        raise ValueError("PDF MAC encapsulated content is missing")
    # Read the OCTET STRING payload itself instead of ``native``. Other
    # installed CMS libraries may register the ISO 32004 OID globally with
    # asn1crypto, causing ``native`` to return a parsed mapping rather than the
    # exact DER bytes whose digest ISO/TS 32004:2024, 6.3.6.3 authenticates.
    return bytes(content)


def internal_unique_attribute(
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


def internal_digest_algorithm(identifier: Any) -> hashes.HashAlgorithm:
    # ISO/TS 32004:2024, Table 8 permits these six digest algorithms.
    oid = internal_require_algorithm(identifier, None, require_absent_parameters=False)
    match oid:
        case value if value == internal_SHA256_OID:
            algorithm: hashes.HashAlgorithm = hashes.SHA256()
        case value if value == internal_SHA384_OID:
            algorithm = hashes.SHA384()
        case value if value == internal_SHA512_OID:
            algorithm = hashes.SHA512()
        case value if value == internal_SHA3_256_OID:
            algorithm = hashes.SHA3_256()
        case value if value == internal_SHA3_384_OID:
            algorithm = hashes.SHA3_384()
        case value if value == internal_SHA3_512_OID:
            algorithm = hashes.SHA3_512()
        case _:
            raise PdfUnsupportedError(f"Unsupported PDF MAC digest algorithm: {oid}")
    parameters = identifier["parameters"]
    if oid in {internal_SHA3_256_OID, internal_SHA3_384_OID, internal_SHA3_512_OID}:
        valid_parameters = isinstance(parameters, core.Void)
    else:
        # SHA-2 AlgorithmIdentifier parameters occur both absent and as NULL
        # in deployed CMS. NIST's SHA-3 identifiers require them to be absent.
        valid_parameters = isinstance(parameters, (core.Void, core.Null))
    if not valid_parameters:
        raise ValueError("invalid PDF MAC digest algorithm parameters")
    return algorithm


def internal_validate_mac_algorithm(identifier: Any) -> None:
    internal_require_algorithm(
        identifier,
        internal_HMAC_SHA256_OID,
        require_absent_parameters=False,
    )
    # ISO/TS 32004:2024, Table 9 selects HMAC-SHA-256 in accordance with
    # RFC 4231:2005, 3.1. RFC 4231 recommends NULL parameters but does not
    # prohibit their omission, so a validator has to accept both encodings.
    parameters = identifier["parameters"]
    parameters_are_null = isinstance(parameters, core.Null) or (
        isinstance(parameters, core.Any) and isinstance(parameters.parsed, core.Null)
    )
    if not isinstance(parameters, core.Void) and not parameters_are_null:
        raise ValueError("invalid PDF MAC algorithm parameters")


def internal_require_algorithm(
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


def internal_algorithm_identifiers_match(left: Any, right: Any) -> bool:
    """Compare RFC 6211 algorithm values without their context-specific tags."""
    if left["algorithm"].dotted != right["algorithm"].dotted:
        return False
    left_parameters = left["parameters"]
    right_parameters = right["parameters"]
    if isinstance(left_parameters, core.Void) or isinstance(right_parameters, core.Void):
        return isinstance(left_parameters, core.Void) and isinstance(right_parameters, core.Void)
    return left_parameters.dump(force=True) == right_parameters.dump(force=True)


def internal_digest(data: bytes, algorithm: hashes.HashAlgorithm) -> bytes:
    digest = hashes.Hash(algorithm)
    digest.update(data)
    return digest.finalize()


def internal_digest_byte_range(
    raw_data: PdfByteBuffer,
    byte_range: tuple[int, int, int, int],
    algorithm: hashes.HashAlgorithm,
) -> bytes:
    # ISO/TS 32004:2024, 6.6.2 hashes both ranges in order to obtain
    # PdfMacIntegrityInfo.dataDigest for an unsigned revision.
    first_start, first_length, second_start, second_length = byte_range
    digest = hashes.Hash(algorithm)
    view = memoryview(raw_data)
    try:
        digest.update(view[first_start : first_start + first_length])
        digest.update(view[second_start : second_start + second_length])
    finally:
        view.release()
    return digest.finalize()


def internal_parse_der(data: bytes, asn1_type: Any) -> Any:
    # ISO/TS 32004:2024, Table 6 and 6.3 require DER, not merely a BER
    # encoding accepted by a tolerant ASN.1 parser. Re-encoding must therefore
    # reproduce every byte, and strict loading forbids unconsumed trailing data.
    value = asn1_type.load(data, strict=True)
    if value.dump(force=True) != data:
        raise ValueError("PDF MAC value is not canonical DER")
    return value


__all__ = ("validate_pdf_mac_extension", "validate_pdf_mac_if_present")

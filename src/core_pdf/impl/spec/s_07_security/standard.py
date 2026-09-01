# SPDX-License-Identifier: AGPL-3.0-only
"""PDF Standard Security parsing, authentication, and object decryption.

Implements ISO 32000-1:2008 revisions 2-4, the Adobe ExtensionLevel 3 revision
5 supplement, ISO 32000-2:2020 revision 6, ISO/TS 32003:2023 revision 7,
and the ISO/TS 32004:2024 PDF MAC signal and key-derivation salt.
"""

from __future__ import annotations

import stringprep
import struct
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import md5, sha256, sha384, sha512
from hmac import compare_digest
from types import MappingProxyType
from typing import Literal, cast

from core_pdf.impl.exceptions import PdfDecryptionError, PdfParseError, PdfUnsupportedError
from core_pdf.impl.primitives import MISSING
from core_pdf.impl.spec.s_07_filters.decode_spec import normalize_stream_decode_spec
from core_pdf.impl.spec.s_07_security.ciphers import (
    internal_aes_cbc_decrypt,
    internal_aes_cbc_encrypt,
    internal_aes_ecb_decrypt,
    internal_aes_gcm_decrypt,
    internal_rc4_crypt,
)
from core_pdf.impl.spec.s_07_syntax.types import Decipher, PdfDict
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    coerce_to_bytes,
    is_pdf_null,
    normalize_pdf_name,
    parse_int,
)

internal_CryptMethod = Literal["V2", "AESV2", "AESV3", "AESV4"]

internal_PASSWORD_PADDING = (
    b"\x28\xbf\x4e\x5e\x4e\x75\x8a\x41\x64\x00\x4e\x56\xff\xfa\x01\x08"
    b"\x2e\x2e\x00\xb6\xd0\x68\x3e\x80\x2f\x0c\xa9\xfe\x64\x53\x69\x7a"
)
internal_SASLPREP_PROHIBITED: tuple[Callable[[str], bool], ...] = (
    stringprep.in_table_c12,
    stringprep.in_table_c21_c22,
    stringprep.in_table_c3,
    stringprep.in_table_c4,
    stringprep.in_table_c5,
    stringprep.in_table_c6,
    stringprep.in_table_c7,
    stringprep.in_table_c8,
    stringprep.in_table_c9,
    stringprep.in_table_a1,
)

# Adobe PDF Reference 1.7, sixth edition (November 2006), Table 3.20, and
# ISO 32000-2:2020, Table 22 define these reserved positions explicitly.
# ISO 32000-1:2008, 7.6.3.2 also requires the reserved high-order bits to be
# one. The Adobe Supplement to ISO 32000, BaseVersion 1.7, ExtensionLevel 3
# (June 2008), 3.5.2 retains the PDF 1.7 permission flags for revision 5.
internal_RESERVED_ZERO_PERMISSION_BITS = (1, 2)
internal_RESERVED_ONE_PERMISSION_BITS = (7, 8, *range(13, 33))
internal_REVISION_3_PERMISSION_BITS = (9, 10, 11, 12)
internal_PDF_MAC_PERMISSION_BIT = 13
internal_PDF_MAC_PERMISSION_MASK = 1 << (internal_PDF_MAC_PERMISSION_BIT - 1)
internal_RESERVED_ZERO_PERMISSION_MASK = sum(
    1 << (bit_position - 1) for bit_position in internal_RESERVED_ZERO_PERMISSION_BITS
)
internal_RESERVED_ONE_PERMISSION_MASK = sum(
    1 << (bit_position - 1) for bit_position in internal_RESERVED_ONE_PERMISSION_BITS
)


@dataclass(frozen=True, slots=True)
class internal_StandardSecurityConfig:
    version: int
    revision: int
    permissions: int
    owner_entry: bytes
    user_entry: bytes
    length_bits: int
    document_id: bytes
    encrypt_metadata: bool
    stream_filter: str
    string_filter: str
    embedded_file_filter: str
    crypt_filters: Mapping[str, internal_CryptMethod]
    owner_encrypted_key: bytes
    user_encrypted_key: bytes
    encrypted_permissions: bytes
    kdf_salt: bytes | None = None
    pdf_mac_required: bool = False


@dataclass(frozen=True, slots=True)
class internal_StandardSecurityHandler:
    config: internal_StandardSecurityConfig
    file_key: bytes
    # ``decrypt`` runs once per string, not once per object, so the per-object
    # MD5 below is re-derived thousands of times per page without this memo.
    object_key_cache: dict[tuple[int, int, bytes], bytes] = field(
        default_factory=dict, compare=False, repr=False
    )

    def decrypt(
        self,
        object_number: int,
        generation_number: int,
        data: bytes,
        attrs: PdfDict | None = None,
        name: str | None = None,
    ) -> bytes:
        config = self.config
        if config.version in (1, 2):
            return internal_rc4_crypt(self.object_key(object_number, generation_number), data)

        default_stream_filter = config.stream_filter
        if attrs is not None:
            object_type = normalize_pdf_name(attrs.get("Type"))
            if not config.encrypt_metadata and object_type == "Metadata":
                return data
            # ISO 32000-1:2008, Table 20; Adobe Supplement to ISO 32000,
            # BaseVersion 1.7, ExtensionLevel 3, June 2008, Table 3.18; and
            # ISO 32000-2:2020, Table 20 assign EFF to embedded-file streams
            # that do not carry their own Crypt filter. If EFF is absent, its
            # parsed value is StmF, as those same tables require.
            if object_type == "EmbeddedFile":
                default_stream_filter = config.embedded_file_filter

        if name is None:
            name = (
                internal_stream_crypt_filter_name(attrs, default_stream_filter)
                if attrs is not None
                else config.string_filter
            )
        method = internal_resolve_crypt_method(name, config.crypt_filters)
        match method:
            case None:
                return data
            case "V2":
                return internal_rc4_crypt(self.object_key(object_number, generation_number), data)
            case "AESV2":
                key = self.object_key(object_number, generation_number, b"sAlT")
                return internal_aes_cbc_decrypt(
                    key,
                    data[:16],
                    data[16:],
                    use_padding=True,
                )
            case "AESV3":
                return internal_aes_cbc_decrypt(
                    self.file_key,
                    data[:16],
                    data[16:],
                    use_padding=True,
                )
            case "AESV4":
                # ISO/TS 32003:2023, 5.2 uses the 32-byte crypt-filter key
                # directly; unlike AESV2, it does not derive an object key.
                return internal_aes_gcm_decrypt(self.file_key, data)

    def object_key(
        self,
        object_number: int,
        generation_number: int,
        extra: bytes = b"",
    ) -> bytes:
        cache = self.object_key_cache
        cache_key = (object_number, generation_number, extra)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        seed = (
            self.file_key
            + struct.pack("<L", object_number)[:3]
            + struct.pack("<L", generation_number)[:2]
            + extra
        )
        key = md5(seed).digest()[: min(len(seed), 16)]
        cache[cache_key] = key
        return key


def create_standard_decipher(
    document_id: Sequence[object],
    params: PdfDict,
    password: str = "",
) -> Decipher:
    """Validate a Standard Security dictionary and return its object decipher."""
    return create_standard_security_handler(document_id, params, password).decrypt


def create_standard_security_handler(
    document_id: Sequence[object],
    params: PdfDict,
    password: str = "",
) -> internal_StandardSecurityHandler:
    """Authenticate a Standard Security dictionary and retain its file key."""
    filter_name = normalize_pdf_name(params.get("Filter"))
    if filter_name is None:
        raise PdfUnsupportedError("Invalid encryption dictionary")
    if filter_name in {"Adobe.PubSec", "PubSec"}:
        raise PdfUnsupportedError("Public-key encryption is not supported")
    if filter_name != "Standard":
        raise PdfUnsupportedError(f"Unsupported encryption filter: {filter_name}")

    version = parse_int(params.get("V"), None)
    if version is None:
        raise PdfUnsupportedError("Invalid encryption dictionary")
    supported_revisions = internal_supported_revisions(version)
    if supported_revisions is None:
        raise PdfUnsupportedError(f"Unsupported standard encryption algorithm V={version}")

    try:
        config = internal_parse_config(document_id, params, version, supported_revisions)
    except (TypeError, ValueError) as exc:
        raise PdfUnsupportedError("Invalid encryption dictionary") from exc
    file_key = internal_authenticate(config, password)
    if file_key is None:
        raise PdfUnsupportedError("Incorrect password")
    if config.revision >= 5 and not internal_validate_permissions(config, file_key):
        raise PdfDecryptionError("Invalid encryption permissions")
    return internal_StandardSecurityHandler(config, file_key)


def internal_parse_config(
    document_id: Sequence[object],
    params: PdfDict,
    version: int,
    supported_revisions: tuple[int, ...],
) -> internal_StandardSecurityConfig:
    revision = internal_required_int(params, "R")
    if revision not in supported_revisions:
        raise ValueError(f"unsupported Standard Security revision R={revision} for V={version}")

    raw_permissions = params.get("P", MISSING)
    if raw_permissions is MISSING or raw_permissions is None:
        raise ValueError("missing encryption permissions")
    permissions = internal_parse_int(raw_permissions, "P")
    # ISO 32000-1:2008, 7.6.3.2 and ISO 32000-2:2020, 7.6.4.2
    # define P as an unsigned 32-bit flag word, even though PDF integer syntax
    # commonly represents it as a negative signed value.
    if not -(1 << 31) <= permissions <= (1 << 32) - 1:
        raise ValueError("encryption permissions are outside the 32-bit range")
    if permissions < 0:
        permissions += 1 << 32

    # ISO 32000-2:2020, Table 22 reserves bit 13 as one. ISO/TS 32004:2024,
    # 5.1.2 and Table 3 supersede that rule for V >= 5: zero means that every
    # revision requires a PDF MAC token located through the trailer AuthCode
    # dictionary. Older encryption versions retain the reserved-one rule.
    pdf_mac_supported_version = version in (5, 6)
    pdf_mac_required = pdf_mac_supported_version and not (
        permissions & internal_PDF_MAC_PERMISSION_MASK
    )
    reserved_one_mask = internal_RESERVED_ONE_PERMISSION_MASK
    if pdf_mac_supported_version:
        reserved_one_mask &= ~internal_PDF_MAC_PERMISSION_MASK
    if permissions & internal_RESERVED_ZERO_PERMISSION_MASK:
        raise ValueError("reserved encryption permission bits 1-2 must be zero")
    if permissions & reserved_one_mask != reserved_one_mask:
        raise ValueError("reserved encryption permission bits must be one")

    if version == 1 and revision == 2:
        # ISO 32000-1:2008, Table 21 requires R=3, rather than R=2, when
        # any permission introduced for revision 3 (bits 9 through 12 in
        # Table 22) is cleared, even though V remains 1. Do not require R=2
        # in the opposite direction: Adobe's authorized ISO 32000-1:2008 PDF
        # itself uses V=1/R=3 with all four revision-3 permissions set.
        revision_3_required = any(
            permissions & (1 << (bit_position - 1)) == 0
            for bit_position in internal_REVISION_3_PERMISSION_BITS
        )
        if revision_3_required:
            raise ValueError(
                "Standard Security V=1 permissions require R=3 when any "
                "revision-3 permission is cleared, got R=2"
            )

    # Entry sizes come from these version-specific definitions:
    # - ISO 32000-1:2008, Table 21: O/U are 32 bytes through revision 4.
    # - Adobe Supplement to ISO 32000, BaseVersion 1.7, ExtensionLevel 3,
    #   June 2008, Table 3.19: R5 uses 48-byte O/U, 32-byte OE/UE, and
    #   a 16-byte Perms entry.
    # - ISO 32000-2:2020, Table 21 and 7.6.4.3.3: the same sizes apply to R6.
    # - ISO/TS 32003:2023, Table 3 and 5.2: R7 uses the R6 password
    #   algorithms, so those same entry sizes apply.
    entry_length = 32 if revision <= 4 else 48
    owner_entry = internal_required_bytes(params, "O", entry_length)
    user_entry = internal_required_bytes(params, "U", entry_length)
    first_document_id = coerce_to_bytes(document_id[0]) if document_id else b""

    raw_length = params.get("Length", MISSING)
    if raw_length is MISSING:
        match version:
            case 1 | 2:
                length_bits = 40
            case 4:
                length_bits = 128
            case 5 | 6:
                length_bits = 256
            case _:
                raise ValueError(f"encryption key length is not defined for V={version}")
    else:
        length_bits = internal_parse_int(raw_length, "Length")

    # ISO 32000-1:2008, Table 20 fixes V=1 at 40 bits and permits V=2
    # lengths from 40 through 128 bits. This backend deliberately implements
    # the interoperable 40- and 128-bit RC4 forms only. ISO 32000-1:2008,
    # 7.6.3.1 fixes the revision-4 Standard handler at 128 bits; the Adobe
    # ExtensionLevel 3 supplement, 3.5 and Algorithm 3.1a, and
    # ISO 32000-2:2020, 7.6.3.3 fix V=5 at 256 bits. ISO/TS 32003:2023,
    # Tables 2 and 4 retain that 32-byte size for V=6/AESV4. qpdf redundantly
    # emits Length for V=1, V=4, and V=5, so accept a declared length only when
    # it states the format-mandated size instead of replacing a contradiction.
    match version:
        case 1:
            if length_bits != 40:
                raise ValueError(f"invalid V=1 encryption key length: {length_bits}")
        case 2:
            if length_bits not in (40, 128):
                raise ValueError(
                    f"unsupported legacy RC4 key length for the cryptography backend: {length_bits}"
                )
        case 4:
            if length_bits != 128:
                raise ValueError(f"invalid V=4 encryption key length: {length_bits}")
        case 5 | 6:
            if length_bits != 256:
                raise ValueError(f"invalid V={version} encryption key length: {length_bits}")

    encrypt_metadata = True
    stream_filter = "Identity"
    string_filter = "Identity"
    embedded_file_filter = "Identity"
    crypt_filters: Mapping[str, internal_CryptMethod] = MappingProxyType({})
    owner_encrypted_key = b""
    user_encrypted_key = b""
    encrypted_permissions = b""
    kdf_salt: bytes | None = None

    if version in (4, 5, 6):
        (
            encrypt_metadata,
            stream_filter,
            string_filter,
            embedded_file_filter,
            crypt_filters,
        ) = internal_parse_crypt_filters(params, version)
    if version in (5, 6):
        owner_encrypted_key = internal_required_bytes(params, "OE", 32)
        user_encrypted_key = internal_required_bytes(params, "UE", 32)
        encrypted_permissions = internal_required_bytes(params, "Perms", 16)

        raw_kdf_salt = params.get("KDFSalt", MISSING)
        if raw_kdf_salt is not MISSING:
            kdf_salt = internal_required_bytes(params, "KDFSalt", 32)
        if pdf_mac_required and kdf_salt is None:
            raise ValueError("PDF MAC requires a 32-byte KDFSalt")
    elif params.get("KDFSalt", MISSING) is not MISSING:
        # ISO/TS 32004:2024, Table 5 requires V >= 5 when AuthCode is
        # present; Table 2 defines KDFSalt only as part of that mechanism.
        raise ValueError("KDFSalt requires encryption algorithm V >= 5")

    return internal_StandardSecurityConfig(
        version=version,
        revision=revision,
        permissions=permissions,
        owner_entry=owner_entry,
        user_entry=user_entry,
        length_bits=length_bits,
        document_id=first_document_id,
        encrypt_metadata=encrypt_metadata,
        stream_filter=stream_filter,
        string_filter=string_filter,
        embedded_file_filter=embedded_file_filter,
        crypt_filters=crypt_filters,
        owner_encrypted_key=owner_encrypted_key,
        user_encrypted_key=user_encrypted_key,
        encrypted_permissions=encrypted_permissions,
        kdf_salt=kdf_salt,
        pdf_mac_required=pdf_mac_required,
    )


def internal_parse_crypt_filters(
    params: PdfDict,
    version: int,
) -> tuple[bool, str, str, str, Mapping[str, internal_CryptMethod]]:
    # The Standard handler's supported filter set is intentionally narrower
    # than the generic crypt-filter grammar. ISO 32000-1:2008, 7.6.3.1 limits
    # R=4 to Identity and StdCF with V2/AESV2 plus AuthEvent=DocOpen. Adobe
    # Supplement to ISO 32000, BaseVersion 1.7, ExtensionLevel 3, June 2008,
    # 3.5.2 applies that rule to V=5 with AESV3. ISO 32000-2:2020, 7.6.4.1
    # retains Identity/StdCF/DocOpen and requires AESV3 for R=6. ISO/TS
    # 32003:2023, Tables 2 and 4 extends the same model with AESV4 for V=6
    # and requires at least one crypt filter using it.
    raw_filters = params.get("CF", MISSING)
    if raw_filters is MISSING:
        filters: PdfDict = {}
    elif isinstance(raw_filters, dict):
        filters = cast(PdfDict, raw_filters)
    else:
        raise ValueError("invalid crypt filter dictionary")

    match version:
        case 4:
            allowed_methods = {"V2", "AESV2"}
        case 5:
            allowed_methods = {"AESV3"}
        case 6:
            allowed_methods = {"AESV4"}
        case _:
            raise ValueError(f"crypt filters are not defined for V={version}")
    crypt_filters: dict[str, internal_CryptMethod] = {}
    for raw_name, raw_config in filters.items():
        filter_name = internal_name(raw_name)
        if not filter_name:
            raise ValueError("invalid crypt filter name")
        # ISO 32000-1:2008, Table 20 says entries using a standard name are
        # ignored in favour of that name's built-in behaviour (Table 26).
        if filter_name == "Identity":
            continue
        if filter_name != "StdCF":
            raise ValueError(f"unsupported Standard Security crypt filter: {filter_name}")
        if not isinstance(raw_config, dict):
            raise ValueError(f"invalid crypt filter dictionary: {raw_name!r}")
        filter_config = cast(PdfDict, raw_config)

        raw_type = filter_config.get("Type", MISSING)
        if raw_type is not MISSING and internal_name(raw_type) != "CryptFilter":
            raise ValueError(f"invalid crypt filter type: {filter_name}")

        method_name = internal_name(filter_config.get("CFM", "None"))
        if method_name not in allowed_methods:
            raise ValueError(f"unknown crypt filter method: {method_name}")

        auth_event = internal_name(filter_config.get("AuthEvent", "DocOpen"))
        if auth_event != "DocOpen":
            raise ValueError(f"unsupported Standard Security authorization event: {auth_event}")

        # ISO 32000-1:2008, Table 25 expresses Standard-handler Length in
        # bytes (16 means 128 bits). Adobe ExtensionLevel 3, Table 3.22 and
        # ISO 32000-2:2020, Table 25 use 32 for the 256-bit AESV3 form.
        # ISO/TS 32003:2023, Table 4 specifies AESV4 Length in the same
        # manner as AESV3.
        raw_filter_length = filter_config.get("Length", MISSING)
        expected_filter_length = 32 if method_name in {"AESV3", "AESV4"} else 16
        if raw_filter_length is not MISSING:
            filter_length = internal_parse_int(raw_filter_length, "CF/Length")
            if filter_length != expected_filter_length:
                raise ValueError(f"invalid {method_name} crypt filter length: {filter_length}")

        crypt_filters[filter_name] = cast(internal_CryptMethod, method_name)

    if version == 6 and "AESV4" not in crypt_filters.values():
        raise ValueError("V=6 requires at least one AESV4 crypt filter")

    raw_stream_filter = params.get("StmF", MISSING)
    stream_filter = internal_name("Identity" if raw_stream_filter is MISSING else raw_stream_filter)
    raw_string_filter = params.get("StrF", MISSING)
    string_filter = internal_name("Identity" if raw_string_filter is MISSING else raw_string_filter)
    raw_embedded_file_filter = params.get("EFF", MISSING)
    embedded_file_filter = internal_name(
        stream_filter if raw_embedded_file_filter is MISSING else raw_embedded_file_filter
    )

    # ISO 32000-1:2008, Table 20; Adobe ExtensionLevel 3, Table 3.18; and
    # ISO 32000-2:2020, Table 20 require each selected default to be either a
    # CF key or the standard Identity filter. EFF defaults to StmF.
    for field_name, filter_name in (
        ("StmF", stream_filter),
        ("StrF", string_filter),
        ("EFF", embedded_file_filter),
    ):
        if not filter_name:
            raise ValueError(f"invalid {field_name} crypt filter")
        if filter_name != "Identity" and filter_name not in crypt_filters:
            raise ValueError(f"undefined {field_name} crypt filter: {filter_name}")

    encrypt_metadata = params.get("EncryptMetadata", MISSING)
    if encrypt_metadata is MISSING:
        encrypt_metadata = True
    if type(encrypt_metadata) is not bool:
        raise ValueError("invalid encryption metadata flag")
    return (
        encrypt_metadata,
        stream_filter,
        string_filter,
        embedded_file_filter,
        MappingProxyType(crypt_filters),
    )


def internal_stream_crypt_filter_name(attrs: PdfDict, default_filter: str) -> str:
    spec = normalize_stream_decode_spec(attrs)
    crypt_indexes = [
        index for index, filter_name in enumerate(spec.filters) if filter_name == "Crypt"
    ]
    if not crypt_indexes:
        return default_filter
    if len(crypt_indexes) != 1 or crypt_indexes[0] != 0:
        raise PdfParseError("Crypt must be the first and only Crypt stream filter")

    params = spec.params[crypt_indexes[0]]
    if is_pdf_null(params):
        return "Identity"
    if not isinstance(params, dict):
        raise PdfParseError("invalid Crypt filter params")
    raw_name = params.get("Name")
    if is_pdf_null(raw_name):
        return "Identity"
    filter_name = normalize_pdf_name(raw_name)
    if filter_name is None:
        raise PdfParseError("invalid Crypt filter name")
    return filter_name


def internal_resolve_crypt_method(
    name: str,
    crypt_filters: Mapping[str, internal_CryptMethod],
) -> internal_CryptMethod | None:
    if name == "Identity":
        return None
    method = crypt_filters.get(name)
    if method is None:
        raise PdfUnsupportedError(f"Undefined crypt filter: {name}")
    return method


def internal_authenticate(
    config: internal_StandardSecurityConfig,
    password: str,
) -> bytes | None:
    match config.revision:
        case 2 | 3 | 4:
            return internal_authenticate_legacy(config, password)
        case 5 | 6 | 7:
            return internal_authenticate_modern(config, password)
        case _:
            raise ValueError(f"unsupported Standard Security revision R={config.revision}")


def internal_authenticate_legacy(
    config: internal_StandardSecurityConfig,
    password: str,
) -> bytes | None:
    password_bytes = password.encode("latin-1")
    key = internal_authenticate_legacy_user(config, password_bytes)
    if key is not None:
        return key

    digest = md5(internal_pad_password(password_bytes)).digest()
    key_length = 5
    if config.revision >= 3:
        digest = internal_md5_50_rounds(digest)
        key_length = config.length_bits // 8
    owner_key = digest[:key_length]
    if config.revision == 2:
        user_password = internal_rc4_crypt(owner_key, config.owner_entry)
    else:
        user_password = internal_rc4_cascade(
            owner_key,
            config.owner_entry,
            range(19, -1, -1),
        )
    return internal_authenticate_legacy_user(config, user_password)


def internal_authenticate_legacy_user(
    config: internal_StandardSecurityConfig,
    password: bytes,
) -> bytes | None:
    key = internal_legacy_file_key(config, password)
    expected = internal_legacy_user_entry(config, key)
    if config.revision == 2:
        valid = compare_digest(expected, config.user_entry)
    else:
        valid = compare_digest(expected[:16], config.user_entry[:16])
    return key if valid else None


def internal_legacy_file_key(
    config: internal_StandardSecurityConfig,
    password: bytes,
) -> bytes:
    digest = md5(internal_pad_password(password))
    digest.update(config.owner_entry)
    digest.update(struct.pack("<L", config.permissions))
    digest.update(config.document_id)
    if config.revision >= 4 and not config.encrypt_metadata:
        digest.update(b"\xff\xff\xff\xff")
    result = digest.digest()
    key_length = 5
    if config.revision >= 3:
        key_length = config.length_bits // 8
        result = internal_md5_50_rounds(result, key_length)
    return result[:key_length]


def internal_legacy_user_entry(
    config: internal_StandardSecurityConfig,
    key: bytes,
) -> bytes:
    if config.revision == 2:
        return internal_rc4_crypt(key, internal_PASSWORD_PADDING)
    digest = md5(internal_PASSWORD_PADDING)
    digest.update(config.document_id)
    result = internal_rc4_crypt(key, digest.digest())
    result = internal_rc4_cascade(key, result, range(1, 20))
    return result + result


def internal_pad_password(password: bytes) -> bytes:
    return (password + internal_PASSWORD_PADDING)[:32]


def internal_md5_50_rounds(digest: bytes, keep: int = 16) -> bytes:
    for _ in range(50):
        digest = md5(digest[:keep]).digest()
    return digest


def internal_rc4_cascade(key: bytes, data: bytes, indexes: range) -> bytes:
    for index in indexes:
        data = internal_rc4_crypt(bytes(byte ^ index for byte in key), data)
    return data


def internal_authenticate_modern(
    config: internal_StandardSecurityConfig,
    password: str,
) -> bytes | None:
    password_bytes = internal_normalize_password(password, config.revision)
    owner_hash = config.owner_entry[:32]
    owner_validation_salt = config.owner_entry[32:40]
    owner_key_salt = config.owner_entry[40:]
    user_hash = config.user_entry[:32]
    user_validation_salt = config.user_entry[32:40]
    user_key_salt = config.user_entry[40:]

    password_hash = internal_password_hash(
        config.revision,
        password_bytes,
        owner_validation_salt,
        config.user_entry,
    )
    if compare_digest(password_hash, owner_hash):
        password_hash = internal_password_hash(
            config.revision,
            password_bytes,
            owner_key_salt,
            config.user_entry,
        )
        return internal_aes_cbc_decrypt(
            password_hash,
            bytes(16),
            config.owner_encrypted_key,
            use_padding=False,
        )

    password_hash = internal_password_hash(
        config.revision,
        password_bytes,
        user_validation_salt,
    )
    if compare_digest(password_hash, user_hash):
        password_hash = internal_password_hash(
            config.revision,
            password_bytes,
            user_key_salt,
        )
        return internal_aes_cbc_decrypt(
            password_hash,
            bytes(16),
            config.user_encrypted_key,
            use_padding=False,
        )
    return None


def internal_validate_permissions(
    config: internal_StandardSecurityConfig,
    file_key: bytes,
) -> bool:
    # The decrypted Perms block binds the file key to P (little-endian),
    # EncryptMetadata, and the fixed markers FF FF FF FF and "adb"; its final
    # four bytes are random. Sources: Adobe Supplement to ISO 32000,
    # BaseVersion 1.7, ExtensionLevel 3, June 2008, Algorithms 3.10 and 3.13
    # (R5); ISO 32000-2:2020, 7.6.4.4.9 Algorithm 10 and 7.6.4.4.12
    # Algorithm 13 (R6). ISO/TS 32003:2023, Table 3 and 5.2 apply the R6
    # password algorithms and entries unchanged to R7.
    decrypted = internal_aes_ecb_decrypt(file_key, config.encrypted_permissions)
    metadata_flag = b"T" if config.encrypt_metadata else b"F"
    expected = struct.pack("<L", config.permissions) + (b"\xff" * 4) + metadata_flag + b"adb"
    return compare_digest(decrypted[:12], expected)


def internal_normalize_password(password: str, revision: int) -> bytes:
    # ISO/TS 32003:2023, 5.2 requires R7 to use the R6 password algorithms
    # from ISO 32000-2:2020, 7.6.4.4, including SASLprep normalization.
    if revision in (6, 7) and password:
        password = internal_saslprep(password)
    return password.encode("utf-8")[:127]


def internal_password_hash(
    revision: int,
    password: bytes,
    salt: bytes,
    vector: bytes | None = None,
) -> bytes:
    # ISO/TS 32003:2023, 5.2 uses ISO 32000-2:2020's R6 Algorithm 2.B for R7.
    if revision == 5:
        digest = sha256(password)
        digest.update(salt)
        if vector is not None:
            digest.update(vector)
        return digest.digest()
    return internal_r6_password_hash(password, salt[:8], vector)


def internal_r6_password_hash(
    password: bytes,
    salt: bytes,
    vector: bytes | None = None,
) -> bytes:
    initial_hash = sha256(password)
    initial_hash.update(salt)
    if vector is not None:
        initial_hash.update(vector)
    result = initial_hash.digest()
    hashes = (sha256, sha384, sha512)
    round_number = last_byte = 0
    while round_number < 64 or last_byte > round_number - 32:
        repeated = (password + result + (vector or b"")) * 64
        encrypted = internal_aes_cbc_encrypt(
            result[:16],
            result[16:32],
            repeated,
            use_padding=False,
        )
        next_hash = hashes[internal_bytes_mod_3(encrypted[:16])]
        result = next_hash(encrypted).digest()
        last_byte = encrypted[-1]
        round_number += 1
    return result[:32]


def internal_bytes_mod_3(value: bytes) -> int:
    return sum(byte % 3 for byte in value) % 3


def internal_saslprep(data: str) -> str:
    in_table_c12 = stringprep.in_table_c12
    in_table_b1 = stringprep.in_table_b1
    data = "".join(
        "\u0020" if in_table_c12(char) else char for char in data if not in_table_b1(char)
    )
    data = unicodedata.ucd_3_2_0.normalize("NFKC", data)
    if not data:
        return data

    prohibited = internal_SASLPREP_PROHIBITED
    in_table_d1 = stringprep.in_table_d1
    if in_table_d1(data[0]):
        if not in_table_d1(data[-1]):
            raise ValueError("SASLprep: failed bidirectional check")
        prohibited = (*prohibited, stringprep.in_table_d2)
    else:
        prohibited = (*prohibited, in_table_d1)

    for char in data:
        if any(in_table(char) for in_table in prohibited):
            raise ValueError("SASLprep: failed prohibited character check")
    return data


def internal_supported_revisions(version: int) -> tuple[int, ...] | None:
    # ISO 32000-1:2008, Tables 20-21 define V=1/R=2-or-3, V=2/R=3,
    # and V=4/R=4; they explicitly prohibit the unpublished V=3 algorithm.
    # Adobe Supplement to ISO 32000, BaseVersion 1.7, ExtensionLevel 3,
    # June 2008, Tables 3.18-3.19 add V=5/R=5. ISO 32000-2:2020,
    # Tables 20-21 use V=5/R=6. ISO/TS 32003:2023, Tables 2-4 add V=6/R=7
    # for AESV4 (AES-GCM), retaining the revision-6 password algorithms.
    match version:
        case 1:
            return (2, 3)
        case 2:
            return (3,)
        case 4:
            return (4,)
        case 5:
            return (5, 6)
        case 6:
            return (7,)
        case _:
            return None


def internal_required_int(params: PdfDict, key: str) -> int:
    raw_value = params.get(key)
    if raw_value is None:
        raise ValueError(f"missing encryption dictionary value {key}")
    return internal_parse_int(raw_value, key)


def internal_required_bytes(params: PdfDict, key: str, length: int) -> bytes:
    raw_value = params.get(key, MISSING)
    if raw_value is MISSING or raw_value is None:
        raise ValueError(f"missing encryption dictionary value {key}")
    try:
        value = coerce_to_bytes(raw_value)
    except TypeError as exc:
        raise ValueError(f"invalid encryption dictionary value {key}") from exc
    if len(value) != length:
        raise ValueError(f"invalid encryption dictionary value {key}: expected {length} bytes")
    return value


def internal_parse_int(value: object, field_name: str) -> int:
    parsed = parse_int(value, None)
    if parsed is None:
        raise ValueError(f"invalid encryption dictionary value {field_name}")
    return parsed


def internal_name(value: object) -> str:
    return normalize_pdf_name(value, "") or ""


__all__ = ("create_standard_decipher", "create_standard_security_handler")

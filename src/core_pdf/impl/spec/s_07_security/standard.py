# SPDX-License-Identifier: AGPL-3.0-only
"""PDF Standard Security parsing, authentication, and object decryption."""

from __future__ import annotations

import stringprep
import struct
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
    internal_rc4_crypt,
)
from core_pdf.impl.spec.s_07_syntax.types import Decipher, PdfDict
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    coerce_to_bytes,
    is_pdf_null,
    normalize_pdf_name,
    parse_int,
)

internal_CryptMethod = Literal["V2", "AESV2", "AESV3"]

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
    crypt_filters: Mapping[str, internal_CryptMethod]
    owner_encrypted_key: bytes
    user_encrypted_key: bytes
    encrypted_permissions: bytes


@dataclass(frozen=True, slots=True)
class internal_StandardSecurityHandler:
    config: internal_StandardSecurityConfig
    file_key: bytes

    def decrypt(
        self,
        object_number: int,
        generation_number: int,
        data: bytes,
        attrs: PdfDict | None = None,
        name: str | None = None,
    ) -> bytes:
        config = self.config
        if config.version <= 3:
            return internal_rc4_crypt(self.object_key(object_number, generation_number), data)

        if not config.encrypt_metadata and attrs is not None:
            object_type = normalize_pdf_name(attrs.get("Type"))
            if object_type == "Metadata":
                return data

        if name is None:
            name = (
                internal_stream_crypt_filter_name(attrs, config.stream_filter)
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

    def object_key(
        self,
        object_number: int,
        generation_number: int,
        extra: bytes = b"",
    ) -> bytes:
        seed = (
            self.file_key
            + struct.pack("<L", object_number)[:3]
            + struct.pack("<L", generation_number)[:2]
            + extra
        )
        return md5(seed).digest()[: min(len(seed), 16)]


def create_standard_decipher(
    document_id: Sequence[object],
    params: PdfDict,
    password: str = "",
) -> Decipher:
    """Validate a Standard Security dictionary and return its object decipher."""
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
    return internal_StandardSecurityHandler(config, file_key).decrypt


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

    # Entry sizes come from these version-specific definitions:
    # - ISO 32000-1:2008, Table 21: O/U are 32 bytes through revision 4.
    # - Adobe Supplement to ISO 32000, BaseVersion 1.7, ExtensionLevel 3,
    #   June 2008, Table 3.19: R5 uses 48-byte O/U, 32-byte OE/UE, and
    #   a 16-byte Perms entry.
    # - ISO 32000-2:2020, Table 21 and 7.6.4.3.3: the same sizes apply to R6.
    entry_length = 32 if revision <= 4 else 48
    owner_entry = internal_required_bytes(params, "O", entry_length)
    user_entry = internal_required_bytes(params, "U", entry_length)
    first_document_id = coerce_to_bytes(document_id[0]) if document_id else b""

    raw_length = params.get("Length", MISSING)
    length_bits = 40 if raw_length is MISSING else internal_parse_int(raw_length, "Length")
    if version <= 3 and length_bits not in (40, 128):
        raise ValueError(
            f"unsupported legacy RC4 key length for the cryptography backend: {length_bits}"
        )

    encrypt_metadata = True
    stream_filter = "Identity"
    string_filter = "Identity"
    crypt_filters: Mapping[str, internal_CryptMethod] = MappingProxyType({})
    owner_encrypted_key = b""
    user_encrypted_key = b""
    encrypted_permissions = b""

    if version >= 4:
        length_bits = 128 if version == 4 else 256
        encrypt_metadata, stream_filter, string_filter, crypt_filters = (
            internal_parse_crypt_filters(params, version)
        )
    if version >= 5:
        owner_encrypted_key = internal_required_bytes(params, "OE", 32)
        user_encrypted_key = internal_required_bytes(params, "UE", 32)
        encrypted_permissions = internal_required_bytes(params, "Perms", 16)

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
        crypt_filters=crypt_filters,
        owner_encrypted_key=owner_encrypted_key,
        user_encrypted_key=user_encrypted_key,
        encrypted_permissions=encrypted_permissions,
    )


def internal_parse_crypt_filters(
    params: PdfDict,
    version: int,
) -> tuple[bool, str, str, Mapping[str, internal_CryptMethod]]:
    raw_filters = params.get("CF")
    if raw_filters is None:
        filters: PdfDict = {}
    elif isinstance(raw_filters, dict):
        filters = cast(PdfDict, raw_filters)
    else:
        raise ValueError("invalid crypt filter dictionary")

    match version:
        case 4:
            allowed_methods = {"V2", "AESV2"}
        case 5 | 6:
            allowed_methods = {"AESV3"}
        case _:
            raise ValueError(f"crypt filters are not defined for V={version}")
    crypt_filters: dict[str, internal_CryptMethod] = {}
    for raw_name, raw_config in filters.items():
        if not isinstance(raw_config, dict):
            raise ValueError(f"invalid crypt filter dictionary: {raw_name!r}")
        method_name = internal_name(raw_config.get("CFM") or "")
        if method_name not in allowed_methods:
            raise ValueError(f"unknown crypt filter method: {method_name}")
        crypt_filters[internal_name(raw_name)] = cast(internal_CryptMethod, method_name)

    raw_stream_filter = params.get("StmF", MISSING)
    stream_filter = internal_name("Identity" if raw_stream_filter is MISSING else raw_stream_filter)
    raw_string_filter = params.get("StrF", MISSING)
    string_filter = internal_name("Identity" if raw_string_filter is MISSING else raw_string_filter)
    if string_filter != "Identity" and string_filter not in crypt_filters:
        raise ValueError(f"undefined string crypt filter: {string_filter}")

    encrypt_metadata = params.get("EncryptMetadata", MISSING)
    if encrypt_metadata is MISSING:
        encrypt_metadata = True
    if type(encrypt_metadata) is not bool:
        raise ValueError("invalid encryption metadata flag")
    return (
        encrypt_metadata,
        stream_filter,
        string_filter,
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
        case 5 | 6:
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
    # Algorithm 13 (R6).
    decrypted = internal_aes_ecb_decrypt(file_key, config.encrypted_permissions)
    metadata_flag = b"T" if config.encrypt_metadata else b"F"
    expected = struct.pack("<L", config.permissions) + (b"\xff" * 4) + metadata_flag + b"adb"
    return compare_digest(decrypted[:12], expected)


def internal_normalize_password(password: str, revision: int) -> bytes:
    if revision == 6 and password:
        password = internal_saslprep(password)
    return password.encode("utf-8")[:127]


def internal_password_hash(
    revision: int,
    password: bytes,
    salt: bytes,
    vector: bytes | None = None,
) -> bytes:
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
    match version:
        case 1 | 2 | 3:
            return (2, 3)
        case 4:
            return (4,)
        case 5 | 6:
            return (5, 6)
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


__all__ = ("create_standard_decipher",)

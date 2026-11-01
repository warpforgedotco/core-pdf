# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 7.6: the values and object encryption that writing an encrypted file needs."""

from __future__ import annotations

import secrets
import struct
from copy import replace
from typing import Any, ClassVar

from core_pdf_crypto.ciphers import aes_cbc_encrypt, aes_ecb_encrypt
from core_pdf_spec.s_07_security.standard import (
    PDF_MAC_PERMISSION_MASK,
    REVISION_3_PERMISSION_BITS,
    CryptMethod,
    StandardSecurityConfig,
    StandardSecurityHandler,
    compute_password_hash,
    legacy_file_key,
    legacy_owner_key,
    legacy_password_bytes,
    legacy_user_entry,
    normalize_password,
    pad_password,
    rc4_cascade,
    rc4_crypt,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax_primitives.coercion import decoded_name
from core_pdf_spec.types import PdfName, PdfReference, PdfString
from core_records import GeneratedRecord

# Name: (V, R, key bits, crypt filter method), with None for the V < 4 implicit RC4.
ALGORITHMS: dict[str, tuple[int, int, int, CryptMethod | None]] = {
    "RC4-40": (1, 2, 40, None),
    "RC4-128": (2, 3, 128, None),
    "AES-128": (4, 4, 128, "AESV2"),
    "AES-256-R5": (5, 5, 256, "AESV3"),
    "AES-256": (5, 6, 256, "AESV3"),
}
ALL_PERMISSIONS = -4
SIGNATURE_TYPES = ("Sig", "DocTimeStamp")


class NewEncryption(GeneratedRecord):
    """A request to encrypt a saved file. The passwords stay out of its repr."""

    __repr_fields__: ClassVar[tuple[str, ...] | None] = ("algorithm", "permissions")

    user_password: str | bytes
    owner_password: str | bytes | None = None
    algorithm: str = "AES-256"
    permissions: int = ALL_PERMISSIONS
    encrypt_metadata: bool = True


def create_security_handler(request: NewEncryption, document_id: bytes) -> StandardSecurityHandler:
    """Make a fresh file key and the /O, /U and related entries that unlock it."""
    if request.algorithm not in ALGORITHMS:
        raise ValueError(f"Algorithm '{request.algorithm}' NOT supported")
    version, revision, length_bits, method = ALGORITHMS[request.algorithm]
    permissions = output_permissions(request.permissions, version, revision)
    owner_password = request.owner_password or request.user_password
    crypt_filters = {} if method is None else {"StdCF": method}
    filter_name = "Identity" if method is None else "StdCF"
    config = StandardSecurityConfig(
        version=version,
        revision=revision,
        permissions=permissions,
        owner_entry=b"",
        user_entry=b"",
        length_bits=length_bits,
        document_id=document_id,
        encrypt_metadata=request.encrypt_metadata,
        stream_filter=filter_name,
        string_filter=filter_name,
        embedded_file_filter=filter_name,
        crypt_filters=crypt_filters,
        owner_encrypted_key=b"",
        user_encrypted_key=b"",
        encrypted_permissions=b"",
    )
    if revision <= 4:
        return legacy_handler(config, request.user_password, owner_password)
    return modern_handler(config, request.user_password, owner_password)


def output_permissions(value: int, version: int, revision: int) -> int:
    if not -(1 << 31) <= value <= (1 << 32) - 1:
        raise ValueError("encryption permissions are outside the 32-bit range")
    permissions = value & 0xFFFFFFFC
    if revision == 2 and any(
        not permissions & (1 << (bit - 1)) for bit in REVISION_3_PERMISSION_BITS
    ):
        raise ValueError("RC4-40 cannot deny the permissions in bits 9 to 12; use RC4-128 or AES")
    # A clear bit 13 would declare a PDF MAC, and this writer does not make one.
    return permissions | PDF_MAC_PERMISSION_MASK if version >= 5 else permissions


def legacy_handler(
    config: StandardSecurityConfig,
    user_password: str | bytes,
    owner_password: str | bytes,
) -> StandardSecurityHandler:
    user_bytes = legacy_password_bytes(user_password)
    owner_key = legacy_owner_key(
        legacy_password_bytes(owner_password), config.revision, config.length_bits
    )
    padded = pad_password(user_bytes)
    if config.revision == 2:
        owner_entry = rc4_crypt(owner_key, padded)
    else:
        owner_entry = rc4_cascade(owner_key, padded, range(20))
    config = replace(config, owner_entry=owner_entry)
    file_key = legacy_file_key(config, user_bytes)
    return StandardSecurityHandler(
        replace(config, user_entry=legacy_user_entry(config, file_key)), file_key
    )


def modern_handler(
    config: StandardSecurityConfig,
    user_password: str | bytes,
    owner_password: str | bytes,
) -> StandardSecurityHandler:
    file_key = secrets.token_bytes(32)
    values = modern_values(
        config.revision,
        normalize_password(user_password, config.revision),
        normalize_password(owner_password, config.revision),
        file_key,
        config.permissions,
        encrypt_metadata=config.encrypt_metadata,
    )
    config = replace(
        config,
        owner_entry=values["/O"],
        user_entry=values["/U"],
        owner_encrypted_key=values["/OE"],
        user_encrypted_key=values["/UE"],
        encrypted_permissions=values["/Perms"],
    )
    return StandardSecurityHandler(config, file_key)


def modern_values(
    revision: int,
    user_password: bytes,
    owner_password: bytes,
    file_key: bytes,
    permissions: int,
    *,
    encrypt_metadata: bool,
) -> dict[str, bytes]:
    """Algorithms 8, 9 and 10: the entries of an AES-256 encryption dictionary."""
    user_entry, user_encrypted_key = wrap_key(revision, user_password, file_key, b"")
    owner_entry, owner_encrypted_key = wrap_key(revision, owner_password, file_key, user_entry)
    return {
        "/O": owner_entry,
        "/U": user_entry,
        "/OE": owner_encrypted_key,
        "/UE": user_encrypted_key,
        "/Perms": encrypted_permissions(file_key, permissions, encrypt_metadata=encrypt_metadata),
    }


def wrap_key(revision: int, password: bytes, file_key: bytes, vector: bytes) -> tuple[bytes, bytes]:
    salts = secrets.token_bytes(16)
    validation_salt, key_salt = salts[:8], salts[8:]
    hidden_vector = vector or None
    entry = compute_password_hash(revision, password, validation_salt, hidden_vector) + salts
    key = compute_password_hash(revision, password, key_salt, hidden_vector)
    return entry, aes_cbc_encrypt(key, bytes(16), file_key, use_padding=False)


def encrypted_permissions(file_key: bytes, permissions: int, *, encrypt_metadata: bool) -> bytes:
    flag = b"T" if encrypt_metadata else b"F"
    block = struct.pack("<L", permissions) + b"\xff" * 4 + flag + b"adb" + secrets.token_bytes(4)
    return aes_ecb_encrypt(file_key, block)


def encryption_dictionary(config: StandardSecurityConfig) -> dict[Any, Any]:
    signed_permissions = config.permissions - (1 << 32) * (config.permissions >= 1 << 31)
    dictionary: dict[Any, Any] = {
        "Filter": PdfName.of("Standard"),
        "V": config.version,
        "R": config.revision,
        "Length": config.length_bits,
        "P": signed_permissions,
        "O": PdfString(config.owner_entry, is_literal=False),
        "U": PdfString(config.user_entry, is_literal=False),
    }
    if config.version >= 4:
        dictionary["CF"] = {
            "StdCF": {
                "AuthEvent": PdfName.of("DocOpen"),
                "CFM": PdfName.of(config.crypt_filters["StdCF"]),
                "Length": config.length_bits // 8,
            }
        }
        dictionary["StmF"] = PdfName.of("StdCF")
        dictionary["StrF"] = PdfName.of("StdCF")
        if not config.encrypt_metadata:
            dictionary["EncryptMetadata"] = False
    if config.version >= 5:
        dictionary["OE"] = PdfString(config.owner_encrypted_key, is_literal=False)
        dictionary["UE"] = PdfString(config.user_encrypted_key, is_literal=False)
        dictionary["Perms"] = PdfString(config.encrypted_permissions, is_literal=False)
    return dictionary


def describe_encryption(config: StandardSecurityConfig) -> str:
    """Name a file's encryption for a report, using the names NewEncryption takes."""
    for name, (version, revision, length_bits, _) in ALGORITHMS.items():
        if (config.version, config.revision, config.length_bits) == (
            version,
            revision,
            length_bits,
        ):
            return name
    if config.version == 6:
        return "AES-256-GCM"
    return f"RC4-{config.length_bits}"


def encrypt_object(handler: StandardSecurityHandler, reference: PdfReference, value: Any) -> Any:
    """Return `value` with its strings and stream data encrypted for `reference`.

    This mirrors what the lexer decrypts. An xref stream and the hex /Contents of a signature
    dictionary stay in the clear, and references, numbers and names are never encrypted.
    """
    number, generation = reference.object_number, reference.generation_number

    def sealed(item: Any) -> Any:
        if isinstance(item, PdfString):
            return PdfString(
                handler.encrypt(number, generation, item.data), is_literal=item.is_literal
            )
        if isinstance(item, (bytes, bytearray, memoryview)):
            return PdfString(handler.encrypt(number, generation, bytes(item)), is_literal=True)
        if isinstance(item, dict):
            return sealed_dictionary(item)
        if isinstance(item, (list, tuple)):
            return [sealed(element) for element in item]
        return item

    def sealed_dictionary(dictionary: dict[Any, Any]) -> dict[Any, Any]:
        result = {}
        for key, item in dictionary.items():
            keep = key == "Contents" and is_signature_contents(dictionary, item)
            result[key] = item if keep else sealed(item)
        return result

    if isinstance(value, PdfStream):
        if decoded_name(value.dictionary.get("Type")) == "XRef":
            return value
        data = handler.encrypt(number, generation, bytes(value.raw_data), value.dictionary)
        return value.replace(dictionary=sealed_dictionary(value.dictionary), raw_data=data)
    return sealed(value)


def is_signature_contents(dictionary: dict[Any, Any], contents: object) -> bool:
    if not isinstance(contents, PdfString) or contents.is_literal is not False:
        return False
    signature_type = dictionary.get("Type")
    if signature_type is None:
        return "ByteRange" in dictionary
    return decoded_name(signature_type) in SIGNATURE_TYPES


__all__ = (
    "ALGORITHMS",
    "ALL_PERMISSIONS",
    "NewEncryption",
    "create_security_handler",
    "describe_encryption",
    "encrypt_object",
    "encryption_dictionary",
    "modern_values",
)

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import stringprep
import struct
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from hashlib import md5, sha256, sha384, sha512
from hmac import compare_digest
from types import MappingProxyType
from typing import Any, ClassVar, Literal, NoReturn, Self, cast

from core_pdf_spec.exceptions import PdfDecryptionError, PdfParseError, PdfUnsupportedError
from core_pdf_spec.s_07_filters.decode_spec import normalize_stream_decode_spec
from core_pdf_spec.s_07_security.ciphers import (
    aes_cbc_decrypt,
    aes_cbc_encrypt,
    aes_ecb_decrypt,
    aes_gcm_decrypt,
    rc4_crypt,
)
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    coerce_to_bytes,
    decoded_name,
    is_pdf_null,
    require_pdf_integer,
)
from core_pdf_spec.types import MISSING

frozen_setattr = object.__setattr__


CryptMethod = Literal["V2", "AESV2", "AESV3", "AESV4"]

PASSWORD_PADDING = (
    b"\x28\xbf\x4e\x5e\x4e\x75\x8a\x41\x64\x00\x4e\x56\xff\xfa\x01\x08"
    b"\x2e\x2e\x00\xb6\xd0\x68\x3e\x80\x2f\x0c\xa9\xfe\x64\x53\x69\x7a"
)
SASLPREP_PROHIBITED: tuple[Callable[[str], bool], ...] = (
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

SUPPORTED_RC4_KEY_BITS = (40, 56, 64, 80, 128)
REVISION_3_PERMISSION_BITS = (9, 10, 11, 12)
PDF_MAC_PERMISSION_BIT = 13
PDF_MAC_PERMISSION_MASK = 1 << (PDF_MAC_PERMISSION_BIT - 1)


class StandardSecurityConfig:
    __slots__ = (
        "version",
        "revision",
        "permissions",
        "owner_entry",
        "user_entry",
        "length_bits",
        "document_id",
        "encrypt_metadata",
        "stream_filter",
        "string_filter",
        "embedded_file_filter",
        "crypt_filters",
        "owner_encrypted_key",
        "user_encrypted_key",
        "encrypted_permissions",
        "kdf_salt",
        "pdf_mac_required",
    )

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
    crypt_filters: Mapping[str, CryptMethod]
    owner_encrypted_key: bytes
    user_encrypted_key: bytes
    encrypted_permissions: bytes
    kdf_salt: bytes | None
    pdf_mac_required: bool

    __fields__: ClassVar[tuple[str, ...]] = (
        "version",
        "revision",
        "permissions",
        "owner_entry",
        "user_entry",
        "length_bits",
        "document_id",
        "encrypt_metadata",
        "stream_filter",
        "string_filter",
        "embedded_file_filter",
        "crypt_filters",
        "owner_encrypted_key",
        "user_encrypted_key",
        "encrypted_permissions",
        "kdf_salt",
        "pdf_mac_required",
    )
    __match_args__ = (
        "version",
        "revision",
        "permissions",
        "owner_entry",
        "user_entry",
        "length_bits",
        "document_id",
        "encrypt_metadata",
        "stream_filter",
        "string_filter",
        "embedded_file_filter",
        "crypt_filters",
        "owner_encrypted_key",
        "user_encrypted_key",
        "encrypted_permissions",
        "kdf_salt",
        "pdf_mac_required",
    )

    def __init__(
        self,
        version: int,
        revision: int,
        permissions: int,
        owner_entry: bytes,
        user_entry: bytes,
        length_bits: int,
        document_id: bytes,
        encrypt_metadata: bool,
        stream_filter: str,
        string_filter: str,
        embedded_file_filter: str,
        crypt_filters: Mapping[str, CryptMethod],
        owner_encrypted_key: bytes,
        user_encrypted_key: bytes,
        encrypted_permissions: bytes,
        kdf_salt: bytes | None = None,
        pdf_mac_required: bool = False,
    ) -> None:
        frozen_setattr(self, "version", version)
        frozen_setattr(self, "revision", revision)
        frozen_setattr(self, "permissions", permissions)
        frozen_setattr(self, "owner_entry", owner_entry)
        frozen_setattr(self, "user_entry", user_entry)
        frozen_setattr(self, "length_bits", length_bits)
        frozen_setattr(self, "document_id", document_id)
        frozen_setattr(self, "encrypt_metadata", encrypt_metadata)
        frozen_setattr(self, "stream_filter", stream_filter)
        frozen_setattr(self, "string_filter", string_filter)
        frozen_setattr(self, "embedded_file_filter", embedded_file_filter)
        frozen_setattr(self, "crypt_filters", crypt_filters)
        frozen_setattr(self, "owner_encrypted_key", owner_encrypted_key)
        frozen_setattr(self, "user_encrypted_key", user_encrypted_key)
        frozen_setattr(self, "encrypted_permissions", encrypted_permissions)
        frozen_setattr(self, "kdf_salt", kdf_salt)
        frozen_setattr(self, "pdf_mac_required", pdf_mac_required)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"version={self.version!r}, "
            f"revision={self.revision!r}, "
            f"permissions={self.permissions!r}, "
            f"owner_entry={self.owner_entry!r}, "
            f"user_entry={self.user_entry!r}, "
            f"length_bits={self.length_bits!r}, "
            f"document_id={self.document_id!r}, "
            f"encrypt_metadata={self.encrypt_metadata!r}, "
            f"stream_filter={self.stream_filter!r}, "
            f"string_filter={self.string_filter!r}, "
            f"embedded_file_filter={self.embedded_file_filter!r}, "
            f"crypt_filters={self.crypt_filters!r}, "
            f"owner_encrypted_key={self.owner_encrypted_key!r}, "
            f"user_encrypted_key={self.user_encrypted_key!r}, "
            f"encrypted_permissions={self.encrypted_permissions!r}, "
            f"kdf_salt={self.kdf_salt!r}, "
            f"pdf_mac_required={self.pdf_mac_required!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.version == other.version
            and self.revision == other.revision
            and self.permissions == other.permissions
            and self.owner_entry == other.owner_entry
            and self.user_entry == other.user_entry
            and self.length_bits == other.length_bits
            and self.document_id == other.document_id
            and self.encrypt_metadata == other.encrypt_metadata
            and self.stream_filter == other.stream_filter
            and self.string_filter == other.string_filter
            and self.embedded_file_filter == other.embedded_file_filter
            and self.crypt_filters == other.crypt_filters
            and self.owner_encrypted_key == other.owner_encrypted_key
            and self.user_encrypted_key == other.user_encrypted_key
            and self.encrypted_permissions == other.encrypted_permissions
            and self.kdf_salt == other.kdf_salt
            and self.pdf_mac_required == other.pdf_mac_required
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.version,
                self.revision,
                self.permissions,
                self.owner_entry,
                self.user_entry,
                self.length_bits,
                self.document_id,
                self.encrypt_metadata,
                self.stream_filter,
                self.string_filter,
                self.embedded_file_filter,
                self.crypt_filters,
                self.owner_encrypted_key,
                self.user_encrypted_key,
                self.encrypted_permissions,
                self.kdf_salt,
                self.pdf_mac_required,
            )
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        version = changes.pop("version", self.version)
        revision = changes.pop("revision", self.revision)
        permissions = changes.pop("permissions", self.permissions)
        owner_entry = changes.pop("owner_entry", self.owner_entry)
        user_entry = changes.pop("user_entry", self.user_entry)
        length_bits = changes.pop("length_bits", self.length_bits)
        document_id = changes.pop("document_id", self.document_id)
        encrypt_metadata = changes.pop("encrypt_metadata", self.encrypt_metadata)
        stream_filter = changes.pop("stream_filter", self.stream_filter)
        string_filter = changes.pop("string_filter", self.string_filter)
        embedded_file_filter = changes.pop("embedded_file_filter", self.embedded_file_filter)
        crypt_filters = changes.pop("crypt_filters", self.crypt_filters)
        owner_encrypted_key = changes.pop("owner_encrypted_key", self.owner_encrypted_key)
        user_encrypted_key = changes.pop("user_encrypted_key", self.user_encrypted_key)
        encrypted_permissions = changes.pop("encrypted_permissions", self.encrypted_permissions)
        kdf_salt = changes.pop("kdf_salt", self.kdf_salt)
        pdf_mac_required = changes.pop("pdf_mac_required", self.pdf_mac_required)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            version,
            revision,
            permissions,
            owner_entry,
            user_entry,
            length_bits,
            document_id,
            encrypt_metadata,
            stream_filter,
            string_filter,
            embedded_file_filter,
            crypt_filters,
            owner_encrypted_key,
            user_encrypted_key,
            encrypted_permissions,
            kdf_salt,
            pdf_mac_required,
        )


class StandardSecurityHandler:
    __slots__ = ("config", "file_key")

    config: StandardSecurityConfig
    file_key: bytes

    __fields__: ClassVar[tuple[str, ...]] = ("config", "file_key")
    __match_args__ = ("config", "file_key")

    def __init__(self, config: StandardSecurityConfig, file_key: bytes) -> None:
        frozen_setattr(self, "config", config)
        frozen_setattr(self, "file_key", file_key)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(config={self.config!r}, file_key={self.file_key!r})"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.config == other.config and self.file_key == other.file_key

    def __hash__(self) -> int:
        return hash((self.config, self.file_key))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        config = changes.pop("config", self.config)
        file_key = changes.pop("file_key", self.file_key)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(config, file_key)

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
            return rc4_crypt(self.object_key(object_number, generation_number), data)

        default_stream_filter = config.stream_filter
        if attrs is not None:
            object_type = decoded_name(attrs.get("Type"))
            if not config.encrypt_metadata and object_type == "Metadata":
                return data
            if object_type == "EmbeddedFile":
                default_stream_filter = config.embedded_file_filter

        if name is None:
            name = (
                stream_crypt_filter_name(attrs, default_stream_filter)
                if attrs is not None
                else config.string_filter
            )
        method = resolve_crypt_method(name, config.crypt_filters)
        match method:
            case None:
                return data
            case "V2":
                return rc4_crypt(self.object_key(object_number, generation_number), data)
            case "AESV2":
                key = self.object_key(object_number, generation_number, b"sAlT")
                return aes_cbc_decrypt(
                    key,
                    data[:16],
                    data[16:],
                    use_padding=True,
                )
            case "AESV3":
                return aes_cbc_decrypt(
                    self.file_key,
                    data[:16],
                    data[16:],
                    use_padding=True,
                )
            case "AESV4":
                return aes_gcm_decrypt(self.file_key, data)

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


def create_standard_security_handler(
    document_id: Sequence[object],
    params: PdfDict,
    password: str = "",
) -> StandardSecurityHandler:
    filter_name = decoded_name(params.get("Filter"))
    if filter_name is None:
        raise PdfUnsupportedError("Invalid encryption dictionary")
    if filter_name in {"Adobe.PubSec", "PubSec"}:
        raise PdfUnsupportedError("Public-key encryption is not supported")
    if filter_name != "Standard":
        raise PdfUnsupportedError(f"Unsupported encryption filter: {filter_name}")

    try:
        version = require_pdf_integer(params.get("V"))
    except ValueError as exc:
        raise PdfUnsupportedError("Invalid encryption dictionary") from exc
    supported_revisions = supported_revisions_for_version(version)
    if supported_revisions is None:
        raise PdfUnsupportedError(f"Unsupported standard encryption algorithm V={version}")

    try:
        config = parse_config(document_id, params, version, supported_revisions)
    except (TypeError, ValueError) as exc:
        raise PdfUnsupportedError("Invalid encryption dictionary") from exc
    file_key = authenticate(config, password)
    if file_key is None:
        raise PdfUnsupportedError("Incorrect password")
    if config.revision >= 5 and not validate_permissions(config, file_key):
        raise PdfDecryptionError("Invalid encryption permissions")
    return StandardSecurityHandler(config, file_key)


def parse_config(
    document_id: Sequence[object],
    params: PdfDict,
    version: int,
    supported_revisions: tuple[int, ...],
) -> StandardSecurityConfig:
    revision = required_int(params, "R")
    if revision not in supported_revisions:
        raise ValueError(f"unsupported Standard Security revision R={revision} for V={version}")

    raw_permissions = params.get("P", MISSING)
    if raw_permissions is MISSING or raw_permissions is None:
        raise ValueError("missing encryption permissions")
    permissions = parse_int(raw_permissions, "P")
    if not -(1 << 31) <= permissions <= (1 << 32) - 1:
        raise ValueError("encryption permissions are outside the 32-bit range")
    if permissions < 0:
        permissions += 1 << 32

    pdf_mac_supported_version = version in (5, 6)
    pdf_mac_required = pdf_mac_supported_version and not (permissions & PDF_MAC_PERMISSION_MASK)

    if version == 1 and revision == 2:
        revision_3_required = any(
            permissions & (1 << (bit_position - 1)) == 0
            for bit_position in REVISION_3_PERMISSION_BITS
        )
        if revision_3_required:
            raise ValueError(
                "Standard Security V=1 permissions require R=3 when any "
                "revision-3 permission is cleared, got R=2"
            )

    entry_length = 32 if revision <= 4 else 48
    owner_entry = required_bytes(params, "O", entry_length)
    user_entry = required_bytes(params, "U", entry_length)
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
        length_bits = parse_int(raw_length, "Length")

    match version:
        case 1:
            if length_bits != 40:
                raise ValueError(f"invalid V=1 encryption key length: {length_bits}")
        case 2:
            if length_bits not in SUPPORTED_RC4_KEY_BITS:
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
    crypt_filters: Mapping[str, CryptMethod] = MappingProxyType({})
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
        ) = parse_crypt_filters(params, version)
    if version in (5, 6):
        owner_encrypted_key = required_bytes(params, "OE", 32)
        user_encrypted_key = required_bytes(params, "UE", 32)
        encrypted_permissions = required_bytes(params, "Perms", 16)

        raw_kdf_salt = params.get("KDFSalt", MISSING)
        if raw_kdf_salt is not MISSING:
            kdf_salt = required_bytes(params, "KDFSalt", 32)
        if pdf_mac_required and kdf_salt is None:
            raise ValueError("PDF MAC requires a 32-byte KDFSalt")
    elif params.get("KDFSalt", MISSING) is not MISSING:
        raise ValueError("KDFSalt requires encryption algorithm V >= 5")

    return StandardSecurityConfig(
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


def parse_crypt_filters(
    params: PdfDict,
    version: int,
) -> tuple[bool, str, str, str, Mapping[str, CryptMethod]]:
    raw_filters = params.get("CF", MISSING)
    if raw_filters is MISSING:
        filters: PdfDict = {}
    elif isinstance(raw_filters, dict):
        filters = raw_filters
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
    crypt_filters: dict[str, CryptMethod] = {}
    for raw_name, raw_config in filters.items():
        filter_name = name(raw_name)
        if not filter_name:
            raise ValueError("invalid crypt filter name")
        if filter_name == "Identity":
            continue
        if filter_name != "StdCF":
            raise ValueError(f"unsupported Standard Security crypt filter: {filter_name}")
        if not isinstance(raw_config, dict):
            raise ValueError(f"invalid crypt filter dictionary: {raw_name!r}")
        filter_config = raw_config

        raw_type = filter_config.get("Type", MISSING)
        if raw_type is not MISSING and name(raw_type) != "CryptFilter":
            raise ValueError(f"invalid crypt filter type: {filter_name}")

        method_name = name(filter_config.get("CFM", "None"))
        if method_name not in allowed_methods:
            raise ValueError(f"unknown crypt filter method: {method_name}")

        auth_event = name(filter_config.get("AuthEvent", "DocOpen"))
        if auth_event != "DocOpen":
            raise ValueError(f"unsupported Standard Security authorization event: {auth_event}")

        raw_filter_length = filter_config.get("Length", MISSING)
        expected_filter_length = 32 if method_name in {"AESV3", "AESV4"} else 16
        if raw_filter_length is not MISSING:
            filter_length = parse_int(raw_filter_length, "CF/Length")
            if filter_length != expected_filter_length:
                raise ValueError(f"invalid {method_name} crypt filter length: {filter_length}")

        crypt_filters[filter_name] = cast(CryptMethod, method_name)

    if version == 6 and "AESV4" not in crypt_filters.values():
        raise ValueError("V=6 requires at least one AESV4 crypt filter")

    raw_stream_filter = params.get("StmF", MISSING)
    stream_filter = name("Identity" if raw_stream_filter is MISSING else raw_stream_filter)
    raw_string_filter = params.get("StrF", MISSING)
    string_filter = name("Identity" if raw_string_filter is MISSING else raw_string_filter)
    raw_embedded_file_filter = params.get("EFF", MISSING)
    embedded_file_filter = name(
        stream_filter if raw_embedded_file_filter is MISSING else raw_embedded_file_filter
    )

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


def stream_crypt_filter_name(attrs: PdfDict, default_filter: str) -> str:
    spec = normalize_stream_decode_spec(attrs)
    crypt_indexes = [index for index, step in enumerate(spec.steps) if step.name == "Crypt"]
    if not crypt_indexes:
        return default_filter
    if len(crypt_indexes) != 1 or crypt_indexes[0] != 0:
        raise PdfParseError("Crypt must be the first and only Crypt stream filter")

    params = spec.steps[crypt_indexes[0]].params
    if is_pdf_null(params):
        return "Identity"
    if not isinstance(params, dict):
        raise PdfParseError("invalid Crypt filter params")
    raw_name = params.get("Name")
    if is_pdf_null(raw_name):
        return "Identity"
    filter_name = decoded_name(raw_name)
    if filter_name is None:
        raise PdfParseError("invalid Crypt filter name")
    return filter_name


def resolve_crypt_method(
    name: str,
    crypt_filters: Mapping[str, CryptMethod],
) -> CryptMethod | None:
    if name == "Identity":
        return None
    method = crypt_filters.get(name)
    if method is None:
        raise PdfUnsupportedError(f"Undefined crypt filter: {name}")
    return method


def authenticate(
    config: StandardSecurityConfig,
    password: str,
) -> bytes | None:
    match config.revision:
        case 2 | 3 | 4:
            return authenticate_legacy(config, password)
        case 5 | 6 | 7:
            return authenticate_modern(config, password)
        case _:
            raise ValueError(f"unsupported Standard Security revision R={config.revision}")


def authenticate_legacy(
    config: StandardSecurityConfig,
    password: str,
) -> bytes | None:
    password_bytes = password.encode("latin-1")
    key = authenticate_legacy_user(config, password_bytes)
    if key is not None:
        return key

    digest = md5(pad_password(password_bytes)).digest()
    key_length = 5
    if config.revision >= 3:
        digest = md5_50_rounds(digest)
        key_length = config.length_bits // 8
    owner_key = digest[:key_length]
    if config.revision == 2:
        user_password = rc4_crypt(owner_key, config.owner_entry)
    else:
        user_password = rc4_cascade(
            owner_key,
            config.owner_entry,
            range(19, -1, -1),
        )
    return authenticate_legacy_user(config, user_password)


def authenticate_legacy_user(
    config: StandardSecurityConfig,
    password: bytes,
) -> bytes | None:
    key = legacy_file_key(config, password)
    expected = legacy_user_entry(config, key)
    if config.revision == 2:
        valid = compare_digest(expected, config.user_entry)
    else:
        valid = compare_digest(expected[:16], config.user_entry[:16])
    return key if valid else None


def legacy_file_key(
    config: StandardSecurityConfig,
    password: bytes,
) -> bytes:
    digest = md5(pad_password(password))
    digest.update(config.owner_entry)
    digest.update(struct.pack("<L", config.permissions))
    digest.update(config.document_id)
    if config.revision >= 4 and not config.encrypt_metadata:
        digest.update(b"\xff\xff\xff\xff")
    result = digest.digest()
    key_length = 5
    if config.revision >= 3:
        key_length = config.length_bits // 8
        result = md5_50_rounds(result, key_length)
    return result[:key_length]


def legacy_user_entry(
    config: StandardSecurityConfig,
    key: bytes,
) -> bytes:
    if config.revision == 2:
        return rc4_crypt(key, PASSWORD_PADDING)
    digest = md5(PASSWORD_PADDING)
    digest.update(config.document_id)
    result = rc4_crypt(key, digest.digest())
    result = rc4_cascade(key, result, range(1, 20))
    return result + result


def pad_password(password: bytes) -> bytes:
    return (password + PASSWORD_PADDING)[:32]


def md5_50_rounds(digest: bytes, keep: int = 16) -> bytes:
    for _ in range(50):
        digest = md5(digest[:keep]).digest()
    return digest


def rc4_cascade(key: bytes, data: bytes, indexes: range) -> bytes:
    for index in indexes:
        data = rc4_crypt(bytes(byte ^ index for byte in key), data)
    return data


def authenticate_modern(
    config: StandardSecurityConfig,
    password: str,
) -> bytes | None:
    password_bytes = normalize_password(password, config.revision)
    owner_hash = config.owner_entry[:32]
    owner_validation_salt = config.owner_entry[32:40]
    owner_key_salt = config.owner_entry[40:]
    user_hash = config.user_entry[:32]
    user_validation_salt = config.user_entry[32:40]
    user_key_salt = config.user_entry[40:]

    password_hash = compute_password_hash(
        config.revision,
        password_bytes,
        owner_validation_salt,
        config.user_entry,
    )
    if compare_digest(password_hash, owner_hash):
        password_hash = compute_password_hash(
            config.revision,
            password_bytes,
            owner_key_salt,
            config.user_entry,
        )
        return aes_cbc_decrypt(
            password_hash,
            bytes(16),
            config.owner_encrypted_key,
            use_padding=False,
        )

    password_hash = compute_password_hash(
        config.revision,
        password_bytes,
        user_validation_salt,
    )
    if compare_digest(password_hash, user_hash):
        password_hash = compute_password_hash(
            config.revision,
            password_bytes,
            user_key_salt,
        )
        return aes_cbc_decrypt(
            password_hash,
            bytes(16),
            config.user_encrypted_key,
            use_padding=False,
        )
    return None


def validate_permissions(
    config: StandardSecurityConfig,
    file_key: bytes,
) -> bool:
    decrypted = aes_ecb_decrypt(file_key, config.encrypted_permissions)
    metadata_flag = b"T" if config.encrypt_metadata else b"F"
    expected = struct.pack("<L", config.permissions) + (b"\xff" * 4) + metadata_flag + b"adb"
    return compare_digest(decrypted[:12], expected)


def normalize_password(password: str, revision: int) -> bytes:
    if revision in (6, 7) and password:
        password = saslprep(password)
    return password.encode("utf-8")[:127]


def compute_password_hash(
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
    return r6_password_hash(password, salt[:8], vector)


def r6_password_hash(
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
        encrypted = aes_cbc_encrypt(
            result[:16],
            result[16:32],
            repeated,
            use_padding=False,
        )
        next_hash = hashes[bytes_mod_3(encrypted[:16])]
        result = next_hash(encrypted).digest()
        last_byte = encrypted[-1]
        round_number += 1
    return result[:32]


def bytes_mod_3(value: bytes) -> int:
    return sum(byte % 3 for byte in value) % 3


def saslprep(data: str) -> str:
    in_table_c12 = stringprep.in_table_c12
    in_table_b1 = stringprep.in_table_b1
    data = "".join(
        "\u0020" if in_table_c12(char) else char for char in data if not in_table_b1(char)
    )
    data = unicodedata.ucd_3_2_0.normalize("NFKC", data)
    if not data:
        return data

    prohibited = SASLPREP_PROHIBITED
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


def supported_revisions_for_version(version: int) -> tuple[int, ...] | None:
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


def required_int(params: PdfDict, key: str) -> int:
    raw_value = params.get(key)
    if raw_value is None:
        raise ValueError(f"missing encryption dictionary value {key}")
    return parse_int(raw_value, key)


def required_bytes(params: PdfDict, key: str, length: int) -> bytes:
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


def parse_int(value: object, field_name: str) -> int:
    return require_pdf_integer(value, f"invalid encryption dictionary value {field_name}")


def name(value: object) -> str:
    return decoded_name(value, "") or ""


__all__ = (
    "StandardSecurityHandler",
    "StandardSecurityConfig",
    "create_standard_security_handler",
)

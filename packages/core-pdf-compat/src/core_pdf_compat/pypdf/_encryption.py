"""pypdf's encryption shapes. The algorithms are in core_pdf.impl.spec.s_07_security."""

from __future__ import annotations

import logging
from enum import Enum, IntEnum
from typing import Any

from core_pdf.impl.spec.s_07_security.standard import StandardSecurityHandler, saslprep
lazy from core_pdf.impl.spec.s_07_security.output import (
    ALGORITHMS,
    NewEncryption,
    create_security_handler,
    encryption_dictionary,
    modern_values,
)

from ._crypt_providers import CryptAES, CryptBase, CryptIdentity, CryptRC4
from .errors import PdfReadError
from .generic import (
    ArrayObject,
    BooleanObject,
    ByteStringObject,
    DictionaryObject,
    NullObject,
    TextStringObject,
)

logger = logging.getLogger(__name__)
_saslprep = saslprep


class PasswordType(IntEnum):
    NOT_DECRYPTED = 0
    USER_PASSWORD = 1
    OWNER_PASSWORD = 2


class EncryptAlgorithm(tuple, Enum):
    # V, R, Length
    RC4_40 = (1, 2, 40)
    RC4_128 = (2, 3, 128)
    AES_128 = (4, 4, 128)
    AES_256_R5 = (5, 5, 256)
    AES_256 = (5, 6, 256)


class AlgV5:
    @staticmethod
    def generate_values(
        R: int,  # noqa: N803
        user_password: bytes,
        owner_password: bytes,
        key: bytes,
        p: int,
        metadata_encrypted: bool,
    ) -> dict[str, bytes]:
        return modern_values(
            R,
            user_password[:127],
            owner_password[:127],
            key,
            p & 0xFFFFFFFF,
            encrypt_metadata=metadata_encrypted,
        )


class Encryption:
    """The writer's side of pypdf's class: an algorithm and the handler made from passwords."""

    def __init__(self, algorithm: EncryptAlgorithm, permissions: int, first_id: bytes) -> None:
        self.V, self.R, self.Length = algorithm.value
        self.P = permissions
        self.id1_entry = first_id
        self.algorithm = algorithm
        self.handler: StandardSecurityHandler | None = None

    def __repr__(self) -> str:
        return f"Encryption(V={self.V}, R={self.R})"

    @staticmethod
    def make(alg: EncryptAlgorithm, permissions: int, first_id_entry: bytes) -> Encryption:
        return Encryption(alg, permissions, first_id_entry)

    def _encode_password(self, password: bytes | str, *, strict: bool = False) -> bytes:
        if isinstance(password, bytes):
            return password
        if self.V < 5:
            try:
                return password.encode("latin-1")
            except UnicodeEncodeError:
                return password.encode("utf-8")
        try:
            password = saslprep(password)
        except (ValueError, IndexError) as error:
            if strict:
                raise ValueError(f"Password SASLprep normalization failed: {error}") from error
            logger.warning("SASLprep normalization failed, using plain UTF-8: %s", error)
        return password.encode("utf-8")

    def write_entry(
        self,
        user_password: bytes | str,
        owner_password: bytes | str | None,
        *,
        strict: bool = False,
    ) -> dict[Any, Any]:
        """Make the handler and return the encryption dictionary, as core objects."""
        user = self._encode_password(user_password, strict=strict)
        owner = self._encode_password(owner_password, strict=strict) if owner_password else user
        name = next(key for key, value in ALGORITHMS.items() if value[:3] == self.algorithm.value)
        permissions = int(self.P)
        request = NewEncryption(user, owner, name, permissions)
        self.handler = create_security_handler(request, self.id1_entry)
        return encryption_dictionary(self.handler.config)


# Ported from pypdf 7.0.0's Encryption.read and its helpers, under pypdf's BSD-style
# license. core-pdf decrypts; these are the checks pypdf makes of the /Encrypt dictionary
# when a reader opens, with its errors and warnings.
def _get_entry(
    dictionary: Any,
    key: str,
    *,
    expected_types: tuple[type, ...],
    type_name: str,
    default: Any = None,
    required: bool = False,
) -> Any:
    """An entry of the encryption dictionary, null being absent."""
    value = dictionary.get(key, NullObject()).get_object()
    if isinstance(value, NullObject):
        if required:
            raise PdfReadError(f"{key} entry of the encryption dictionary is missing")
        return default
    if not isinstance(value, expected_types):
        raise PdfReadError(f"{key} entry of the encryption dictionary is not a {type_name}")
    return value


def _get_crypt_filter(filters: Any, name: str) -> Any:
    crypt_filter = filters.get(name, NullObject()).get_object()
    if not isinstance(crypt_filter, DictionaryObject):
        raise PdfReadError(f"Crypt filter {name} is missing from /CF or not a dictionary")
    return crypt_filter


def check_encryption_entry(entry: Any) -> None:
    """Check an /Encrypt dictionary as pypdf reads it."""
    if entry.get("/Filter") != "/Standard":
        raise NotImplementedError("only Standard PDF encryption handler is available")
    if "/SubFilter" in entry:
        raise NotImplementedError("/SubFilter NOT supported")
    version = entry.get("/V", 0)
    if version not in (1, 2, 3, 4, 5):
        raise NotImplementedError(f"Encryption V={version} NOT supported")
    name, number, string = (str,), (int, float), (ByteStringObject, TextStringObject)
    stream_crypt_filter = None
    stream_method = "/V2"
    if version >= 4:
        # /CF is needed only to look up named crypt filters, which default to /Identity.
        filters = entry.get("/CF", DictionaryObject()).get_object()
        if not isinstance(filters, DictionaryObject):
            raise PdfReadError("/CF entry of the encryption dictionary is not a dictionary")
        stream_filter = _get_entry(
            entry, "/StmF", expected_types=name, type_name="name", default="/Identity"
        )
        string_filter = _get_entry(
            entry, "/StrF", expected_types=name, type_name="name", default="/Identity"
        )
        file_filter = _get_entry(
            entry, "/EFF", expected_types=name, type_name="name", default=stream_filter
        )
        methods = []
        for index, filter_name in enumerate((stream_filter, string_filter, file_filter)):
            method = filter_name
            if filter_name != "/Identity":
                crypt_filter = _get_crypt_filter(filters, filter_name)
                if index == 0:
                    stream_crypt_filter = crypt_filter
                method = _get_entry(
                    crypt_filter, "/CFM", expected_types=name, type_name="name", default="/None"
                )
            methods.append(method)
        stream_method = methods[0]
        allowed = ("/Identity", "/V2", "/AESV2", "/AESV3")
        for label, method in zip(("StmF", "StrF", "EFF"), methods, strict=True):
            if method not in allowed:
                raise NotImplementedError(f"{label} Method {method} NOT supported!")
    _get_entry(entry, "/R", expected_types=number, type_name="number", required=True)
    _get_entry(entry, "/P", expected_types=number, type_name="number", required=True)
    _get_entry(entry, "/Length", expected_types=number, type_name="number", default=40)
    if version == 4 and stream_method == "/AESV2" and stream_crypt_filter is not None:
        _get_entry(
            stream_crypt_filter, "/Length", expected_types=number, type_name="number", default=16
        )
    encrypt_metadata = entry.get("/EncryptMetadata", NullObject()).get_object()
    if not isinstance(encrypt_metadata, (BooleanObject, NullObject)):
        logger.warning("Ignoring non-boolean /EncryptMetadata entry of the encryption dictionary")
    for key in ("/O", "/U"):
        _get_entry(entry, key, expected_types=string, type_name="string", required=True)
    for key in ("/OE", "/UE", "/Perms"):
        _get_entry(entry, key, expected_types=string, type_name="string")


def encrypts_metadata(entry: Any) -> bool:
    """/EncryptMetadata as pypdf reads it: anything but a boolean is true."""
    value = entry.get("/EncryptMetadata", NullObject()).get_object()
    return value.value if isinstance(value, BooleanObject) else True


def check_file_identifier(identifier: Any) -> bool:
    """Warn, as pypdf does, about a trailer /ID it cannot take its first entry from.

    Whether the /ID is usable; pypdf uses an empty identifier in place of a malformed one.
    """
    if isinstance(identifier, ArrayObject) and identifier:
        if isinstance(identifier[0].get_object(), (ByteStringObject, TextStringObject)):
            return True
        logger.warning("Ignoring non-string first /ID entry of the trailer")
        return False
    if not isinstance(identifier, NullObject):
        logger.warning("Ignoring malformed /ID entry of the trailer")
        return False
    return True


__all__ = (
    "AlgV5",
    "CryptAES",
    "CryptBase",
    "CryptIdentity",
    "CryptRC4",
    "EncryptAlgorithm",
    "Encryption",
    "PasswordType",
    "_saslprep",
)

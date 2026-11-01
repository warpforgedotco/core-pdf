"""pypdf's encryption shapes. The algorithms are in core_pdf_spec.s_07_security."""

from __future__ import annotations

import logging
from enum import Enum, IntEnum
from typing import Any

from core_pdf_spec.s_07_security.output import (
    ALGORITHMS,
    NewEncryption,
    create_security_handler,
    encryption_dictionary,
    modern_values,
)
from core_pdf_spec.s_07_security.standard import StandardSecurityHandler, saslprep

from ._crypt_providers import CryptAES, CryptBase, CryptIdentity, CryptRC4

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

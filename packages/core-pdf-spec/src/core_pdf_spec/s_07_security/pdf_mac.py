# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core_pdf_crypto.errors import UnsupportedAlgorithmError
from core_pdf_crypto.pdf_mac import validate_pdf_mac_token
from core_pdf_spec.exceptions import PdfDecryptionError, PdfUnsupportedError
from core_pdf_spec.s_07_security.standard import StandardSecurityHandler
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.types import MISSING, PdfByteBuffer, PdfName, PdfString

HEXADECIMAL_DIGITS = frozenset(b"0123456789ABCDEFabcdef")


def validate_pdf_mac_if_present(
    raw_data: PdfByteBuffer,
    trailer: PdfDict,
    handler: StandardSecurityHandler,
) -> bool:
    raw_auth_code = trailer.get("AuthCode", MISSING)
    has_auth_code = raw_auth_code is not MISSING
    has_kdf_salt = handler.config.kdf_salt is not None

    if not (handler.config.pdf_mac_required or has_auth_code or has_kdf_salt):
        return False
    if handler.config.version < 5:
        raise PdfDecryptionError("Invalid PDF MAC")
    if not isinstance(raw_auth_code, dict) or handler.config.kdf_salt is None:
        raise PdfDecryptionError("Invalid PDF MAC")

    try:
        validate_standalone_pdf_mac(
            raw_data,
            raw_auth_code,
            handler.file_key,
            handler.config.kdf_salt,
        )
    except UnsupportedAlgorithmError as exc:
        raise PdfUnsupportedError(str(exc)) from exc
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        raise PdfDecryptionError("Invalid PDF MAC") from exc
    return True


def validate_pdf_mac_extension(declarations: object) -> None:
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


def validate_standalone_pdf_mac(
    raw_data: PdfByteBuffer,
    auth_code: PdfDict,
    file_key: bytes,
    kdf_salt: bytes,
) -> None:
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

    byte_range, token = extract_standalone_token(raw_data, auth_code)
    validate_pdf_mac_token(raw_data, byte_range, token, file_key, kdf_salt)


def extract_standalone_token(
    raw_data: PdfByteBuffer,
    auth_code: PdfDict,
) -> tuple[tuple[int, int, int, int], bytes]:
    raw_byte_range = auth_code.get("ByteRange", MISSING)
    if not isinstance(raw_byte_range, list) or len(raw_byte_range) != 4:
        raise ValueError("invalid PDF MAC ByteRange")
    if any(type(value) is not int or value < 0 for value in raw_byte_range):
        raise ValueError("invalid PDF MAC ByteRange")
    # The loop above has already rejected anything that is not a non-negative
    # int, so the annotation records what was checked rather than asserting it.
    byte_range: tuple[int, int, int, int] = tuple(raw_byte_range)  # type: ignore[assignment]
    first_start, first_length, second_start, second_length = byte_range

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

    serialized_mac = bytes(raw_data[first_length:second_start])
    encoded_token = serialized_mac[1:-1]
    if (
        len(serialized_mac) != (2 * len(raw_mac.data)) + 2
        or not serialized_mac.startswith(b"<")
        or not serialized_mac.endswith(b">")
        or any(byte not in HEXADECIMAL_DIGITS for byte in encoded_token)
        or bytes.fromhex(encoded_token.decode("ascii")) != raw_mac.data
    ):
        raise ValueError("invalid serialized standalone PDF MAC")

    return (
        first_start,
        first_length,
        second_start,
        second_length,
    ), raw_mac.data


__all__ = (
    "validate_pdf_mac_if_present",
    "validate_pdf_mac_extension",
)

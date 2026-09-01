#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Generate the ISO/TS 32004:2024 PDF MAC interoperability fixtures.

pyHanko is an offline fixture generator and independent validation oracle, not
a project or test dependency. The fixed byte sources below make the committed
artifacts reproducible; predictable keys, salts, and nonces are safe only
because these PDFs contain public test data.

The generator also proves that pyHanko accepts each pristine fixture with both
passwords and rejects deterministic changes to the covered document bytes, MAC
value, KDF salt, byte range, and file extent. Corrupt variants are constructed
in memory rather than committed because each mutation is fully described by
the test suite.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import secrets
from dataclasses import asdict, dataclass
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

internal_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
internal_SECURITY_FIXTURES = internal_REPOSITORY_ROOT / "tests" / "fixtures" / "security_interop"
internal_DEFAULT_SOURCE = internal_SECURITY_FIXTURES / "source.pdf"
internal_DEFAULT_OUTPUT = internal_SECURITY_FIXTURES / "pdf_mac"
internal_MANIFEST_NAME = "manifest.json"
internal_PYHANKO_VERSION = "0.37.0"
internal_PYHANKO_COMMIT = "00362ec2772b2d39e5d9ba2c0287efb4077421d8"
internal_SOURCE_SHA256 = "f06eccd62dc412f8774af0562441b88953ceb2ab705d4a8ecf212a17e8a2a851"
internal_EXPECTED_TEXT = b"Security Interoperability"
internal_BASE_RANDOM_SEED = b"core-pdf ISO/TS 32004:2024 fixture v1:"
internal_BYTE_RANGE_PATTERN = re.compile(rb"/ByteRange\s*\[\s*0\s+(\d+)\s+(\d+)\s+(\d+)\s*\]")
internal_TAMPER_CHECKS = (
    "covered-document-byte",
    "mac-byte",
    "kdf-salt",
    "byte-range",
    "truncated-file",
    "trailing-file-bytes",
)


@dataclass(frozen=True, slots=True)
class internal_FixtureSpec:
    filename: str
    algorithm: str
    crypt_filter_method: str
    version: int
    revision: int
    owner_password: str
    user_password: str
    use_gcm: bool


internal_FIXTURES = (
    internal_FixtureSpec(
        filename="aes-256-r6-cbc-mac.pdf",
        algorithm="AES-256-CBC",
        crypt_filter_method="AESV3",
        version=5,
        revision=6,
        owner_password="owner-mac-cbc",
        user_password="user-mac-cbc",
        use_gcm=False,
    ),
    internal_FixtureSpec(
        filename="aes-256-r7-gcm-mac.pdf",
        algorithm="AES-256-GCM",
        crypt_filter_method="AESV4",
        version=6,
        revision=7,
        owner_password="owner-mac-gcm",
        user_password="user-mac-gcm",
        use_gcm=True,
    ),
)


class internal_DeterministicTokenBytes:
    """Provide distinct reproducible bytes to pyHanko's fixture-only writer."""

    def __init__(self, seed: bytes) -> None:
        self.seed = seed
        self.counter = 0

    def __call__(self, length: int) -> bytes:
        output = bytearray()
        while len(output) < length:
            output.extend(hashlib.sha256(self.seed + self.counter.to_bytes(8, "big")).digest())
            self.counter += 1
        return bytes(output[:length])


def internal_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def internal_pyhanko() -> tuple[Any, Any, Any]:
    try:
        installed_version = metadata.version("pyHanko")
    except metadata.PackageNotFoundError as exc:
        raise SystemExit(
            "pyHanko is required only to regenerate these fixtures; run with "
            f"uv run --with pyhanko=={internal_PYHANKO_VERSION}"
        ) from exc
    if installed_version != internal_PYHANKO_VERSION:
        raise SystemExit(f"expected pyHanko {internal_PYHANKO_VERSION}, got {installed_version}")
    writer = cast(Any, import_module("pyhanko.pdf_utils.writer"))
    reader = cast(Any, import_module("pyhanko.pdf_utils.reader"))
    generic = cast(Any, import_module("pyhanko.pdf_utils.generic"))
    return writer, reader.PdfFileReader, generic


def internal_hex_entry_bounds(data: bytes, key: bytes) -> tuple[int, int]:
    marker = b"/" + key + b" <"
    start = data.index(marker) + len(marker)
    return start, data.index(b">", start)


def internal_change_hex_digit(data: bytes, key: bytes, *, from_end: bool = False) -> bytes:
    start, end = internal_hex_entry_bounds(data, key)
    index = end - 1 if from_end else start
    corrupted = bytearray(data)
    corrupted[index] = ord("0") if corrupted[index] != ord("0") else ord("1")
    return bytes(corrupted)


def internal_change_covered_document_byte(data: bytes) -> bytes:
    second_comment = data.index(b"\n%", len(b"%PDF-")) + 2
    corrupted = bytearray(data)
    corrupted[second_comment] ^= 1
    return bytes(corrupted)


def internal_change_byte_range(data: bytes) -> bytes:
    match = internal_BYTE_RANGE_PATTERN.search(data)
    if match is None:
        raise ValueError("fixture has no standalone PDF MAC byte range")
    old_length = match.group(3)
    new_length = str(int(old_length) - 1).encode("ascii")
    if len(new_length) != len(old_length):
        raise ValueError("fixture byte-range mutation would change its serialized width")
    return data[: match.start(3)] + new_length + data[match.end(3) :]


def internal_tampered_variants(data: bytes) -> dict[str, bytes]:
    return {
        "covered-document-byte": internal_change_covered_document_byte(data),
        "mac-byte": internal_change_hex_digit(data, b"MAC", from_end=True),
        "kdf-salt": internal_change_hex_digit(data, b"KDFSalt"),
        "byte-range": internal_change_byte_range(data),
        # Removing only the final line feed leaves a parseable PDF while making
        # the ISO/TS 32004:2024, 6.5.1 whole-file coverage check fail.
        "truncated-file": data[:-1],
        "trailing-file-bytes": data + b"% PDF MAC coverage tamper\n",
    }


def internal_authenticate(data: bytes, password: str, pdf_reader: Any) -> tuple[Any, Any]:
    source = io.BytesIO(data)
    reader = pdf_reader(source)
    return reader, reader.decrypt(password)


def internal_verify_valid_fixture(
    destination: Path,
    fixture: internal_FixtureSpec,
    pdf_reader: Any,
    generic: Any,
) -> None:
    data = destination.read_bytes()
    for password, expected_status in (
        (fixture.user_password, "USER"),
        (fixture.owner_password, "OWNER"),
    ):
        reader, authentication = internal_authenticate(data, password, pdf_reader)
        if authentication.status.name != expected_status:
            raise ValueError(
                f"pyHanko could not authenticate {fixture.filename} as {expected_status}"
            )
        if authentication.mac_status.name != "SUCCESSFUL":
            raise ValueError(f"pyHanko did not validate the PDF MAC in {fixture.filename}")
        page = reader.root["/Pages"]["/Kids"][0]
        if internal_EXPECTED_TEXT not in page["/Contents"].data:
            raise ValueError(f"{fixture.filename} content did not round-trip")

        encryption = reader.trailer_view["/Encrypt"]
        for key, expected in {
            "/V": fixture.version,
            "/R": fixture.revision,
            "/P": -4100,
        }.items():
            if int(encryption[key]) != expected:
                raise ValueError(f"unexpected {fixture.filename} encryption entry {key}")
        if str(encryption["/CF"]["/StdCF"]["/CFM"]) != f"/{fixture.crypt_filter_method}":
            raise ValueError(f"{fixture.filename} selects the wrong crypt filter")
        if len(encryption["/KDFSalt"].original_bytes) != 32:
            raise ValueError(f"{fixture.filename} does not contain a 32-byte KDFSalt")

        auth_code = reader.trailer_view.raw_get("/AuthCode")
        if not isinstance(auth_code, generic.DictionaryObject):
            raise ValueError(f"{fixture.filename} AuthCode is not a direct dictionary")
        if str(auth_code["/MACLocation"]) != "/Standalone":
            raise ValueError(f"{fixture.filename} is not a standalone PDF MAC fixture")

        extension_records = {
            (
                int(extension.get_object()["/ExtensionLevel"]),
                str(extension.get_object()["/ExtensionRevision"]),
            )
            for extension in reader.root["/Extensions"]["/ISO_"]
        }
        expected_extensions = {(32004, ":2024")}
        if fixture.use_gcm:
            expected_extensions.add((32003, ":2023"))
        if extension_records != expected_extensions:
            raise ValueError(f"{fixture.filename} declares the wrong ISO extensions")


def internal_verify_tamper_rejection(
    destination: Path,
    fixture: internal_FixtureSpec,
    pdf_reader: Any,
) -> None:
    data = destination.read_bytes()
    variants = internal_tampered_variants(data)
    if variants.keys() != dict.fromkeys(internal_TAMPER_CHECKS).keys():
        raise AssertionError("tamper-check manifest and generator disagree")
    reader_logger = logging.getLogger("pyhanko.pdf_utils.reader")
    logging_was_disabled = reader_logger.disabled
    reader_logger.disabled = True
    try:
        for name, corrupted in variants.items():
            try:
                _, authentication = internal_authenticate(
                    corrupted,
                    fixture.user_password,
                    pdf_reader,
                )
            except Exception:  # noqa: BLE001 - rejection can happen while parsing or validating
                continue
            if authentication.status.name != "FAILED" or authentication.mac_status.name != "FAILED":
                raise ValueError(f"pyHanko accepted {name} corruption in {fixture.filename}")
    finally:
        reader_logger.disabled = logging_was_disabled


def internal_generate_fixture(
    source: Path,
    destination: Path,
    fixture: internal_FixtureSpec,
    writer: Any,
    pdf_reader: Any,
    generic: Any,
) -> None:
    deterministic_bytes = internal_DeterministicTokenBytes(
        internal_BASE_RANDOM_SEED + fixture.filename.encode("ascii")
    )
    with (
        patch.object(os, "urandom", deterministic_bytes),
        patch.object(secrets, "token_bytes", deterministic_bytes),
        source.open("rb") as input_stream,
        destination.open("wb") as output_stream,
    ):
        output = writer.copy_into_new_writer(pdf_reader(input_stream))
        # pyHanko v0.37.0 is the independent ISO/TS 32004:2024 producer and
        # validator. It creates the PDF and CMS structures; core-pdf only reads
        # the resulting committed bytes in its tests.
        output.encrypt(
            fixture.owner_password,
            fixture.user_password,
            pdf_mac=True,
            use_gcm=fixture.use_gcm,
        )
        # Preserve the deliberately minimal XMP packet byte-for-byte. Its XML
        # declaration is legal PDF metadata but pyHanko's optional updater
        # expects an XML fragment when it receives an already-decoded string.
        output._update_meta = lambda: None
        output.write(output_stream)

    internal_verify_valid_fixture(destination, fixture, pdf_reader, generic)
    internal_verify_tamper_rejection(destination, fixture, pdf_reader)


def internal_generate(source: Path, output_directory: Path) -> None:
    if internal_sha256(source) != internal_SOURCE_SHA256:
        raise ValueError("security fixture source has changed")
    writer, pdf_reader, generic = internal_pyhanko()
    output_directory.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for fixture in internal_FIXTURES:
        destination = output_directory / fixture.filename
        internal_generate_fixture(source, destination, fixture, writer, pdf_reader, generic)
        record = asdict(fixture)
        record.pop("use_gcm")
        record.update(
            {
                "kdf_salt_bytes": 32,
                "mac_algorithm": "HMAC-SHA-256",
                "mac_digest_algorithm": "SHA-256",
                "mac_key_wrap_algorithm": "AES-256-KW",
                "mac_location": "Standalone",
                "pdf_mac": True,
                "sha256": internal_sha256(destination),
                "tamper_checks": list(internal_TAMPER_CHECKS),
            }
        )
        records.append(record)

    manifest = {
        "expected": {
            "text": internal_EXPECTED_TEXT.decode("ascii"),
        },
        "fixtures": records,
        "generator": {
            "commit": internal_PYHANKO_COMMIT,
            "license": "MIT",
            "name": "pyHanko",
            "version": internal_PYHANKO_VERSION,
            "website": "https://github.com/MatthiasValvekens/pyHanko",
        },
        "regenerate": [
            "uv",
            "run",
            "--with",
            f"pyhanko=={internal_PYHANKO_VERSION}",
            "python",
            "scripts/generate_pdf_mac_interop_fixtures.py",
        ],
        "source": {
            "filename": "../source.pdf",
            "sha256": internal_SOURCE_SHA256,
        },
        "specification": "ISO/TS 32004:2024",
        "warning": "Deterministic cryptographic material is for public test fixtures only.",
    }
    (output_directory / internal_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def internal_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=internal_DEFAULT_SOURCE,
        help="plaintext source PDF",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=internal_DEFAULT_OUTPUT,
        help="output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = internal_parse_args()
    internal_generate(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

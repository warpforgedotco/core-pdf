#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import secrets
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, ClassVar, NoReturn, Self, cast
from unittest.mock import patch

frozen_setattr = object.__setattr__


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SECURITY_FIXTURES = REPOSITORY_ROOT / "tests" / "fixtures" / "security_interop"
DEFAULT_SOURCE = SECURITY_FIXTURES / "source.pdf"
DEFAULT_OUTPUT = SECURITY_FIXTURES / "pdf_mac"
MANIFEST_NAME = "manifest.json"
PYHANKO_VERSION = "0.37.0"
PYHANKO_COMMIT = "00362ec2772b2d39e5d9ba2c0287efb4077421d8"
SOURCE_SHA256 = "f06eccd62dc412f8774af0562441b88953ceb2ab705d4a8ecf212a17e8a2a851"
EXPECTED_TEXT = b"Security Interoperability"
BASE_RANDOM_SEED = b"core-pdf ISO/TS 32004:2024 fixture v1:"
BYTE_RANGE_PATTERN = re.compile(rb"/ByteRange\s*\[\s*0\s+(\d+)\s+(\d+)\s+(\d+)\s*\]")
TAMPER_CHECKS = (
    "covered-document-byte",
    "mac-byte",
    "kdf-salt",
    "byte-range",
    "truncated-file",
    "trailing-file-bytes",
)


class FixtureSpec:
    __slots__ = (
        "filename",
        "algorithm",
        "crypt_filter_method",
        "version",
        "revision",
        "owner_password",
        "user_password",
        "use_gcm",
    )

    filename: str
    algorithm: str
    crypt_filter_method: str
    version: int
    revision: int
    owner_password: str
    user_password: str
    use_gcm: bool

    __fields__: ClassVar[tuple[str, ...]] = (
        "filename",
        "algorithm",
        "crypt_filter_method",
        "version",
        "revision",
        "owner_password",
        "user_password",
        "use_gcm",
    )
    __match_args__ = (
        "filename",
        "algorithm",
        "crypt_filter_method",
        "version",
        "revision",
        "owner_password",
        "user_password",
        "use_gcm",
    )

    def __init__(
        self,
        filename: str,
        algorithm: str,
        crypt_filter_method: str,
        version: int,
        revision: int,
        owner_password: str,
        user_password: str,
        use_gcm: bool,
    ) -> None:
        frozen_setattr(self, "filename", filename)
        frozen_setattr(self, "algorithm", algorithm)
        frozen_setattr(self, "crypt_filter_method", crypt_filter_method)
        frozen_setattr(self, "version", version)
        frozen_setattr(self, "revision", revision)
        frozen_setattr(self, "owner_password", owner_password)
        frozen_setattr(self, "user_password", user_password)
        frozen_setattr(self, "use_gcm", use_gcm)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"filename={self.filename!r}, "
            f"algorithm={self.algorithm!r}, "
            f"crypt_filter_method={self.crypt_filter_method!r}, "
            f"version={self.version!r}, "
            f"revision={self.revision!r}, "
            f"owner_password={self.owner_password!r}, "
            f"user_password={self.user_password!r}, "
            f"use_gcm={self.use_gcm!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.filename == other.filename
            and self.algorithm == other.algorithm
            and self.crypt_filter_method == other.crypt_filter_method
            and self.version == other.version
            and self.revision == other.revision
            and self.owner_password == other.owner_password
            and self.user_password == other.user_password
            and self.use_gcm == other.use_gcm
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.filename,
                self.algorithm,
                self.crypt_filter_method,
                self.version,
                self.revision,
                self.owner_password,
                self.user_password,
                self.use_gcm,
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
        filename = changes.pop("filename", self.filename)
        algorithm = changes.pop("algorithm", self.algorithm)
        crypt_filter_method = changes.pop("crypt_filter_method", self.crypt_filter_method)
        version = changes.pop("version", self.version)
        revision = changes.pop("revision", self.revision)
        owner_password = changes.pop("owner_password", self.owner_password)
        user_password = changes.pop("user_password", self.user_password)
        use_gcm = changes.pop("use_gcm", self.use_gcm)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            filename,
            algorithm,
            crypt_filter_method,
            version,
            revision,
            owner_password,
            user_password,
            use_gcm,
        )


FIXTURES = (
    FixtureSpec(
        filename="aes-256-r6-cbc-mac.pdf",
        algorithm="AES-256-CBC",
        crypt_filter_method="AESV3",
        version=5,
        revision=6,
        owner_password="owner-mac-cbc",
        user_password="user-mac-cbc",
        use_gcm=False,
    ),
    FixtureSpec(
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


class DeterministicTokenBytes:
    def __init__(self, seed: bytes) -> None:
        self.seed = seed
        self.counter = 0

    def __call__(self, length: int) -> bytes:
        output = bytearray()
        while len(output) < length:
            output.extend(hashlib.sha256(self.seed + self.counter.to_bytes(8, "big")).digest())
            self.counter += 1
        return bytes(output[:length])


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def pyhanko() -> tuple[Any, Any, Any]:
    try:
        installed_version = metadata.version("pyHanko")
    except metadata.PackageNotFoundError as exc:
        raise SystemExit(
            "pyHanko is required only to regenerate these fixtures; run with "
            f"uv run --with pyhanko=={PYHANKO_VERSION}"
        ) from exc
    if installed_version != PYHANKO_VERSION:
        raise SystemExit(f"expected pyHanko {PYHANKO_VERSION}, got {installed_version}")
    writer = cast(Any, import_module("pyhanko.pdf_utils.writer"))
    reader = cast(Any, import_module("pyhanko.pdf_utils.reader"))
    generic = cast(Any, import_module("pyhanko.pdf_utils.generic"))
    return writer, reader.PdfFileReader, generic


def hex_entry_bounds(data: bytes, key: bytes) -> tuple[int, int]:
    marker = b"/" + key + b" <"
    start = data.index(marker) + len(marker)
    return start, data.index(b">", start)


def change_hex_digit(data: bytes, key: bytes, *, from_end: bool = False) -> bytes:
    start, end = hex_entry_bounds(data, key)
    index = end - 1 if from_end else start
    corrupted = bytearray(data)
    corrupted[index] = ord("0") if corrupted[index] != ord("0") else ord("1")
    return bytes(corrupted)


def change_covered_document_byte(data: bytes) -> bytes:
    second_comment = data.index(b"\n%", len(b"%PDF-")) + 2
    corrupted = bytearray(data)
    corrupted[second_comment] ^= 1
    return bytes(corrupted)


def change_byte_range(data: bytes) -> bytes:
    match = BYTE_RANGE_PATTERN.search(data)
    if match is None:
        raise ValueError("fixture has no standalone PDF MAC byte range")
    old_length = match.group(3)
    new_length = str(int(old_length) - 1).encode("ascii")
    if len(new_length) != len(old_length):
        raise ValueError("fixture byte-range mutation would change its serialized width")
    return data[: match.start(3)] + new_length + data[match.end(3) :]


def tampered_variants(data: bytes) -> dict[str, bytes]:
    return {
        "covered-document-byte": change_covered_document_byte(data),
        "mac-byte": change_hex_digit(data, b"MAC", from_end=True),
        "kdf-salt": change_hex_digit(data, b"KDFSalt"),
        "byte-range": change_byte_range(data),
        "truncated-file": data[:-1],
        "trailing-file-bytes": data + b"% PDF MAC coverage tamper\n",
    }


def authenticate(data: bytes, password: str, pdf_reader: Any) -> tuple[Any, Any]:
    source = io.BytesIO(data)
    reader = pdf_reader(source)
    return reader, reader.decrypt(password)


def verify_valid_fixture(
    destination: Path,
    fixture: FixtureSpec,
    pdf_reader: Any,
    generic: Any,
) -> None:
    data = destination.read_bytes()
    for password, expected_status in (
        (fixture.user_password, "USER"),
        (fixture.owner_password, "OWNER"),
    ):
        reader, authentication = authenticate(data, password, pdf_reader)
        if authentication.status.name != expected_status:
            raise ValueError(
                f"pyHanko could not authenticate {fixture.filename} as {expected_status}"
            )
        if authentication.mac_status.name != "SUCCESSFUL":
            raise ValueError(f"pyHanko did not validate the PDF MAC in {fixture.filename}")
        page = reader.root["/Pages"]["/Kids"][0]
        if EXPECTED_TEXT not in page["/Contents"].data:
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


def verify_tamper_rejection(
    destination: Path,
    fixture: FixtureSpec,
    pdf_reader: Any,
) -> None:
    data = destination.read_bytes()
    variants = tampered_variants(data)
    if variants.keys() != dict.fromkeys(TAMPER_CHECKS).keys():
        raise AssertionError("tamper-check manifest and generator disagree")
    reader_logger = logging.getLogger("pyhanko.pdf_utils.reader")
    logging_was_disabled = reader_logger.disabled
    reader_logger.disabled = True
    try:
        for name, corrupted in variants.items():
            try:
                _, authentication = authenticate(
                    corrupted,
                    fixture.user_password,
                    pdf_reader,
                )
            except Exception:  # noqa: BLE001
                continue
            if authentication.status.name != "FAILED" or authentication.mac_status.name != "FAILED":
                raise ValueError(f"pyHanko accepted {name} corruption in {fixture.filename}")
    finally:
        reader_logger.disabled = logging_was_disabled


def generate_fixture(
    source: Path,
    destination: Path,
    fixture: FixtureSpec,
    writer: Any,
    pdf_reader: Any,
    generic: Any,
) -> None:
    deterministic_bytes = DeterministicTokenBytes(
        BASE_RANDOM_SEED + fixture.filename.encode("ascii")
    )
    with (
        patch.object(os, "urandom", deterministic_bytes),
        patch.object(secrets, "token_bytes", deterministic_bytes),
        source.open("rb") as input_stream,
        destination.open("wb") as output_stream,
    ):
        output = writer.copy_into_new_writer(pdf_reader(input_stream))
        output.encrypt(
            fixture.owner_password,
            fixture.user_password,
            pdf_mac=True,
            use_gcm=fixture.use_gcm,
        )
        output._update_meta = lambda: None
        output.write(output_stream)

    verify_valid_fixture(destination, fixture, pdf_reader, generic)
    verify_tamper_rejection(destination, fixture, pdf_reader)


def generate(source: Path, output_directory: Path) -> None:
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("security fixture source has changed")
    writer, pdf_reader, generic = pyhanko()
    output_directory.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for fixture in FIXTURES:
        destination = output_directory / fixture.filename
        generate_fixture(source, destination, fixture, writer, pdf_reader, generic)
        record = {name: getattr(fixture, name) for name in fixture.__fields__}
        record.pop("use_gcm")
        record.update(
            {
                "kdf_salt_bytes": 32,
                "mac_algorithm": "HMAC-SHA-256",
                "mac_digest_algorithm": "SHA-256",
                "mac_key_wrap_algorithm": "AES-256-KW",
                "mac_location": "Standalone",
                "pdf_mac": True,
                "sha256": sha256(destination),
                "tamper_checks": list(TAMPER_CHECKS),
            }
        )
        records.append(record)

    manifest = {
        "expected": {
            "text": EXPECTED_TEXT.decode("ascii"),
        },
        "fixtures": records,
        "generator": {
            "commit": PYHANKO_COMMIT,
            "license": "MIT",
            "name": "pyHanko",
            "version": PYHANKO_VERSION,
            "website": "https://github.com/MatthiasValvekens/pyHanko",
        },
        "regenerate": [
            "uv",
            "run",
            "--with",
            f"pyhanko=={PYHANKO_VERSION}",
            "python",
            "scripts/generate_pdf_mac_interop_fixtures.py",
        ],
        "source": {
            "filename": "../source.pdf",
            "sha256": SOURCE_SHA256,
        },
        "specification": "ISO/TS 32004:2024",
        "warning": "Deterministic cryptographic material is for public test fixtures only.",
    }
    (output_directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="plaintext source PDF",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

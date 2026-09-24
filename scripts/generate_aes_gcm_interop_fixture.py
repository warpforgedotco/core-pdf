#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from importlib import import_module, metadata
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SECURITY_FIXTURES = REPOSITORY_ROOT / "tests" / "fixtures" / "security_interop"
DEFAULT_SOURCE = SECURITY_FIXTURES / "source.pdf"
DEFAULT_OUTPUT = SECURITY_FIXTURES / "aes_gcm"
FIXTURE_NAME = "aes-256-r7-gcm.pdf"
MANIFEST_NAME = "manifest.json"
PYHANKO_VERSION = "0.37.0"
PYHANKO_COMMIT = "00362ec2772b2d39e5d9ba2c0287efb4077421d8"
SOURCE_SHA256 = "f06eccd62dc412f8774af0562441b88953ceb2ab705d4a8ecf212a17e8a2a851"
USER_PASSWORD = "user-gcm"
OWNER_PASSWORD = "owner-gcm"
EXPECTED_TEXT = b"Security Interoperability"
RANDOM_SEED = b"core-pdf ISO/TS 32003:2023 fixture v1"


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


def pyhanko() -> tuple[Any, Any]:
    try:
        installed_version = metadata.version("pyHanko")
    except metadata.PackageNotFoundError as exc:
        raise SystemExit(
            "pyHanko is required only to regenerate this fixture; run with "
            f"uv run --with pyhanko=={PYHANKO_VERSION}"
        ) from exc
    if installed_version != PYHANKO_VERSION:
        raise SystemExit(f"expected pyHanko {PYHANKO_VERSION}, got {installed_version}")
    writer = import_module("pyhanko.pdf_utils.writer")
    reader = import_module("pyhanko.pdf_utils.reader")
    return writer, reader.PdfFileReader


def verify_fixture(destination: Path, pdf_reader: Any) -> None:
    with destination.open("rb") as source:
        reader = pdf_reader(source)
        authentication = reader.decrypt(USER_PASSWORD)
        if authentication.status.name != "USER":
            raise ValueError("pyHanko could not authenticate its AES-GCM fixture")
        page = reader.root["/Pages"]["/Kids"][0]
        if EXPECTED_TEXT not in page["/Contents"].data:
            raise ValueError("AES-GCM fixture content did not round-trip")
        encryption = reader.trailer_view["/Encrypt"]
        expected_entries = {
            "/V": 6,
            "/R": 7,
            "/P": -4,
        }
        for key, expected in expected_entries.items():
            if int(encryption[key]) != expected:
                raise ValueError(f"unexpected AES-GCM fixture entry {key}")
        if str(encryption["/CF"]["/StdCF"]["/CFM"]) != "/AESV4":
            raise ValueError("AES-GCM fixture does not select AESV4")
        if "/KDFSalt" in encryption:
            raise ValueError("fixture unexpectedly enables ISO/TS 32004 PDF MAC")
        extensions = reader.root["/Extensions"]["/ISO_"]
        extension = extensions[0].get_object()
        if int(extension["/ExtensionLevel"]) != 32003:
            raise ValueError("fixture does not declare ISO/TS 32003")
        if str(extension["/ExtensionRevision"]) != ":2023":
            raise ValueError("fixture has the wrong ISO/TS 32003 revision")


def generate(source: Path, output_directory: Path) -> None:
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("security fixture source has changed")
    writer, pdf_reader = pyhanko()
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / FIXTURE_NAME
    deterministic_bytes = DeterministicTokenBytes(RANDOM_SEED)

    with (
        patch.object(os, "urandom", deterministic_bytes),
        patch.object(secrets, "token_bytes", deterministic_bytes),
        source.open("rb") as input_stream,
        destination.open("wb") as output_stream,
    ):
        output = writer.copy_into_new_writer(pdf_reader(input_stream))
        output.encrypt(
            OWNER_PASSWORD,
            USER_PASSWORD,
            pdf_mac=False,
            use_gcm=True,
        )
        output._update_meta = lambda: None
        output.write(output_stream)

    verify_fixture(destination, pdf_reader)
    manifest = {
        "expected": {
            "text": EXPECTED_TEXT.decode("ascii"),
        },
        "fixture": {
            "algorithm": "AES-256-GCM",
            "crypt_filter_method": "AESV4",
            "filename": destination.name,
            "owner_password": OWNER_PASSWORD,
            "pdf_mac": False,
            "revision": 7,
            "sha256": sha256(destination),
            "user_password": USER_PASSWORD,
            "version": 6,
        },
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
            "scripts/generate_aes_gcm_interop_fixture.py",
        ],
        "source": {
            "filename": "../source.pdf",
            "sha256": SOURCE_SHA256,
        },
        "specification": "ISO/TS 32003:2023",
        "warning": "Deterministic cryptographic material is for this public test fixture only.",
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

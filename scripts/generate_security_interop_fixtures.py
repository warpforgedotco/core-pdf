#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, ClassVar, NoReturn, Self

frozen_setattr = object.__setattr__


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "tests" / "fixtures" / "security_interop"
EXPECTED_TEXT = "Security Interoperability"
EXPECTED_TITLE = "Core PDF Security Fixture"
XMP_MARKER = "security-xmp-marker"


class FixtureSpec:
    filename: str
    algorithm: str
    revision: int
    bits: int
    user_password: str
    owner_password: str
    encrypt_metadata: bool
    qpdf_options: tuple[str, ...]
    weak_crypto: bool

    __fields__: ClassVar[tuple[str, ...]] = (
        "filename",
        "algorithm",
        "revision",
        "bits",
        "user_password",
        "owner_password",
        "encrypt_metadata",
        "qpdf_options",
        "weak_crypto",
    )
    __match_args__ = (
        "filename",
        "algorithm",
        "revision",
        "bits",
        "user_password",
        "owner_password",
        "encrypt_metadata",
        "qpdf_options",
        "weak_crypto",
    )

    def __init__(
        self,
        filename: str,
        algorithm: str,
        revision: int,
        bits: int,
        user_password: str,
        owner_password: str,
        encrypt_metadata: bool = True,
        qpdf_options: tuple[str, ...] = (),
        weak_crypto: bool = False,
    ) -> None:
        frozen_setattr(self, "filename", filename)
        frozen_setattr(self, "algorithm", algorithm)
        frozen_setattr(self, "revision", revision)
        frozen_setattr(self, "bits", bits)
        frozen_setattr(self, "user_password", user_password)
        frozen_setattr(self, "owner_password", owner_password)
        frozen_setattr(self, "encrypt_metadata", encrypt_metadata)
        frozen_setattr(self, "qpdf_options", qpdf_options)
        frozen_setattr(self, "weak_crypto", weak_crypto)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"filename={self.filename!r}, "
            f"algorithm={self.algorithm!r}, "
            f"revision={self.revision!r}, "
            f"bits={self.bits!r}, "
            f"user_password={self.user_password!r}, "
            f"owner_password={self.owner_password!r}, "
            f"encrypt_metadata={self.encrypt_metadata!r}, "
            f"qpdf_options={self.qpdf_options!r}, "
            f"weak_crypto={self.weak_crypto!r}"
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
            and self.revision == other.revision
            and self.bits == other.bits
            and self.user_password == other.user_password
            and self.owner_password == other.owner_password
            and self.encrypt_metadata == other.encrypt_metadata
            and self.qpdf_options == other.qpdf_options
            and self.weak_crypto == other.weak_crypto
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.filename,
                self.algorithm,
                self.revision,
                self.bits,
                self.user_password,
                self.owner_password,
                self.encrypt_metadata,
                self.qpdf_options,
                self.weak_crypto,
            )
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __replace__(self, /, **changes: Any) -> Self:
        filename = changes.pop("filename", self.filename)
        algorithm = changes.pop("algorithm", self.algorithm)
        revision = changes.pop("revision", self.revision)
        bits = changes.pop("bits", self.bits)
        user_password = changes.pop("user_password", self.user_password)
        owner_password = changes.pop("owner_password", self.owner_password)
        encrypt_metadata = changes.pop("encrypt_metadata", self.encrypt_metadata)
        qpdf_options = changes.pop("qpdf_options", self.qpdf_options)
        weak_crypto = changes.pop("weak_crypto", self.weak_crypto)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            filename,
            algorithm,
            revision,
            bits,
            user_password,
            owner_password,
            encrypt_metadata,
            qpdf_options,
            weak_crypto,
        )


FIXTURES = (
    FixtureSpec(
        filename="rc4-40-r2.pdf",
        algorithm="RC4-40",
        revision=2,
        bits=40,
        user_password="user-40",
        owner_password="owner-40",
        weak_crypto=True,
    ),
    FixtureSpec(
        filename="rc4-128-r3.pdf",
        algorithm="RC4-128",
        revision=3,
        bits=128,
        user_password="user-128",
        owner_password="owner-128",
        qpdf_options=("--use-aes=n",),
        weak_crypto=True,
    ),
    FixtureSpec(
        filename="aes-128-r4-cleartext-metadata.pdf",
        algorithm="AES-128",
        revision=4,
        bits=128,
        user_password="user-aes128",
        owner_password="owner-aes128",
        encrypt_metadata=False,
        qpdf_options=("--use-aes=y", "--force-V4", "--cleartext-metadata"),
    ),
    FixtureSpec(
        filename="aes-256-r5.pdf",
        algorithm="AES-256",
        revision=5,
        bits=256,
        user_password="user-r5",
        owner_password="owner-r5",
        qpdf_options=("--force-R5",),
    ),
    FixtureSpec(
        filename="aes-256-r6.pdf",
        algorithm="AES-256",
        revision=6,
        bits=256,
        user_password="user-r6",
        owner_password="owner-r6",
    ),
    FixtureSpec(
        filename="aes-256-r6-blank-user.pdf",
        algorithm="AES-256",
        revision=6,
        bits=256,
        user_password="",
        owner_password="owner-blank",
    ),
)


def internal_stream(contents: bytes, *, extra_attributes: bytes = b"") -> bytes:
    attributes = b"/Length " + str(len(contents)).encode("ascii")
    if extra_attributes:
        attributes += b" " + extra_attributes
    return b"<< " + attributes + b" >>\nstream\n" + contents + b"\nendstream"


def source_pdf() -> bytes:
    page_contents = b"BT\n/F1 18 Tf\n72 720 Td\n(Security Interoperability) Tj\nET\n"
    xmp = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<xmpmeta xmlns="urn:core-pdf:test">\n'
        b"  <marker>security-xmp-marker</marker>\n"
        b"</xmpmeta>\n"
    )
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R /Metadata 7 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        internal_stream(page_contents),
        (
            b"<< /Title (Core PDF Security Fixture) /Author (core-pdf) "
            b"/Subject (qpdf interoperability) >>"
        ),
        internal_stream(xmp, extra_attributes=b"/Type /Metadata /Subtype /XML"),
    )

    output = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, value in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode())
        output.extend(value)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())

    document_id = b"636f72652d7064662d7365637572697479"
    output.extend(
        b"trailer\n"
        + f"<< /Size {len(objects) + 1} /Root 1 0 R /Info 6 0 R ".encode()
        + b"/ID [<"
        + document_id
        + b"><"
        + document_id
        + b">] >>\n"
        + b"startxref\n"
        + str(xref_offset).encode()
        + b"\n%%EOF\n"
    )
    return bytes(output)


def qpdf_command(
    qpdf: str,
    source: Path,
    destination: Path,
    fixture: FixtureSpec,
) -> list[str]:
    command = [
        qpdf,
        "--object-streams=disable",
        "--stream-data=uncompress",
    ]
    if fixture.weak_crypto:
        command.append("--allow-weak-crypto")
    command.extend(
        [
            str(source),
            str(destination),
            "--encrypt",
            f"--user-password={fixture.user_password}",
            f"--owner-password={fixture.owner_password}",
            f"--bits={fixture.bits}",
            *fixture.qpdf_options,
            "--",
        ]
    )
    return command


def display_command(fixture: FixtureSpec) -> list[str]:
    return qpdf_command(
        "qpdf",
        Path("source.pdf"),
        Path(fixture.filename),
        fixture,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(qpdf: str, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    source = output_directory / "source.pdf"
    source.write_bytes(source_pdf())

    version_result = subprocess.run(
        [qpdf, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    qpdf_version = version_result.stdout.splitlines()[0].removeprefix("qpdf version ")
    records: list[dict[str, object]] = []
    for fixture in FIXTURES:
        destination = output_directory / fixture.filename
        subprocess.run(
            qpdf_command(qpdf, source, destination, fixture),
            check=True,
        )
        subprocess.run(
            [qpdf, f"--password={fixture.user_password}", "--check", str(destination)],
            check=True,
            capture_output=True,
        )
        record = {name: getattr(fixture, name) for name in fixture.__fields__}
        record["qpdf_options"] = list(fixture.qpdf_options)
        record["command"] = display_command(fixture)
        record["sha256"] = sha256(destination)
        records.append(record)

    manifest = {
        "generator": {
            "name": "qpdf",
            "version": qpdf_version,
            "website": "https://qpdf.sourceforge.io/",
        },
        "source": {
            "filename": source.name,
            "sha256": sha256(source),
        },
        "expected": {
            "text": EXPECTED_TEXT,
            "info_title": EXPECTED_TITLE,
            "xmp_marker": XMP_MARKER,
        },
        "fixtures": records,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qpdf",
        default=shutil.which("qpdf"),
        help="qpdf executable (default: discovered on PATH)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="fixture output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.qpdf:
        raise SystemExit("qpdf was not found; install it or pass --qpdf")
    generate(args.qpdf, args.output.resolve())


if __name__ == "__main__":
    main()

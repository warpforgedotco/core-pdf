#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Generate the qpdf security interoperability fixtures.

qpdf is an offline fixture generator, not a project or test dependency. Weak
crypto is enabled only for the legacy RC4 formats that this compatibility suite
must be able to read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

internal_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
internal_DEFAULT_OUTPUT = internal_REPOSITORY_ROOT / "tests" / "fixtures" / "security_interop"
internal_EXPECTED_TEXT = "Security Interoperability"
internal_EXPECTED_TITLE = "Core PDF Security Fixture"
internal_XMP_MARKER = "security-xmp-marker"


@dataclass(frozen=True)
class internal_FixtureSpec:
    filename: str
    algorithm: str
    revision: int
    bits: int
    user_password: str
    owner_password: str
    encrypt_metadata: bool = True
    qpdf_options: tuple[str, ...] = ()
    weak_crypto: bool = False


internal_FIXTURES = (
    internal_FixtureSpec(
        filename="rc4-40-r2.pdf",
        algorithm="RC4-40",
        revision=2,
        bits=40,
        user_password="user-40",
        owner_password="owner-40",
        weak_crypto=True,
    ),
    internal_FixtureSpec(
        filename="rc4-128-r3.pdf",
        algorithm="RC4-128",
        revision=3,
        bits=128,
        user_password="user-128",
        owner_password="owner-128",
        qpdf_options=("--use-aes=n",),
        weak_crypto=True,
    ),
    internal_FixtureSpec(
        filename="aes-128-r4-cleartext-metadata.pdf",
        algorithm="AES-128",
        revision=4,
        bits=128,
        user_password="user-aes128",
        owner_password="owner-aes128",
        encrypt_metadata=False,
        qpdf_options=("--use-aes=y", "--force-V4", "--cleartext-metadata"),
    ),
    internal_FixtureSpec(
        filename="aes-256-r5.pdf",
        algorithm="AES-256",
        revision=5,
        bits=256,
        user_password="user-r5",
        owner_password="owner-r5",
        qpdf_options=("--force-R5",),
    ),
    internal_FixtureSpec(
        filename="aes-256-r6.pdf",
        algorithm="AES-256",
        revision=6,
        bits=256,
        user_password="user-r6",
        owner_password="owner-r6",
    ),
    internal_FixtureSpec(
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


def internal_source_pdf() -> bytes:
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


def internal_qpdf_command(
    qpdf: str,
    source: Path,
    destination: Path,
    fixture: internal_FixtureSpec,
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


def internal_display_command(fixture: internal_FixtureSpec) -> list[str]:
    return internal_qpdf_command(
        "qpdf",
        Path("source.pdf"),
        Path(fixture.filename),
        fixture,
    )


def internal_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def internal_generate(qpdf: str, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    source = output_directory / "source.pdf"
    source.write_bytes(internal_source_pdf())

    version_result = subprocess.run(
        [qpdf, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    qpdf_version = version_result.stdout.splitlines()[0].removeprefix("qpdf version ")
    records: list[dict[str, object]] = []
    for fixture in internal_FIXTURES:
        destination = output_directory / fixture.filename
        subprocess.run(
            internal_qpdf_command(qpdf, source, destination, fixture),
            check=True,
        )
        subprocess.run(
            [qpdf, f"--password={fixture.user_password}", "--check", str(destination)],
            check=True,
            capture_output=True,
        )
        record = asdict(fixture)
        record["qpdf_options"] = list(fixture.qpdf_options)
        record["command"] = internal_display_command(fixture)
        record["sha256"] = internal_sha256(destination)
        records.append(record)

    manifest = {
        "generator": {
            "name": "qpdf",
            "version": qpdf_version,
            "website": "https://qpdf.sourceforge.io/",
        },
        "source": {
            "filename": source.name,
            "sha256": internal_sha256(source),
        },
        "expected": {
            "text": internal_EXPECTED_TEXT,
            "info_title": internal_EXPECTED_TITLE,
            "xmp_marker": internal_XMP_MARKER,
        },
        "fixtures": records,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def internal_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qpdf",
        default=shutil.which("qpdf"),
        help="qpdf executable (default: discovered on PATH)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=internal_DEFAULT_OUTPUT,
        help="fixture output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = internal_parse_args()
    if not args.qpdf:
        raise SystemExit("qpdf was not found; install it or pass --qpdf")
    internal_generate(args.qpdf, args.output.resolve())


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from core_pdf import PdfDocument


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="core-pdf",
        description="High-Performance PDF Engine",
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "--mode",
        choices=("text", "markdown"),
        default="text",
        help="Plain output format",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def extract_output(document: PdfDocument, output_format: str) -> str:
    if output_format == "markdown":
        return document.to_markdown()
    return document.extract_text()


def print_plain_output(output: str) -> None:
    sys.stdout.write(output)
    if output and not output.endswith("\n"):
        sys.stdout.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        with PdfDocument(args.pdf) as document:
            print_plain_output(extract_output(document, args.mode))
        return 0
    except Exception as exc:
        print(f"core-pdf: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

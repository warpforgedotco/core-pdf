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
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Path to PDF file(s) or directory containing PDF files",
    )
    parser.add_argument(
        "--mode",
        choices=("markdown",),
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively scan directories for PDF files",
    )
    parser.add_argument(
        "-p",
        "--print",
        dest="print_content",
        action="store_true",
        help="Print extracted content to stdout",
    )
    parser.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="Emit output Markdown files to disk",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save output files (implies --write)",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def resolve_pdf_paths(paths: Sequence[Path], *, recursive: bool = False) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            pattern = "**/*.pdf" if recursive else "*.pdf"
            found = sorted(p for p in path.glob(pattern) if p.is_file())
            resolved.extend(found)
        elif path.is_file():
            resolved.append(path)
        else:
            raise FileNotFoundError(f"Path does not exist or is invalid: {path}")
    return resolved


def process_pdf(
    path: Path,
    output_format: str,
    *,
    print_content: bool = False,
    write_files: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    with PdfDocument(path) as document:
        if not print_content and not write_files and output_dir is None:
            # Parse document without emitting MD
            document.extract()
            return None

        if output_format != "markdown":
            raise ValueError(f"unsupported output format: {output_format}")
        content = document.extract().to_markdown()

    if print_content:
        sys.stdout.write(content)
        if content and not content.endswith("\n"):
            sys.stdout.write("\n")
        return None

    ext = ".md" if output_format == "markdown" else ".txt"
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{path.stem}{ext}"
    else:
        target = path.with_suffix(ext)

    target.write_text(content, encoding="utf-8")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        pdf_paths = resolve_pdf_paths(args.paths, recursive=args.recursive)
        if not pdf_paths:
            print("core-pdf: no PDF files found", file=sys.stderr)
            return 1

        total = len(pdf_paths)
        succeeded = 0
        failed = 0

        for path in pdf_paths:
            try:
                written = process_pdf(
                    path,
                    args.mode,
                    print_content=args.print_content,
                    write_files=args.write,
                    output_dir=args.output_dir,
                )
                if written is not None:
                    print(f"Emitted {path} -> {written}")
                elif not args.print_content:
                    print(f"Parsed {path}")
                succeeded += 1
            except Exception as exc:
                print(f"core-pdf [{path}]: {exc}", file=sys.stderr)
                failed += 1

        success_rate = (succeeded / total * 100) if total > 0 else 0.0
        print(
            f"Processed {total} PDF(s): {succeeded} succeeded, "
            f"{failed} failed ({success_rate:.1f}% success rate)"
        )

        return 1 if failed > 0 else 0
    except Exception as exc:
        print(f"core-pdf: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

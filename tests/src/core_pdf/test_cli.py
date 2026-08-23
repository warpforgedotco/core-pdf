# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from pathlib import Path

import pytest

from core_pdf.cli import build_parser, main, resolve_pdf_paths


def test_cli_parser_defaults(tmp_path: Path) -> None:
    pdf_file = tmp_path / "sample.pdf"
    parser = build_parser()
    args = parser.parse_args([str(pdf_file)])
    assert args.paths == [pdf_file]
    assert args.mode == "markdown"
    assert not args.recursive
    assert not args.print_content
    assert not args.write
    assert args.output_dir is None


def test_cli_parser_flags(tmp_path: Path) -> None:
    pdf_file = tmp_path / "sample.pdf"
    parser = build_parser()
    args = parser.parse_args(["-p", "-w", str(pdf_file)])
    assert args.print_content
    assert args.write


def test_resolve_pdf_paths_directory(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    pdf1 = tmp_path / "a.pdf"
    pdf2 = sub / "b.pdf"
    non_pdf = tmp_path / "c.txt"
    pdf1.write_bytes(b"%PDF-1.4...")
    pdf2.write_bytes(b"%PDF-1.4...")
    non_pdf.write_text("hello")

    assert resolve_pdf_paths([pdf1]) == [pdf1]

    shallow = resolve_pdf_paths([tmp_path], recursive=False)
    assert shallow == [pdf1]

    recursive = resolve_pdf_paths([tmp_path], recursive=True)
    assert sorted(recursive) == [pdf1, pdf2]


def test_cli_main_nonexistent_path(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["nonexistent.pdf"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "core-pdf:" in captured.err


def test_cli_main_summary_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from core_pdf.impl.engine.writing import serialize_pdf_file
    from core_pdf.impl.objects import PdfStream
    from core_pdf.impl.primitives import PdfName, PdfReference

    objects = {
        1: {PdfName.of("Type"): PdfName.of("Catalog"), PdfName.of("Pages"): PdfReference(2)},
        2: {
            PdfName.of("Type"): PdfName.of("Pages"),
            PdfName.of("Kids"): [PdfReference(3)],
            PdfName.of("Count"): 1,
        },
        3: {
            PdfName.of("Type"): PdfName.of("Page"),
            PdfName.of("Parent"): PdfReference(2),
            PdfName.of("MediaBox"): [0, 0, 10, 10],
            PdfName.of("Contents"): PdfReference(4),
        },
        4: PdfStream({}, b"q\nQ\n"),
    }
    pdf_bytes = serialize_pdf_file(objects, trailer={PdfName.of("Root"): PdfReference(1)})
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(pdf_bytes)

    exit_code = main([str(pdf_file)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Processed 1 PDF(s): 1 succeeded, 0 failed (100.0% success rate)" in captured.out

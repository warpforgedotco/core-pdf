from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core_pdf import cli as native_cli
from core_pdf_ocr import PdfDocument
from core_pdf_ocr import cli as ocr_cli
from tests.helpers.pdf_bytes import text_pages_pdf


def test_ocr_cli_selects_companion_without_changing_native_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "document.pdf"
    source.write_bytes(text_pages_pdf(["Native document text"]))
    calls: list[PdfDocument] = []

    def recognized(document: PdfDocument) -> SimpleNamespace:
        calls.append(document)
        return SimpleNamespace(to_markdown=lambda: "Recovered companion text")

    monkeypatch.setattr(PdfDocument, "extract", recognized)
    assert ocr_cli.main([str(source), "--print"]) == 0
    assert "Recovered companion text" in capsys.readouterr().out
    assert len(calls) == 1
    assert calls[0].closed

    assert native_cli.main([str(source), "--print"]) == 0
    native_output = capsys.readouterr().out
    assert "Native document text" in native_output
    assert "Recovered companion text" not in native_output
    assert len(calls) == 1


def test_ocr_cli_writes_markdown_from_recursive_selection(tmp_path: Path) -> None:
    input_dir = tmp_path / "input" / "nested"
    input_dir.mkdir(parents=True)
    source = input_dir / "document.pdf"
    source.write_bytes(text_pages_pdf(["Native document text"]))
    output_dir = tmp_path / "output"

    assert ocr_cli.main([str(input_dir.parent), "--recursive", "-o", str(output_dir)]) == 0
    assert "Native document text" in (output_dir / "document.md").read_text()


def test_ocr_cli_reports_its_program_name_for_missing_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ocr_cli.main([str(tmp_path / "missing.pdf")]) == 1
    assert capsys.readouterr().err.startswith("core-pdf-ocr:")


def test_ocr_module_entrypoint_extracts_with_installed_package(tmp_path: Path) -> None:
    source = tmp_path / "document.pdf"
    source.write_bytes(text_pages_pdf(["Native document text"]))

    completed = subprocess.run(
        [sys.executable, "-m", "core_pdf_ocr", str(source), "--print"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Native document text" in completed.stdout

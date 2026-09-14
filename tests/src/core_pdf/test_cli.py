from pathlib import Path

import pytest

from core_pdf.cli import run


@pytest.mark.parametrize("arguments", [[], ["--print"], ["--write"], ["--output-dir", "output"]])
def test_cli_output_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    text_pdf_bytes: bytes,
    arguments: list[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "example.pdf"
    source.write_bytes(text_pdf_bytes)
    assert run([str(source), *arguments]) == 0
    output = capsys.readouterr()
    assert not output.err
    target = tmp_path / ("output/example.md" if "--output-dir" in arguments else "example.md")
    if "--print" in arguments:
        assert "Hello maintenance" in output.out
        assert not target.exists()
    elif arguments:
        assert "Hello maintenance" in target.read_text()
    else:
        assert "Parsed" in output.out
        assert not target.exists()


def test_cli_print_precedes_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    text_pdf_bytes: bytes,
) -> None:
    source = tmp_path / "example.pdf"
    source.write_bytes(text_pdf_bytes)
    assert run([str(source), "--print", "--write"]) == 0
    assert "Hello maintenance" in capsys.readouterr().out
    assert not source.with_suffix(".md").exists()


def test_cli_discovery_and_failure_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    text_pdf_bytes: bytes,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "good.pdf").write_bytes(text_pdf_bytes)
    assert run([str(tmp_path)]) == 1
    assert "no PDF files found" in capsys.readouterr().err
    assert run([str(tmp_path), "--recursive"]) == 0
    assert "1 succeeded" in capsys.readouterr().out
    (nested / "bad.pdf").write_bytes(b"not a PDF")
    assert run([str(tmp_path), "--recursive"]) == 1
    assert "1 failed" in capsys.readouterr().out
    assert run([str(tmp_path / "missing.pdf")]) == 1
    assert "does not exist" in capsys.readouterr().err

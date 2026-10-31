# Ported from pdfplumber's repair.py, Copyright (c) 2015, Jeremy Singer-Vine, under the
# MIT License.
from __future__ import annotations

import pathlib
import shutil
import subprocess
from io import BufferedReader, BytesIO
from typing import Literal

RepairSetting = Literal["default", "prepress", "printer", "ebook", "screen"]
RepairInput = str | pathlib.Path | BufferedReader | BytesIO


def repair_stream(
    source: RepairInput,
    password: str | None = None,
    gs_path: str | pathlib.Path | None = None,
    setting: RepairSetting = "default",
) -> BytesIO:
    executable = (
        gs_path or shutil.which("gs") or shutil.which("gswin32c") or shutil.which("gswin64c")
    )
    if executable is None:
        raise RuntimeError(
            "Cannot find Ghostscript, which is required for repairs.\n"
            "Visit https://www.ghostscript.com/ for installation instructions."
        )
    command = [
        str(executable),
        "-sstdout=%stderr",
        "-o",
        "-",
        "-sDEVICE=pdfwrite",
        f"-dPDFSETTINGS=/{setting}",
    ]
    if password:
        command.append(f"-sPDFPassword={password}")
    is_path = isinstance(source, (str, pathlib.Path))
    command.append(str(pathlib.Path(source).absolute()) if is_path else "-")  # type: ignore[arg-type]
    process = subprocess.run(  # noqa: S603
        command,
        input=None if is_path else source.read(),  # type: ignore[union-attr]
        capture_output=True,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.decode("utf-8"))
    return BytesIO(process.stdout)


def repair(
    source: RepairInput,
    outfile: str | pathlib.Path | None = None,
    password: str | None = None,
    gs_path: str | pathlib.Path | None = None,
    setting: RepairSetting = "default",
) -> BytesIO | None:
    repaired = repair_stream(source, password, gs_path=gs_path, setting=setting)
    if outfile:
        pathlib.Path(outfile).write_bytes(repaired.read())
        return None
    return repaired


__all__ = ("RepairSetting", "repair", "repair_stream")

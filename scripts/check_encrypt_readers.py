"""Check that other readers open what the pypdf facade's PdfWriter.encrypt writes.

uv run --all-packages --with "pypdf[full]==7.0.0" python scripts/check_encrypt_readers.py

Each sample file is cloned into the facade's PdfWriter, encrypted with each algorithm pypdf
offers, and written. Every output must be encrypted and must open with the user password in
the real pypdf, which also has to extract the same text the facade does, and in each of
Poppler (pdftotext), Ghostscript and qpdf found on the PATH. At least one of those three
must be present. Poppler and Ghostscript must print words of the text the real pypdf reads
and no error, as Ghostscript exits 0 even when a password does not work; qpdf's --check
exit status reflects the password. A wrong password must be refused by every reader.
"""

from __future__ import annotations

import collections
import io
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pypdf as real  # ty: ignore[unresolved-import]

from core_pdf_compat import pypdf as facade

FILES = (
    "tests/upstream/pypdf/resources/crazyones.pdf",
    "tests/upstream/pypdf/resources/AutoCad_Simple.pdf",
    "tests/upstream/pypdf/resources/attachment.pdf",
)
ALGORITHMS = ("RC4-40", "RC4-128", "AES-128", "AES-256-R5", "AES-256")
# Ghostscript refuses AES-256-R5 with a short password, on real pypdf's own files too, so
# the passwords are long enough for every reader.
USER = "userpass"
OWNER = "ownerpass"


def poppler(path: Path) -> list[str]:
    return ["pdftotext", "-upw", USER, str(path), "-"]


def ghostscript(path: Path) -> list[str]:
    return [
        "gs",
        "-q",
        "-dNOPAUSE",
        "-dBATCH",
        "-sDEVICE=txtwrite",
        "-sOutputFile=-",
        f"-sPDFPassword={USER}",
        str(path),
    ]


def qpdf(path: Path) -> list[str]:
    return ["qpdf", f"--password={USER}", "--check", str(path)]


# name -> (executable, command, whether it prints the page text)
READERS: dict[str, tuple[str, Callable[[Path], list[str]], bool]] = {
    "Poppler": ("pdftotext", poppler, True),
    "Ghostscript": ("gs", ghostscript, True),
    "qpdf": ("qpdf", qpdf, False),
}


def opened(command: list[str], words: list[str], prints_text: bool) -> str | None:
    """None if the reader opened the file; otherwise why not."""
    result = subprocess.run(command, capture_output=True, timeout=120, check=False)
    output = (result.stdout + result.stderr).decode(errors="replace")
    if result.returncode != 0:
        return f"exit {result.returncode}: {output.strip()[:200]}"
    if "Error" in output:
        return output.strip()[:200]
    if prints_text:
        flat = " ".join(output.split())
        missing = [word for word in words if word not in flat]
        if missing:
            return f"text lacks {missing[:3]}"
    return None


def sample_words(pages: list[str]) -> list[str]:
    """A few plain words from the first page with text, to look for in a reader's output."""
    text = next((page for page in pages if page.strip()), "")
    return [word for word in text.split() if word.isalpha() and len(word) > 2][:3]


def encrypted(source: str, algorithm: str) -> bytes:
    writer = facade.PdfWriter(clone_from=source)
    writer.encrypt(user_password=USER, owner_password=OWNER, algorithm=algorithm)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def page_text(library: Any, data: bytes, password: str) -> list[str]:
    reader = library.PdfReader(io.BytesIO(data))
    if not reader.is_encrypted:
        raise AssertionError("not encrypted")
    if not reader.decrypt(password):
        raise AssertionError("password refused")
    return [page.extract_text() for page in reader.pages]


def main() -> int:
    present = {
        name: (build, prints_text)
        for name, (tool, build, prints_text) in READERS.items()
        if shutil.which(tool)
    }
    if not present:
        print("no other reader found: install Poppler, Ghostscript or qpdf")
        return 1
    counts: collections.Counter[str] = collections.Counter()
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as directory:
        for source in FILES:
            for algorithm in ALGORITHMS:
                label = f"{Path(source).name} {algorithm}"
                data = encrypted(source, algorithm)
                path = Path(directory) / "out.pdf"
                path.write_bytes(data)
                expected: list[str] = []
                try:
                    expected = page_text(facade, data, USER)
                    if page_text(real, data, USER) != expected:
                        problems.append(f"{label}: real pypdf reads other text")
                    else:
                        counts["real pypdf"] += 1
                except Exception as error:  # noqa: BLE001
                    problems.append(f"{label}: real pypdf: {type(error).__name__}: {error}")
                try:
                    page_text(real, data, "wrong password")
                    problems.append(f"{label}: real pypdf accepted a wrong password")
                except AssertionError:
                    counts["real pypdf, wrong password refused"] += 1
                words = sample_words(expected)
                for name, (build, prints_text) in present.items():
                    reason = opened(build(path), words, prints_text)
                    if reason is None:
                        counts[name] += 1
                    else:
                        problems.append(f"{label}: {name}: {reason}")
                    wrong = [part.replace(USER, "wrong-password") for part in build(path)]
                    if opened(wrong, words, prints_text) is None:
                        problems.append(f"{label}: {name} opened it with a wrong password")
                    else:
                        counts[f"{name}, wrong password refused"] += 1
    total = len(FILES) * len(ALGORITHMS)
    print(f"{total} encrypted outputs")
    for name, count in sorted(counts.items()):
        print(f"  {count:5d}  {name}")
    for name in READERS.keys() - present.keys():
        print(f"  {name}: not installed, not checked")
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

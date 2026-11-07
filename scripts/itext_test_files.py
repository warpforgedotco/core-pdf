"""iText's test PDFs for ISO 32003 (AES-GCM, revision 7) and ISO 32004 (PDF MAC).

No file under tests/upstream uses either, and no library the facades follow writes them. iText
Core 9 does, and its test resources include files from a third-party tool and tampered copies.
tests/compat/itext_test_files.tsv lists each file used: its name, sha256, and path in the
itext-java repository at RELEASE. fetched() downloads any the cache lacks and checks each hash,
so a changed or missing file fails the check instead of shrinking it.
"""

import hashlib
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LISTING = ROOT / "tests" / "compat" / "itext_test_files.tsv"
CACHE = ROOT / ".upstream-results" / "itext-test-files"
RELEASE = "9.8.0"
BASE = f"https://raw.githubusercontent.com/itext/itext-java/{RELEASE}/"
PAUSES = (5, 15, 30)


def listed() -> dict[str, tuple[str, str]]:
    """Each file's name, mapped to its sha256 and its path in the itext-java repository."""
    rows = (line.split("\t") for line in LISTING.read_text().splitlines() if line)
    return {name: (digest, path) for name, digest, path in rows}


def download(url: str) -> bytes:
    for pause in PAUSES:
        request = urllib.request.Request(url, headers={"User-Agent": "core-pdf-tests"})  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                return response.read()
        except Exception as error:  # noqa: BLE001
            print(f"download of {url} failed: {error!r}; retrying in {pause} s", flush=True)
            time.sleep(pause)
    raise SystemExit(f"could not download {url}")


def fetched() -> dict[str, Path]:
    """Each listed file's name, mapped to its path in the cache, downloading any it lacks."""
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (digest, path) in listed().items():
        target = CACHE / name
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            data = download(BASE + path)
            if hashlib.sha256(data).hexdigest() != digest:
                raise SystemExit(f"{name}: sha256 differs from {LISTING.name}")
            target.write_bytes(data)
        paths[name] = target
    return paths

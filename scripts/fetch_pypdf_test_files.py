"""Download the files pypdf's tests fetch from the web, into tests/upstream/pypdf/tests/pdf_cache.

    uv run --group test python scripts/fetch_pypdf_test_files.py
    uv run --group test python scripts/fetch_pypdf_test_files.py --record [--write]

tests/compat/pypdf_test_files.tsv lists every such file, by name and URL. The script fetches
each one the cache lacks, pausing between attempts as the servers rate-limit, and exits 1
naming any it still cannot get, so a gap fails CI instead of shrinking what the checks read.

--record runs pypdf's whole suite on the real pypdf, out-of-scope tests included, and records
every file the tests ask for. It fails if that differs from the list; --write rewrites the
list instead. Run it with --write after moving the pypdf checkout to a new release.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_upstream_suite import ROOT, SUITES, UPSTREAM, environment_command  # noqa: E402

CACHE = UPSTREAM / "pypdf" / "tests" / "pdf_cache"
LISTING = ROOT / "tests" / "compat" / "pypdf_test_files.tsv"
PAUSES = (5, 15, 30, 60)


def listed() -> dict[str, str]:
    """Each file pypdf's tests download: its name in the cache, and its URL."""
    split = (line.partition("\t") for line in LISTING.read_text().splitlines() if line)
    return {name: url for name, _, url in split}


def missing() -> list[str]:
    return [name for name in listed() if not (CACHE / name).exists()]


def fetch(name: str, url: str) -> bool:
    path = CACHE / name
    for pause in PAUSES:
        request = urllib.request.Request(url, headers={"User-Agent": "core-pdf-tests"})  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                data = response.read()
        except Exception as error:  # noqa: BLE001
            print(f"download of {url} failed: {error!r}; retrying in {pause} s", flush=True)
            time.sleep(pause)
            continue
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(data)
        partial.replace(path)
        return True
    return False


def fetch_missing() -> int:
    """Download every listed file the cache lacks; 1 if any could not be fetched."""
    CACHE.mkdir(parents=True, exist_ok=True)
    urls = listed()
    wanted = missing()
    failed = [name for name in wanted if not fetch(name, urls[name])]
    print(f"pypdf test files: {len(urls)} listed, {len(wanted) - len(failed)} downloaded now")
    if failed:
        print("could not download: " + ", ".join(failed))
        return 1
    return 0


def record(*, write: bool) -> int:
    """Run pypdf's whole suite on the real pypdf and compare the files it asks for."""
    suite = SUITES["pypdf"]
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "files.json"
        environment = {
            **os.environ,
            "PYTHONPATH": str(ROOT / "tests" / "compat"),
            "PYPDF_FILE_RECORD": str(out),
        }
        command = [
            *environment_command(suite, reference=True),
            *("python", "-m", "pytest", *suite.tests, "-o", "addopts=", "-o", "required_plugins="),
            *("-p", "no:cacheprovider", "-p", "pypdf_file_record", "--timeout=120", "-q"),
            *("--continue-on-collection-errors", "--ignore=tests/scripts/test_make_release.py"),
        ]
        subprocess.run(
            command,
            cwd=UPSTREAM / suite.checkout,
            env=environment,
            stdout=subprocess.DEVNULL,
            check=False,
        )
        recorded: dict[str, str | None] = json.loads(out.read_text())
    unknown = sorted(name for name, url in recorded.items() if url is None)
    if unknown:
        print("no URL for: " + ", ".join(unknown))
        return 1
    found = {name: url for name, url in recorded.items() if url is not None}
    if write:
        LISTING.write_text("".join(f"{name}\t{url}\n" for name, url in sorted(found.items())))
        print(f"wrote {len(found)} files to {LISTING.relative_to(ROOT)}")
        return 0
    old = listed()
    added = sorted(found.keys() - old.keys())
    dropped = sorted(old.keys() - found.keys())
    moved = sorted(name for name in found.keys() & old.keys() if found[name] != old[name])
    for label, names in (("not listed", added), ("no longer used", dropped), ("URL moved", moved)):
        if names:
            print(f"{label}: " + ", ".join(names))
    if added or dropped or moved:
        print(f"rerun with --write to update {LISTING.relative_to(ROOT)}")
        return 1
    print(f"the {len(found)} files pypdf's tests download are all listed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="check the list against a run")
    parser.add_argument("--write", action="store_true", help="with --record: rewrite the list")
    arguments = parser.parse_args()
    if arguments.record and (code := record(write=arguments.write)):
        return code
    return fetch_missing()


if __name__ == "__main__":
    sys.exit(main())

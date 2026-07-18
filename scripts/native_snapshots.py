#!/usr/bin/env python3
"""Check or update native extraction Markdown snapshots."""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path
from typing import Any, cast

from core_pdf.impl.engine.extraction.document import PdfDocument
from core_pdf.impl.engine.extraction.page_text.engine import build_page_extraction_result
from core_pdf.impl.engine.extraction.page_text.snapshots import native_snapshot

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "SCORE-Bench" / "src"
SNAPSHOT_ROOT = ROOT / "tests" / "snapshots" / "native"
FIXTURE_PATTERN = re.compile(r"^fixture: (?P<name>[^\n]+)$", re.MULTILINE)


def snapshot_for_fixture(fixture_name: str) -> str:
    """Extract the first page of a fixture and render its snapshot."""
    fixture = FIXTURE_ROOT / fixture_name
    if not fixture.is_file():
        raise FileNotFoundError(f"native extraction fixture does not exist: {fixture}")
    with PdfDocument.open(fixture) as document:
        page = cast(Any, document.pages[0])
        result = build_page_extraction_result(page)
        return native_snapshot(fixture_name, page, result)


def snapshot_cases() -> tuple[tuple[str, Path], ...]:
    """Return checked-in snapshot cases in stable order."""
    cases: list[tuple[str, Path]] = []
    for snapshot in sorted(SNAPSHOT_ROOT.glob("*.md")):
        match = FIXTURE_PATTERN.search(snapshot.read_text())
        if match is None:
            raise ValueError(f"snapshot has no fixture front matter: {snapshot}")
        cases.append((match.group("name"), snapshot))
    if not cases:
        raise ValueError(f"no native snapshots found in {SNAPSHOT_ROOT}")
    return tuple(cases)


def check_snapshots(*, update: bool = False) -> int:
    """Check snapshots, or rewrite them when ``update`` is true."""
    failures = 0
    for fixture_name, snapshot in snapshot_cases():
        actual = snapshot_for_fixture(fixture_name)
        if update:
            snapshot.write_text(actual)
            continue
        expected = snapshot.read_text()
        if actual == expected:
            continue
        failures += 1
        diff = difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=str(snapshot),
            tofile=f"{snapshot} (actual)",
        )
        print("".join(diff), end="")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite checked-in snapshots from current native extraction output",
    )
    args = parser.parse_args()
    failures = check_snapshots(update=args.update)
    if failures:
        print(f"{failures} native snapshot(s) differ")
        return 1
    print("native extraction snapshots are up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

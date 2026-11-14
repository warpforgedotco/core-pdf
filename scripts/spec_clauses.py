# SPDX-License-Identifier: AGPL-3.0-only

"""Extract a specification's clause numbers and headings from its PDF's outline.

scripts/spec_index.py checks every cited clause against these lists, and shows each clause's
heading in docs/spec-index.md. The lists live here rather than in the package, so no wheel
carries the headings. Rerun this when a new edition or corrigendum is cited:

uv run --all-packages python scripts/spec_clauses.py ISO_32000-2_sponsored_EC3.pdf "ISO 32000-2"
uv run --all-packages python scripts/spec_clauses.py PDF32000_2008.pdf "ISO 32000-1"

Each outline entry that starts with a clause number ("7.9.2.2 Text string type"), an annex
clause ("D.2 Latin character set and encodings") or an annex ("Annex D (normative) ...") is
kept, the first time it appears; errata entries and front matter are not.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from core_pdf_compat import pypdf

CLAUSE = re.compile(
    r"^(?P<clause>[0-9]+(?:\.[0-9]+)*|[A-Z](?:\.[0-9]+)+|Annex [A-Z])\s+(?P<heading>.+)$"
)
DIRECTORY = Path(__file__).resolve().parent / "spec_clauses"


def outline_titles(items: list[object]) -> list[str]:
    titles: list[str] = []
    for item in items:
        if isinstance(item, list):
            titles.extend(outline_titles(item))
        else:
            titles.append(str(getattr(item, "title", "")))
    return titles


def clauses(path: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for title in outline_titles(pypdf.PdfReader(path).outline):
        match = CLAUSE.match(" ".join(title.split()))
        if match and match["clause"] not in found:
            found[match["clause"]] = match["heading"]
    return found


def file_for(specification: str) -> Path:
    return DIRECTORY / (specification.replace(" ", "_") + ".tsv")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    path, specification = argv
    found = clauses(path)
    DIRECTORY.mkdir(exist_ok=True)
    target = file_for(specification)
    target.write_text("".join(f"{clause}\t{heading}\n" for clause, heading in found.items()))
    print(f"{len(found)} clauses of {specification} written to {target.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

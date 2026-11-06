"""Report reference libraries whose pinned version is behind their latest release.

    uv run python scripts/check_upstream_versions.py [--markdown]

The pins come from the reference packages of the suites in scripts/run_upstream_suite.py,
which the upstream suites and the facade comparison scripts test against, and each is looked
up on PyPI. Their own dependencies, such as x-ray's PyMuPDF, follow whatever that release
requires, so they are not checked here. The
script prints one row per package and exits 1 if any pin is behind, so a weekly workflow can
open an issue. --markdown prints the table as Markdown, for that issue.
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_upstream_suite import SUITES  # noqa: E402

PIN = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==([^\s;]+)$")
PYPI = "https://pypi.org/pypi/{}/json"


def pins() -> dict[str, tuple[str, set[str]]]:
    """Each pinned package: its version and the suites that pin it."""
    found: dict[str, tuple[str, set[str]]] = {}
    for name, suite in SUITES.items():
        # A suite's first reference package is the library its facade follows; the others,
        # such as llama_index's pypdf, are whatever that library's latest release accepts.
        for requirement in suite.reference[:1]:
            match = PIN.match(requirement)
            if match is None:
                continue
            package, version = match[1].lower(), match[2]
            previous = found.get(package)
            suites = {name} if previous is None else previous[1] | {name}
            found[package] = (version, suites)
    return found


def latest(package: str) -> str:
    with urllib.request.urlopen(PYPI.format(package), timeout=30) as response:  # noqa: S310
        return json.load(response)["info"]["version"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true")
    arguments = parser.parse_args()
    rows = []
    for package, (pinned, suites) in sorted(pins().items()):
        newest = latest(package)
        rows.append((package, pinned, newest, ", ".join(sorted(suites)), pinned != newest))
    behind = [row for row in rows if row[4]]
    if arguments.markdown:
        print("| Package | Pinned | Latest | Suites |")
        print("| ------- | ------ | ------ | ------ |")
        for package, pinned, newest, suites, is_behind in rows:
            mark = " (behind)" if is_behind else ""
            print(f"| {package} | {pinned} | {newest}{mark} | {suites} |")
    else:
        for package, pinned, newest, suites, is_behind in rows:
            state = "BEHIND" if is_behind else "current"
            print(f"{state:8} {package:30} pinned {pinned:12} latest {newest:12} ({suites})")
    return 1 if behind else 0


if __name__ == "__main__":
    sys.exit(main())

"""Report reference libraries whose pinned version is behind their latest release.

    uv run python scripts/check_upstream_versions.py [--markdown]

The pins come from the reference packages of the suites in scripts/run_upstream_suite.py,
which the upstream suites and the facade comparison scripts test against, and each is looked
up on PyPI. Their own dependencies, such as x-ray's PyMuPDF, follow whatever that release
requires, so they are not checked here. The
script prints one row per package and exits 1 if any pin is behind, so a weekly workflow can
open an issue. --markdown prints the table as Markdown, for that issue.

    uv run --no-project python scripts/check_upstream_versions.py --checkouts

--checkouts makes no network call: it compares the commit HEAD records for each checkout
under tests/upstream with the release commit RELEASES pins next to the suites, and exits 1
if a checkout is elsewhere or has no entry, since the suites run the checkout's tests.
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_upstream_suite import RELEASES, ROOT, SUITES  # noqa: E402

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


def recorded_checkouts() -> dict[str, str]:
    """The commit HEAD records for each submodule under tests/upstream, by checkout name."""
    listing = subprocess.run(
        ["git", "ls-tree", "HEAD", "tests/upstream/"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = {}
    for line in listing.splitlines():
        info, path = line.split("\t", 1)
        _mode, kind, commit = info.split()
        if kind == "commit":
            found[Path(path).name] = commit
    return found


def check_checkouts() -> int:
    """Print each checkout's release and return 1 if any is not at its release commit."""
    recorded = recorded_checkouts()
    names = recorded.keys() | RELEASES.keys() | {suite.checkout for suite in SUITES.values()}
    wrong = []
    for checkout in sorted(names):
        commit = recorded.get(checkout)
        release = RELEASES.get(checkout)
        if commit is None:
            wrong.append(f"tests/upstream/{checkout}: no submodule recorded at HEAD")
        elif release is None:
            wrong.append(f"tests/upstream/{checkout}: at {commit}, which RELEASES does not list")
        elif commit != release[1]:
            wrong.append(f"tests/upstream/{checkout}: at {commit}, not {release[0]} ({release[1]})")
        else:
            print(f"release  tests/upstream/{checkout:14} {release[0]:10} {commit}")
    for problem in wrong:
        print(f"WRONG    {problem}")
    return 1 if wrong else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--checkouts", action="store_true")
    arguments = parser.parse_args()
    if arguments.checkouts:
        return check_checkouts()
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

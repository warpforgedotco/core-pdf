"""Check that every skip in a facade's upstream run has a reason.

    uv run --group test python scripts/run_upstream_suite.py pypdf
    uv run --group test python scripts/run_upstream_suite.py pypdf --reference
    python3 scripts/audit_upstream_skips.py pypdf

A skip is accounted for when the real library skips the same test, or when it reached a
stand-in for a private module, which out_of_scope.toml rules out. Anything else is printed,
and the exit status is 1.
"""

import argparse
import collections
import sys
import tomllib
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / ".upstream-results"
OUT_OF_SCOPE = ROOT / "tests" / "compat" / "out_of_scope.toml"
STAND_IN_PREFIX = "out of scope: "


def skipped_tests(junit: Path) -> dict[str, str]:
    """Each skipped test's id and skip message."""
    tests: dict[str, str] = {}
    for case in ElementTree.parse(junit).iter("testcase"):  # noqa: S314
        for outcome in case:
            if outcome.tag == "skipped":
                tests[f"{case.get('classname')}::{case.get('name')}"] = (
                    outcome.get("message") or ""
                ).splitlines()[0]
    return tests


def is_private(message: str, upstream: str) -> bool:
    """Whether the message names something below a private module of ``upstream``."""
    if not message.startswith(STAND_IN_PREFIX):
        return False
    dotted = message.removeprefix(STAND_IN_PREFIX).split(".")
    if dotted[0] != upstream:
        return False
    # The last part is the name reached, and a method may follow its class.
    return any(part.startswith("_") for part in dotted[1:-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite")
    arguments = parser.parse_args()
    scope = tomllib.loads(OUT_OF_SCOPE.read_text()).get(arguments.suite, {})
    facade = skipped_tests(RESULTS / f"{arguments.suite}-facade.xml")
    reference = skipped_tests(RESULTS / f"{arguments.suite}-reference.xml")
    upstream = next(iter(arguments.suite.split(".")))
    sorted_out: collections.Counter[str] = collections.Counter()
    unexplained: list[tuple[str, str]] = []
    for test, message in sorted(facade.items()):
        if test in reference:
            sorted_out["also skipped on the real library"] += 1
        elif is_private(message, upstream):
            sorted_out["reaches a private module"] += 1
        else:
            unexplained.append((test, message))
    print(f"{arguments.suite}: {len(facade)} skipped. Listed reason: {scope.get('reason')}")
    for reason, count in sorted_out.most_common():
        print(f"{count:5} {reason}")
    print(f"{len(unexplained):5} without a reason")
    for test, message in unexplained:
        print(f"  {test}: {message}")
    return 1 if unexplained else 0


if __name__ == "__main__":
    sys.exit(main())

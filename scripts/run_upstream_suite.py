"""Run a reference library's own test suite against its core_pdf_compat facade.

    uv run --group test python scripts/run_upstream_suite.py pdfplumber
    uv run --group test python scripts/run_upstream_suite.py pdfplumber --reference

--reference runs the same tests on the real library, to tell failures caused by the
environment from gaps in the facade. Extra arguments go to pytest.
"""

import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "tests" / "upstream"
RESULTS = ROOT / ".upstream-results"


class Suite(NamedTuple):
    checkout: str
    tests: tuple[str, ...]
    aliases: str
    reference: tuple[str, ...]
    requires: tuple[str, ...] = ()


LLAMA_READERS = "llama-index-integrations/readers/llama-index-readers-file"
SUITES = {
    "pdfminer": Suite("pdfminer.six", ("tests",), "pdfminer=pdfminer", ("pdfminer.six==20260107",)),
    "pdfplumber": Suite(
        "pdfplumber", ("tests",), "pdfplumber=pdfplumber", ("pdfplumber==0.11.10",), ("pandas",)
    ),
    "pypdf": Suite(
        "pypdf",
        ("tests",),
        "pypdf=pypdf",
        ("pypdf[full]==6.15.0",),
        ("pyyaml", "pillow", "cryptography", "fpdf2", "defusedxml", "pytest-socket"),
    ),
    "pikepdf": Suite(
        "pikepdf",
        ("tests",),
        "pikepdf=pikepdf",
        ("pikepdf==10.11.0",),
        ("hypothesis", "pillow", "psutil", "numpy", "python-dateutil"),
    ),
    "xray": Suite("x-ray", ("tests",), "xray=xray", ("x-ray",), ("pymupdf", "requests")),
    "unstructured": Suite(
        "unstructured",
        ("test_unstructured/partition/pdf_image",),
        "unstructured.partition.pdf=unstructured",
        ("unstructured[pdf]==0.25.2",),
        ("requests", "pandas", "numpy", "pillow", "unstructured-inference"),
    ),
    "llamaindex": Suite(
        "llama_index",
        (f"{LLAMA_READERS}/tests",),
        "llama_index.readers.file=llamaindex",
        ("llama-index-readers-file", "pypdf"),
    ),
}


def pytest_command(suite: Suite, *, reference: bool, junit: Path, extra: list[str]) -> list[str]:
    base = ["uv", "run", "--python", "3.14"]
    if reference:
        base += ["--no-project", "--isolated"]
        packages = suite.reference
    else:
        base += ["--project", str(ROOT), "--all-packages", "--group", "test"]
        packages = ()
    for package in (*packages, *suite.requires, "pytest", "pytest-timeout", "pytest-xdist"):
        base += ["--with", package]
    return [
        *base,
        "python",
        "-m",
        "pytest",
        *suite.tests,
        "-o",
        "addopts=",
        "-o",
        "required_plugins=",
        "-p",
        "no:cacheprovider",
        "--timeout=60",
        "--tb=short",
        "-q",
        "--continue-on-collection-errors",
        f"--junitxml={junit}",
        *(("-p", "compat_alias") if not reference else ()),
        *extra,
    ]


def summarize(junit: Path) -> dict[str, object]:
    counts: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    failing: list[str] = []
    for case in ElementTree.parse(junit).iter("testcase"):  # noqa: S314
        outcome = next((child for child in case if child.tag != "system-out"), None)
        kind = "passed" if outcome is None else outcome.tag
        if kind in {"system-err", "properties"}:
            kind = "passed"
        counts[kind] += 1
        if kind in {"failure", "error"} and outcome is not None:
            failing.append(f"{case.get('classname')}::{case.get('name')}")
            reasons[(outcome.get("message") or "")[:100]] += 1
    return {"counts": dict(counts), "reasons": reasons.most_common(25), "failing": failing}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=SUITES)
    parser.add_argument("--reference", action="store_true")
    arguments, extra = parser.parse_known_args()
    suite = SUITES[arguments.suite]
    mode = "reference" if arguments.reference else "facade"
    RESULTS.mkdir(exist_ok=True)
    junit = RESULTS / f"{arguments.suite}-{mode}.xml"
    junit.unlink(missing_ok=True)
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "tests" / "compat")}
    if arguments.reference:
        environment.pop("PYTHONPATH")
    else:
        environment["CORE_PDF_COMPAT_ALIASES"] = suite.aliases
    command = pytest_command(suite, reference=arguments.reference, junit=junit, extra=extra)
    log = RESULTS / f"{arguments.suite}-{mode}.log"
    with log.open("w") as handle:
        code = subprocess.run(
            command,
            cwd=UPSTREAM / suite.checkout,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if not junit.exists():
        print(f"{arguments.suite} {mode}: no report, see {log}")
        return code or 1
    summary = summarize(junit)
    (RESULTS / f"{arguments.suite}-{mode}.json").write_text(json.dumps(summary, indent=2))
    print(f"{arguments.suite} {mode}: {summary['counts']}")
    return code


if __name__ == "__main__":
    sys.exit(main())

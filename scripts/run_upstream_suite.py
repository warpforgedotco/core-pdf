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
import tomllib
from pathlib import Path
from typing import NamedTuple, cast
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "tests" / "upstream"
RESULTS = ROOT / ".upstream-results"
OUT_OF_SCOPE = ROOT / "tests" / "compat" / "out_of_scope.toml"


class Suite(NamedTuple):
    checkout: str
    tests: tuple[str, ...]
    aliases: str
    reference: tuple[str, ...]
    requires: tuple[str, ...] = ()
    # The --reference run's Python, when the real library cannot run on the workspace's.
    reference_python: str | None = None
    # The facade run's packages in place of reference, when they differ.
    facade_requires: tuple[str, ...] = ()


LLAMA_READERS = "llama-index-integrations/readers/llama-index-readers-file"
SUITES = {
    "pdfminer": Suite("pdfminer.six", ("tests",), "pdfminer=pdfminer", ("pdfminer.six==20260107",)),
    "pdfplumber": Suite(
        "pdfplumber",
        ("tests",),
        "pdfplumber=pdfplumber,pdfminer=pdfminer",
        ("pdfplumber==0.11.10",),
        ("pandas",),
    ),
    "pypdf": Suite(
        "pypdf",
        ("tests",),
        "pypdf=pypdf",
        ("pypdf[full]==7.0.0",),
        ("pyyaml", "pillow", "cryptography", "fpdf2", "defusedxml", "pytest-socket", "brotli"),
    ),
    # x-ray 0.4.0 pins PyMuPDF 1.24.14; on PyMuPDF 2.1 its own tests fail (see check_xray_output).
    "xray": Suite("x-ray", ("tests",), "xray=xray", ("x-ray==0.4.0",), ("pymupdf==1.24.14",)),
    "unstructured": Suite(
        "unstructured",
        ("test_unstructured/partition/pdf_image",),
        "unstructured.partition.pdf=unstructured,unstructured.errors=unstructured.errors",
        ("unstructured[pdf]==0.27.37",),
        (
            "pytest-mock",
            "spacy>=3.8.15,<3.9.0",
            "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
        ),
        # unstructured[pdf] needs unstructured-inference, and that onnxruntime, which has no
        # CPython 3.15 wheels yet: the real library runs on 3.14, and the facade on the
        # workspace's 3.15 with the pdf extra's other packages, both of those stubbed
        # (out_of_scope.toml).
        reference_python="3.14",
        facade_requires=(
            "unstructured==0.27.37",
            "pdf2image>=1.17.0,<2",
            "pdfminer.six>=20251230,<20270000",
            "pi-heif>=1.2.0,<2",
            "pikepdf>=10.3.0,<11",
            "pypdf>=6.9.1,<7",
            "unstructured-pytesseract>=0.3.15,<1",
            # unstructured imports these, which unstructured-inference brings otherwise.
            "opencv-python>=4.13.0.90",
            "pandas>=1.5.0",
            "scipy>=1.17.0",
        ),
    ),
    "llamaindex": Suite(
        "llama_index",
        (f"{LLAMA_READERS}/tests",),
        "llama_index.readers.file=llamaindex",
        # pypdf pinned, so its text changes don't show up here. llama-index-readers-file 0.7.0
        # requires pypdf<7, so this is 6.21.4, not the pypdf suite's 7.0.0.
        ("llama-index-readers-file==0.7.0", "pypdf[full]==6.21.4"),
        ("llama-index-core==0.14.25", "pytest-mock"),
    ),
}

# The release tag each checkout under tests/upstream sits at, and that tag's commit. Moving a
# pin above means moving the checkout and this entry with it; check_upstream_versions.py
# --checkouts fails when a recorded checkout is not at its commit.
RELEASES = {
    "pdfminer.six": ("20260107", "9e1243c4ad000bf9bbe60e81fc8dde2fccc0ed3b"),
    "pdfplumber": ("v0.11.10", "7d4f2f582f2d99f9e60ba522fdf7afd2f6d54c62"),
    "pypdf": ("7.0.0", "8eeb0509d712e324e68e2bd38aff69e3b672e0cd"),
    "x-ray": ("v0.4.0", "4dd5b0bcc5cdd774b9e82fc47b1d7208e92e4779"),
    "unstructured": ("0.27.37", "293d235cfcd5f5e89c3cc5ff85562005e182ae33"),
    # llama_index tags only its core releases. v0.14.25 is llama-index-core 0.14.25, and its
    # llama-index-readers-file directory is the 0.7.0 release's (98c52d8, the version bump).
    "llama_index": ("v0.14.25", "f12d46acab73f5b2243ef49c2f00101617b38ce4"),
}


def pytest_command(
    suite: Suite,
    *,
    reference: bool,
    junit: Path,
    extra: list[str],
    excluded_files: tuple[str, ...],
) -> list[str]:
    return [
        *environment_command(suite, reference=reference),
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
        "-p",
        "compat_scope",
        *(f"--ignore={path}" for path in excluded_files),
        *extra,
    ]


def environment_command(suite: Suite, *, reference: bool) -> list[str]:
    """The `uv run` prefix that gives a suite's tests their environment."""
    base = ["uv", "run"]
    if reference:
        if suite.reference_python is not None:
            base += ["--python", suite.reference_python]
        base += ["--isolated", "--no-project"]
        packages = suite.reference
    else:
        base += ["--project", str(ROOT), "--all-packages", "--group", "test"]
        packages = suite.facade_requires
    for package in (*packages, *suite.requires, "pytest", "pytest-timeout", "pytest-xdist"):
        base += ["--with", package]
    return base


def download_network_files() -> None:
    """Fetch every file pypdf's tests download, into the checkout's tests/pdf_cache.

    The tests read some of them by name only, expecting an earlier test to have fetched them,
    which parallel runs do not guarantee. pypdf's own helper retries without pausing and drops
    the error, so a file a rate-limited server refuses would go missing silently; this fails
    naming any file it cannot get.
    """
    script = Path(__file__).with_name("fetch_pypdf_test_files.py")
    subprocess.run([sys.executable, str(script)], check=True)


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
            message = (outcome.get("message") or "").split("\n")[0][:200]
            failing.append(f"{case.get('classname')}::{case.get('name')}: {message}")
            reasons[(outcome.get("message") or "")[:100]] += 1
    return {"counts": dict(counts), "reasons": reasons.most_common(25), "failing": failing}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=SUITES)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument(
        "--network",
        action="store_true",
        help="pypdf only: download the test files and run the tests that need the network too",
    )
    arguments, extra = parser.parse_known_args()
    suite = SUITES[arguments.suite]
    mode = "reference" if arguments.reference else "facade"
    RESULTS.mkdir(exist_ok=True)
    junit = RESULTS / f"{arguments.suite}-{mode}.xml"
    junit.unlink(missing_ok=True)
    scope = tomllib.loads(OUT_OF_SCOPE.read_text()).get(arguments.suite, {})
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "tests" / "compat"),
        "CORE_PDF_COMPAT_OUT_OF_SCOPE": ",".join(scope.get("names", ())),
        "CORE_PDF_COMPAT_KEEP": ",".join(scope.get("keep", ())),
        "CORE_PDF_COMPAT_STRATEGIES": ",".join(scope.get("strategies", ())),
        "CORE_PDF_COMPAT_STRATEGY_AGNOSTIC": ",".join(scope.get("strategy_agnostic", ())),
    }
    if not arguments.reference:
        environment["CORE_PDF_COMPAT_ALIASES"] = suite.aliases
        environment["CORE_PDF_COMPAT_STUBS"] = ",".join(scope.get("stubs", ()))
        environment["CORE_PDF_COMPAT_PARTIAL"] = ",".join(scope.get("partial", ()))
    command = pytest_command(
        suite,
        reference=arguments.reference,
        junit=junit,
        extra=[
            *extra,
            *(("-m", scope["markers"]) if "markers" in scope and not arguments.network else ()),
            *(f"--deselect={node}" for node in scope.get("deselect", ())),
        ],
        excluded_files=tuple(scope.get("files", ())),
    )
    # pypdf's tests that fetch files are in scope unless a marker deselects them.
    network = arguments.network or (arguments.suite == "pypdf" and "markers" not in scope)
    if network:
        download_network_files()
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
    for name in cast("list[str]", summary["failing"]):
        print(f"  failed: {name}")
    return code


if __name__ == "__main__":
    sys.exit(main())

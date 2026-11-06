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
        ("pypdf[full]==6.21.4",),
        ("pyyaml", "pillow", "cryptography", "fpdf2", "defusedxml", "pytest-socket", "brotli"),
    ),
    # x-ray 0.4.0 pins PyMuPDF 1.24.14; on PyMuPDF 2.1 its own tests fail (see check_xray_output).
    "xray": Suite("x-ray", ("tests",), "xray=xray", ("x-ray==0.4.0",), ("pymupdf==1.24.14",)),
    "unstructured": Suite(
        "unstructured",
        ("test_unstructured/partition/pdf_image",),
        "unstructured.partition.pdf=unstructured",
        ("unstructured[pdf]==0.27.36",),
        (
            "unstructured[pdf]==0.27.36",
            "pytest-mock",
            "spacy>=3.8.15,<3.9.0",
            "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
        ),
    ),
    "llamaindex": Suite(
        "llama_index",
        (f"{LLAMA_READERS}/tests",),
        "llama_index.readers.file=llamaindex",
        # pypdf as the pypdf suite pins it, so its text changes don't show up here.
        ("llama-index-readers-file==0.7.0", "pypdf[full]==6.21.4"),
        ("llama-index-core==0.14.25", "pytest-mock"),
    ),
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
    base = ["uv", "run", "--python", "3.14"]
    if reference:
        base += ["--no-project", "--isolated"]
        packages = suite.reference
    else:
        base += ["--project", str(ROOT), "--all-packages", "--group", "test"]
        packages = ()
    for package in (*packages, *suite.requires, "pytest", "pytest-timeout", "pytest-xdist"):
        base += ["--with", package]
    return base


def download_network_files(suite: Suite, *, reference: bool, environment: dict[str, str]) -> None:
    """Fetch the files pypdf's tests download, with pypdf's own helper, as its CI does first.

    The tests read some of them by name only, expecting an earlier test to have fetched them,
    which parallel runs do not guarantee. They land in the checkout's tests/pdf_cache.
    """
    command = [
        *environment_command(suite, reference=reference),
        "python",
        "-c",
        "from tests import download_test_pdfs; download_test_pdfs()\n" + FETCH_MISSING,
    ]
    subprocess.run(command, cwd=UPSTREAM / suite.checkout, env=environment, check=True)


# pypdf's helper retries without pausing and its thread pool drops the error, so a file that
# a rate-limited server refuses goes missing silently. This fetches what is missing, with
# pauses, and fails naming any file it still cannot get.
FETCH_MISSING = """
import sys, time, urllib.request
from pathlib import Path
import yaml
cache = Path("tests/pdf_cache")
missing = []
for entry in yaml.safe_load(Path("tests/example_files.yaml").read_text()):
    path = cache / entry["local_filename"]
    for pause in (5, 15, 30, 60):
        if path.exists():
            break
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "core-pdf-tests"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())
        except Exception as error:
            print(f"download of {entry['url']} failed: {error!r}; retrying in {pause} s")
            time.sleep(pause)
    if not path.exists():
        missing.append(entry["local_filename"])
if missing:
    sys.exit("could not download: " + ", ".join(missing))
"""


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
        download_network_files(suite, reference=arguments.reference, environment=environment)
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

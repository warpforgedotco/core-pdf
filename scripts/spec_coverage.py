# SPDX-License-Identifier: AGPL-3.0-only

"""Write docs/spec-coverage.md: what each PDF version defines that core never reads.

The Arlington PDF Model (PDF Association, Apache-2.0) lists every object PDF defines, each
key with the version that introduced it, and the names a name-valued key can take. This
compares it with the names core's source mentions -- every string and bytes literal under
src/core_pdf, a leading "/" dropped -- and lists, by the version or extension that added
them, the keys core never names and, for keys it does name, the values it never names.

A key core never names is certainly not read. One it names may still be read only in some
of the objects that have it, so this finds gaps, not proof of support.

uv run --all-packages python scripts/spec_coverage.py [path to an Arlington checkout]

Without a path it downloads the pinned commit below into .upstream-results/.
"""

from __future__ import annotations

import ast
import csv
import io
import re
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src" / "core_pdf"
REPORT = ROOT / "docs" / "spec-coverage.md"
ARLINGTON_COMMIT = "603c174b024d1d2bb15a33686af83800d5c41269"
ARLINGTON_URL = (
    "https://codeload.github.com/pdf-association/arlington-pdf-model/tar.gz/" + ARLINGTON_COMMIT
)
VERSIONS = ("2.0", "1.7", "1.6", "1.5", "1.4", "1.3", "1.2", "1.1", "1.0")
VERSION = re.compile(r"^[12]\.[0-9]$")
# fn:Extension(NAME,1.7), alone or in fn:Eval(fn:Extension(NAME,1.7) || 2.0).
EXTENSION = re.compile(r"fn:Extension\(([A-Za-z0-9_]+)")
LATER_VERSION = re.compile(r"\|\|\s*([12]\.[0-9])\)")
SINCE_VALUE = re.compile(r"fn:SinceVersion\(([12]\.[0-9]),\s*([^,()\]]+)\)")
SHOWN_OBJECTS = 6


def arlington_tables(argv: list[str]) -> Path:
    if argv:
        return Path(argv[0]) / "tsv" / "latest"
    target = ROOT / ".upstream-results" / f"arlington-{ARLINGTON_COMMIT}"
    tables = target / "tsv" / "latest"
    if not tables.is_dir():
        with urllib.request.urlopen(ARLINGTON_URL) as response:  # noqa: S310
            data = response.read()
        prefix = f"arlington-pdf-model-{ARLINGTON_COMMIT}/tsv/latest/"
        tables.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                if not (member.isfile() and member.name.startswith(prefix)):
                    continue
                if member.name.endswith(".tsv"):
                    extracted = archive.extractfile(member)
                    assert extracted is not None
                    (tables / Path(member.name).name).write_bytes(extracted.read())
    return tables


def named_by_core() -> set[str]:
    """Every string and bytes literal in core's source, a leading "/" dropped."""
    names: set[str] = set()
    for path in SOURCE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
                value = node.value if isinstance(node.value, str) else node.value.decode("latin-1")
                names.add(value.removeprefix("/"))
    return names


def folded(name: str) -> str:
    """A name as core may spell a value it normalizes: lower case, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def added_by(since: str) -> str:
    """The version, or extension, that added a key or value."""
    since = since.strip()
    if VERSION.match(since):
        return since
    later = LATER_VERSION.search(since)
    if later:
        return later.group(1)
    extension = EXTENSION.search(since)
    return f"extension {extension.group(1)}" if extension else "other"


def name_values(possible: str) -> list[tuple[str, str]]:
    """The names a key can take, each with the version that added it ("" for the key's)."""
    values: list[tuple[str, str]] = []
    for version, name in SINCE_VALUE.findall(possible):
        values.append((name.strip(), version))
    plain = SINCE_VALUE.sub("", possible)
    for group in re.findall(r"\[([^\]]*)\]", plain):
        for token in group.split(","):
            token = token.strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.#-]*", token) and not token.startswith("fn"):
                values.append((token, ""))
    return values


def objects_list(objects: set[str]) -> str:
    shown = sorted(objects)[:SHOWN_OBJECTS]
    rest = len(objects) - len(shown)
    return ", ".join(shown) + (f" and {rest} more" if rest else "")


def main(argv: list[str]) -> int:
    tables = arlington_tables(argv)
    core = named_by_core()
    # Keys are read by their exact names; values are often normalized ("Multiply" as
    # "multiply"), so they are compared folded.
    core_folded = {folded(name) for name in core}
    # added by -> key -> objects; and added by -> (key, value) -> objects.
    keys: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    values: dict[str, dict[tuple[str, str], set[str]]] = defaultdict(lambda: defaultdict(set))
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for table in sorted(tables.glob("*.tsv")):
        obj = table.stem
        for row in csv.DictReader(table.open(newline=""), delimiter="\t"):
            key = row["Key"]
            if key.isdigit() or key == "*" or row.get("DeprecatedIn"):
                continue
            version = added_by(row["SinceVersion"])
            totals[version][0] += 1
            if key in core:
                totals[version][1] += 1
                if "name" in row["Type"].split(";"):
                    for value, value_version in name_values(row["PossibleValues"]):
                        if folded(value) not in core_folded:
                            values[value_version or version][(key, value)].add(obj)
            else:
                keys[version][key].add(obj)
    order = [*VERSIONS, *sorted(version for version in totals if version not in VERSIONS)]
    lines = [
        "# Specification coverage",
        "",
        "Generated by `scripts/spec_coverage.py` from the [Arlington PDF Model]"
        f"(https://github.com/pdf-association/arlington-pdf-model/tree/{ARLINGTON_COMMIT})",
        f"(commit `{ARLINGTON_COMMIT[:8]}`), against the names core's source mentions. For each",
        "version or extension: the keys PDF defines that core never names, and, for keys it",
        "does name, the values it never names (compared ignoring case and punctuation, as",
        "core often normalizes values). A key core never names is certainly not read;",
        "one it names may be read only in some of the objects that have it. Deprecated keys",
        "are left out. See also the [specification index](spec-index.md).",
        "",
        "| Added by | Keys, by object | Named by core | Never named |",
        "| --- | --- | --- | --- |",
    ]
    for version in order:
        total, named = totals[version]
        label = f"PDF {version}" if VERSION.match(version) else version
        lines.append(f"| {label} | {total} | {named} | {total - named} |")
    for version in order:
        label = (
            f"PDF {version}"
            if VERSION.match(version)
            else version.replace("extension", "Extension")
        )
        missing_keys = keys.get(version, {})
        missing_values = values.get(version, {})
        if not missing_keys and not missing_values:
            continue
        lines += ["", f"## {label}"]
        if missing_keys:
            lines += ["", "Keys core never names:", ""]
            lines += [
                f"- `{key}` in {objects_list(objects)}"
                for key, objects in sorted(missing_keys.items())
            ]
        if missing_values:
            lines += ["", "Values core never names, of keys it reads:", ""]
            lines += [
                f"- `{key}` `{value}` in {objects_list(objects)}"
                for (key, value), objects in sorted(missing_values.items())
            ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

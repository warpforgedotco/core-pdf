# SPDX-License-Identifier: AGPL-3.0-only

"""Write or check docs/spec-index.md: core's code by specification and by PDF version.

Reads, from the source of core_pdf.impl without importing it, every `cites` marker
(core_pdf.impl.spec.citations) and every module docstring that starts with a citation such as
"ISO 32000-2 7.8: ...", and lists them by PDF version (the rules that depend on it), by
extension, by profile and by specification. The specifications, extensions and profiles
themselves come from core_pdf.impl.spec.citations and core_pdf.impl.spec.standards.

uv run --all-packages python scripts/spec_index.py           # rewrite the index
uv run --all-packages python scripts/spec_index.py --check   # fail if it would change

--check also fails on a function that compares a PDF version (against PdfVersion(...) or a
PDF_x_y constant) without a cites marker giving since or until, and on a marker whose
arguments are not literals. Every module of spec and of the floor packages below it
(CITED_PACKAGES, other than __init__.py) must start its docstring with a module citation, and
a docstring anywhere whose first line looks like a citation must name a known specification.
"""

from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

from core_pdf.impl.spec.citations import SPECIFICATIONS, implemented_extension_ids
from core_pdf.impl.spec.standards import (
    IMPLEMENTED_EXTENSIONS,
    STANDARD_PROFILES,
    get_standard_profile,
)
from core_records import GeneratedRecord

ROOT = Path(__file__).resolve().parent.parent
IMPL = ROOT / "src" / "core_pdf" / "impl"
INDEX = ROOT / "docs" / "spec-index.md"
VERSION_CONSTANT = re.compile(r"^PDF_\d_\d$")
# The clause numbers and headings of each specification scripts/spec_clauses.py extracted.
CLAUSES: dict[str, dict[str, str]] = {
    path.stem.replace("_", " "): dict(
        line.split("\t", 1) for line in path.read_text().splitlines() if "\t" in line
    )
    for path in sorted((ROOT / "scripts" / "spec_clauses").glob("*.tsv"))
}
SPEC_IDS = sorted(SPECIFICATIONS, key=len, reverse=True)
MODULE_CITATION = re.compile(
    r"^(?P<spec>"
    + "|".join(re.escape(spec) for spec in SPEC_IDS)
    + r")(?: (?P<clause>Annex [A-Z]|[A-Z]?[0-9][0-9.]*|[A-Z](?:\.[0-9]+)*))?: (?P<title>.+)$"
)
# A first line that starts like a citation, to catch a misspelt or unregistered specification.
CITATION_LIKE = re.compile(r"^(?:ISO|ITU|RFC|Adobe|FIPS|ICC|PDF|PostScript|TIFF)\b[^:]*: ")
# The packages whose every module, other than __init__.py, carries a module citation.
CITED_PACKAGES = ("spec", "adobe_fonts", "jbig2", "crypto", "postscript", "data")
KINDS = (
    ("pdf", "The PDF standard"),
    ("extension", "ISO extensions"),
    ("embedded", "Embedded standards"),
)


class Citation(GeneratedRecord):
    specification: str
    clause: str = ""
    since: str | None = None
    until: str | None = None
    introduced: str | None = None
    extension: str | None = None
    profile: str | None = None
    rule: str = ""


class Marked:
    """A function's cites markers."""

    def __init__(self, module: str, name: str) -> None:
        self.module = module
        self.name = name
        self.citations: list[Citation] = []


class Survey:
    """The cited modules and functions found, and what is wrong with the markers."""

    def __init__(self) -> None:
        self.modules: list[tuple[str, str, str, str]] = []
        self.functions: list[Marked] = []
        self.problems: list[str] = []


def compares_version(function: ast.AST) -> bool:
    def is_version(node: ast.AST) -> bool:
        if isinstance(node, ast.Call):
            callee = node.func
            return (isinstance(callee, ast.Name) and callee.id == "PdfVersion") or (
                isinstance(callee, ast.Attribute) and callee.attr == "PdfVersion"
            )
        if isinstance(node, ast.Name):
            return VERSION_CONSTANT.match(node.id) is not None
        if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
            return any(is_version(element) for element in node.elts)
        return False

    return any(
        isinstance(node, ast.Compare) and any(map(is_version, (node.left, *node.comparators)))
        for node in ast.walk(function)
    )


def citation_of(decorator: ast.expr) -> Citation | None | str:
    """The citation a decorator makes, None if it is not cites, or an error."""
    if not isinstance(decorator, ast.Call):
        return None
    callee = decorator.func
    name = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
    if name != "cites":
        return None
    keywords: dict[str, object] = {}
    try:
        positional = [ast.literal_eval(argument) for argument in decorator.args]
        for keyword in decorator.keywords:
            if keyword.arg is None:
                return "cites arguments must be literals"
            keywords[keyword.arg] = ast.literal_eval(keyword.value)
    except ValueError:
        return "cites arguments must be literals"
    return Citation(*positional, **keywords)


def clause_problem(specification: str, clause: str) -> str | None:
    """Why a cited clause is wrong: one the specification's outline does not have."""
    known = CLAUSES.get(specification)
    if known is None or not clause or clause in known:
        return None
    return f"{specification} has no clause {clause!r} (see scripts/spec_clauses)"


def clause_title(specification: str, clause: str) -> str:
    title = CLAUSES.get(specification, {}).get(clause)
    return f" ({title})" if title else ""


def citation_problems(citation: Citation) -> list[str]:
    problems = []
    if citation.specification not in SPECIFICATIONS:
        problems.append(f"unknown specification {citation.specification!r}")
    wrong_clause = clause_problem(citation.specification, citation.clause)
    if wrong_clause:
        problems.append(wrong_clause)
    problems.extend(
        f"malformed PDF version {version!r}"
        for version in (citation.since, citation.until, citation.introduced)
        if version is not None and re.fullmatch(r"[12]\.[0-9]", version) is None
    )
    if citation.extension is not None and citation.extension not in implemented_extension_ids():
        problems.append(f"no implemented extension {citation.extension!r}")
    if citation.profile is not None and get_standard_profile(citation.profile) is None:
        problems.append(f"unknown profile {citation.profile!r}")
    if not citation.rule:
        problems.append("a cites marker needs a rule")
    return problems


def survey() -> Survey:
    found = Survey()
    for path in sorted(IMPL.rglob("*.py")):
        module = path.relative_to(IMPL.parent.parent).with_suffix("").as_posix().replace("/", ".")
        tree = ast.parse(path.read_text(), str(path))
        docstring = ast.get_docstring(tree)
        first_line = docstring.splitlines()[0] if docstring else ""
        match = MODULE_CITATION.match(first_line)
        if match:
            found.modules.append(
                (match["spec"], match["clause"] or "", module, match["title"].rstrip("."))
            )
            wrong_clause = clause_problem(match["spec"], match["clause"] or "")
            if wrong_clause:
                found.problems.append(f"{path.relative_to(ROOT)}: {wrong_clause}")
        elif CITATION_LIKE.match(first_line):
            found.problems.append(
                f"{path.relative_to(ROOT)}: the docstring starts like a citation but names no "
                f"known specification and clause: {first_line!r}"
            )
        elif path.name != "__init__.py" and path.relative_to(IMPL).parts[0] in CITED_PACKAGES:
            found.problems.append(
                f"{path.relative_to(ROOT)}: a spec or floor module needs a docstring starting "
                'with a citation, such as "ISO 32000-2 7.8: ..."'
            )
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            marked = Marked(module, node.name)
            for decorator in node.decorator_list:
                citation = citation_of(decorator)
                if isinstance(citation, str):
                    found.problems.append(f"{path}:{node.lineno} {node.name}: {citation}")
                elif citation is not None:
                    marked.citations.append(citation)
                    found.problems.extend(
                        f"{path.relative_to(ROOT)}:{node.lineno} {node.name}: {problem}"
                        for problem in citation_problems(citation)
                    )
            if marked.citations:
                found.functions.append(marked)
            if compares_version(node) and not any(
                citation.since or citation.until for citation in marked.citations
            ):
                found.problems.append(
                    f"{path.relative_to(ROOT)}:{node.lineno} {node.name} compares a PDF version "
                    "without a cites marker giving since or until"
                )
    return found


def version_key(version: str) -> tuple[int, int]:
    major, minor = version.split(".")
    return (int(major), int(minor))


def where(marked: Marked) -> str:
    return f"`{marked.module}.{marked.name}`"


def clause_order(entry: tuple[str, str, str]) -> tuple[object, ...]:
    """Order entries by clause as the specification does: 7.2, 7.10, 11, then the annexes."""
    clause, code, text = entry
    parts = [part for part in re.split(r"[ .]", clause) if part]
    return (tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts), code, text)


def labelled(specification: str, clause: str) -> str:
    """A clause and its heading, ahead of what cites it."""
    return f"{clause}{clause_title(specification, clause)} " if clause else ""


def cited(citation: Citation) -> str:
    return f"{citation.specification} {citation.clause}".strip() + clause_title(
        citation.specification, citation.clause
    )


def render(found: Survey) -> str:
    lines = [
        "# Specification index",
        "",
        "Generated by `scripts/spec_index.py` from the `cites` markers and module citations in",
        "`core_pdf.impl`; run it after citing new code (`--check` runs in CI). Core shares one",
        "implementation of each chapter across PDF versions; this lists where it depends on the",
        "version, an extension or a profile, and which specification each piece implements.",
        "",
        "## By PDF version",
        "",
        "Rules that depend on the document's effective PDF version (`SemanticContext`).",
        "",
        "| Applies | Rule | Clause | Code |",
        "| --- | --- | --- | --- |",
    ]
    rows: list[tuple[tuple[int, int], int, str]] = []
    for marked in found.functions:
        for citation in marked.citations:
            if citation.since:
                key, when = version_key(citation.since), f"from PDF {citation.since}"
            elif citation.until:
                key, when = version_key(citation.until), f"before PDF {citation.until}"
            else:
                continue
            row = f"| {when} | {citation.rule} | {cited(citation)} | {where(marked)} |"
            rows.append((key, 0 if citation.until else 1, row))
    lines.extend(row for _key, _order, row in sorted(rows))
    introduced = sorted(
        (version_key(citation.introduced), f"| PDF {citation.introduced} | {citation.rule} | "
         f"{cited(citation)} | {where(marked)} |")
        for marked in found.functions
        for citation in marked.citations
        if citation.introduced
    )  # fmt: skip
    if introduced:
        lines += [
            "",
            "Features a version introduced, which core reads whatever version a document declares:",
            "",
            "| Introduced | Feature | Clause | Code |",
            "| --- | --- | --- | --- |",
            *(row for _key, row in introduced),
        ]
    lines += ["", "## Extensions", "", "| Extension | Implemented | Code |", "| --- | --- | --- |"]
    by_extension: dict[str, list[str]] = defaultdict(list)
    for marked in found.functions:
        for citation in marked.citations:
            if citation.extension:
                by_extension[citation.extension].append(where(marked))
    for coverage in IMPLEMENTED_EXTENSIONS:
        key = f"{coverage.prefix}/{coverage.base_version}/{coverage.extension_level}"
        revision = coverage.extension_revision or ""
        code = ", ".join(sorted(by_extension.get(key, []))) or "-"
        lines.append(f"| {key}{revision} | {'; '.join(coverage.features)} | {code} |")
    lines += [
        "",
        "## Profiles",
        "",
        "Core recognizes these conformance claims (`PdfDocument.standards`) but does not change",
        "its behaviour for them; `core-pdf-validate` checks conformance with veraPDF.",
        "",
        "| Profile | Family | Edition | Base version |",
        "| --- | --- | --- | --- |",
        *(
            f"| {profile.identifier} | {profile.family} | {profile.edition} | "
            f"{profile.base_version} |"
            for profile in STANDARD_PROFILES
        ),
        "",
        "## By specification",
    ]
    modules: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for spec, clause, module, title in found.modules:
        modules[spec].append((clause, module, title))
    functions: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for marked in found.functions:
        for citation in marked.citations:
            functions[citation.specification].append(
                (citation.clause, where(marked), citation.rule)
            )
    for kind, heading in KINDS:
        lines += ["", f"### {heading}"]
        for spec, specification in SPECIFICATIONS.items():
            if specification.kind != kind:
                continue
            entries = [
                f"- {labelled(spec, clause)}`{module}`: {title}"
                for clause, module, title in sorted(modules.get(spec, []), key=clause_order)
            ] + [
                f"- {labelled(spec, clause)}{code}: {rule}"
                for clause, code, rule in sorted(functions.get(spec, []), key=clause_order)
            ]
            lines += ["", f"**{spec}**: {specification.title} ({specification.edition})", ""]
            lines += entries or ["- not cited in code yet"]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    found = survey()
    text = render(found)
    if "--check" in argv:
        problems = list(found.problems)
        if not INDEX.exists() or INDEX.read_text() != text:
            problems.append(f"{INDEX.relative_to(ROOT)} is stale: run scripts/spec_index.py")
        for problem in problems:
            print(problem)
        if problems:
            return 1
        print(
            f"{len(found.functions)} cited functions and {len(found.modules)} cited modules;"
            " the index is current"
        )
        return 0
    INDEX.write_text(text)
    for problem in found.problems:
        print(problem)
    print(f"wrote {INDEX.relative_to(ROOT)}")
    return 1 if found.problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

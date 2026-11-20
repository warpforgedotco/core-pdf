# SPDX-License-Identifier: AGPL-3.0-only

"""Check that Markdown and HTML keep markup open across spans that share it.

A line keeps a span per text run, and runs split at every font or glyph-run change, so
neighbouring spans often share their markup. render_styled_line keeps a layer open across
the spans that share it: two bold runs `U` and `N` are `**UN**`, never `**U****N**`, and a
bold highlighted run then a bold one is `**<mark>a</mark>b**`. This checks it on lines
built here for every layer the serializer writes (bold, italic, both, strikeout,
underline, highlight, superscript, subscript and an abbreviation's expansion), for spaces
between and around spans, and for neighbours that differ, then on the upstream test PDFs:
no line's Markdown joins two emphasis markers (`****`, `~~~~`), no HTML
closes a tag only to reopen it, and no Markdown emphasis marker sits on whitespace (the
Markdown checks skip a line whose text has a literal * or ~, which Markdown output does
not escape).

uv run --all-packages python scripts/check_styled_spans.py [--all]

Without --all, the corpus scan reads the upstream PDFs that showed split markup; with it,
every PDF in the checkouts CI's core shard has.
"""

from __future__ import annotations

import glob
import logging
import re
import sys
import warnings
from pathlib import Path

from core_pdf import PdfDocument
from core_pdf.impl.output.model import Block, TextLine, TextSpan
from core_pdf.impl.output.serialize import html_line, markdown_line

S = TextSpan
# Upstream PDFs whose Markdown wrapped neighbouring spans that share markup one by one.
CORPUS = (
    "tests/upstream/llama_index/docs/examples/query_engine/pdf_tables/billionaires_page.pdf",
    "tests/upstream/llama_index/llama-index-integrations/retrievers/"
    "llama-index-retrievers-vectorize/tests/research.pdf",
    "tests/upstream/pdfminer.six/samples/contrib/issue-1008-inline-ascii85.pdf",
    "tests/upstream/pdfminer.six/samples/contrib/pr-00530-ml-lines.pdf",
    "tests/upstream/pdfminer.six/samples/nonfree/dmca.pdf",
    "tests/upstream/pdfminer.six/samples/nonfree/f1040nr.pdf",
    "tests/upstream/pdfminer.six/samples/nonfree/i1040nr.pdf",
    "tests/upstream/pdfminer.six/samples/nonfree/naacl06-shinyama.pdf",
    "tests/upstream/pdfplumber/tests/pdfs/chelsea_pdta.pdf",
    "tests/upstream/pdfplumber/tests/pdfs/issue-192-example.pdf",
    "tests/upstream/pdfplumber/tests/pdfs/issue-987-test.pdf",
    "tests/upstream/pdfplumber/tests/pdfs/word365_structure.pdf",
    "tests/upstream/pypdf/resources/GeoBase_NHNC1_Data_Model_UML_EN.pdf",
    "tests/upstream/pypdf/resources/issue-604.pdf",
    "tests/upstream/pypdf/sample-files/016-libre-office-link/libre-office-link.pdf",
    "tests/upstream/pypdf/sample-files/022-pdfkit/pdfkit.pdf",
)
# The checkouts CI's core shard has.
FULL_CORPUS = (
    "tests/upstream/llama_index/**/*.pdf",
    "tests/upstream/pdfminer.six/**/*.pdf",
    "tests/upstream/pdfplumber/**/*.pdf",
    "tests/upstream/pypdf/**/*.pdf",
)
# (name, spans, Markdown, HTML)
CASES: tuple[tuple[str, tuple[TextSpan, ...], str, str], ...] = (
    (
        "bold runs",
        (S("U", bold=True), S("N", bold=True), S("I", bold=True)),
        "**UNI**",
        "<strong>UNI</strong>",
    ),
    ("italic runs", (S("ab", italic=True), S("c", italic=True)), "*abc*", "<em>abc</em>"),
    (
        "bold italic runs",
        (S("T", bold=True, italic=True), S("he", bold=True, italic=True)),
        "***The***",
        "<em><strong>The</strong></em>",
    ),
    ("struck runs", (S("x", strikeout=True), S("y", strikeout=True)), "~~xy~~", "<del>xy</del>"),
    ("underlined runs", (S("a", underline=True), S("b", underline=True)), "<u>ab</u>", "<u>ab</u>"),
    (
        "highlighted runs",
        (S("2004", mark=True), S("-", mark=True), S("08", mark=True)),
        "<mark>2004-08</mark>",
        "<mark>2004-08</mark>",
    ),
    (
        "superscript runs",
        (S("x"), S("2", superscript=True), S("3", superscript=True)),
        "x<sup>23</sup>",
        "x<sup>23</sup>",
    ),
    (
        "subscript runs",
        (S("H"), S("2", subscript=True), S("i", subscript=True), S("O")),
        "H<sub>2i</sub>O",
        "H<sub>2i</sub>O",
    ),
    (
        "superscript then subscript",
        (S("x"), S("2", superscript=True), S("i", subscript=True)),
        "x<sup>2</sup><sub>i</sub>",
        "x<sup>2</sup><sub>i</sub>",
    ),
    (
        "one abbreviation in two runs",
        (S("CI", expansion="Central Agency"), S("A", expansion="Central Agency")),
        '<abbr title="Central Agency">CIA</abbr>',
        '<abbr title="Central Agency">CIA</abbr>',
    ),
    (
        "two abbreviations",
        (S("a", expansion="x"), S("b", expansion="y")),
        '<abbr title="x">a</abbr><abbr title="y">b</abbr>',
        '<abbr title="x">a</abbr><abbr title="y">b</abbr>',
    ),
    (
        "bold and underlined runs",
        (S("En", bold=True, underline=True), S("-", bold=True, underline=True)),
        "**<u>En-</u>**",
        "<strong><u>En-</u></strong>",
    ),
    # A layer shared by neighbours that differ otherwise stays open across them.
    (
        "bold highlight then bold",
        (S("PAP-11", bold=True, mark=True), S(",", bold=True)),
        "**<mark>PAP-11</mark>,**",
        "<strong><mark>PAP-11</mark>,</strong>",
    ),
    (
        "bold then bold italic",
        (S('"', bold=True), S("stroke", bold=True, italic=True), S('"', bold=True)),
        '**"*stroke*"**',
        "<strong>&quot;<em>stroke</em>&quot;</strong>",
    ),
    (
        "bold italic then bold",
        (S("A", bold=True, italic=True), S("B", bold=True)),
        "***A*B**",
        "<strong><em>A</em>B</strong>",
    ),
    (
        "italic then bold italic",
        (S("checks.", italic=True), S("Safe.", bold=True, italic=True)),
        "*checks.**Safe.***",
        "<em>checks.<strong>Safe.</strong></em>",
    ),
    (
        "underline around a highlight",
        (S("is ", underline=True), S("a link", underline=True, mark=True), S(".", underline=True)),
        "<u>is <mark>a link</mark>.</u>",
        "<u>is <mark>a link</mark>.</u>",
    ),
    (
        "bold then italic",
        (S("a", bold=True), S("b", italic=True)),
        "**a***b*",
        "<strong>a</strong><em>b</em>",
    ),
    (
        "plain between bold",
        (S("a", bold=True), S("b"), S("c", bold=True)),
        "**a**b**c**",
        "<strong>a</strong>b<strong>c</strong>",
    ),
    # Spaces join spans under the markup they carry, never under markup they lack.
    (
        "bold space between bold runs",
        (S("A", bold=True), S(" ", bold=True), S("B", bold=True)),
        "**A B**",
        "<strong>A B</strong>",
    ),
    (
        "bold run starting with a space",
        (S("A", bold=True), S(" B", bold=True)),
        "**A B**",
        "<strong>A B</strong>",
    ),
    (
        "bold run ending with a space",
        (S("A ", bold=True), S("B", bold=True)),
        "**A B**",
        "<strong>A B</strong>",
    ),
    (
        "plain space between bold runs",
        (S("A", bold=True), S(" "), S("B", bold=True)),
        "**A** **B**",
        "<strong>A</strong> <strong>B</strong>",
    ),
    (
        "italic space between bold runs",
        (S("A", bold=True), S(" ", italic=True), S("B", bold=True)),
        "**A** **B**",
        "<strong>A</strong><em> </em><strong>B</strong>",
    ),
    (
        "bold space between highlighted bold runs",
        (S("A", bold=True, mark=True), S(" ", bold=True), S("B", bold=True, mark=True)),
        "**<mark>A</mark> <mark>B</mark>**",
        "<strong><mark>A</mark> <mark>B</mark></strong>",
    ),
    (
        "bold space before a plain run",
        (S("A", bold=True, mark=True), S(" ", bold=True), S("c")),
        "**<mark>A</mark>** c",
        "<strong><mark>A</mark> </strong>c",
    ),
    (
        "end spaces",
        (S("a "), S(" word  ", bold=True), S("b", italic=True), S(" ", italic=True)),
        "a  **word**  *b* ",
        "a  <strong>word</strong>  <em>b </em>",
    ),
    (
        "bold spaces at the end",
        (S("A ", bold=True), S("  ", bold=True)),
        "**A**   ",
        "<strong>A</strong>   ",
    ),
    (
        "bold spaces at the start",
        (S("  ", bold=True), S(" A", bold=True)),
        "   **A**",
        "<strong>   A</strong>",
    ),
    (
        "underlined space",
        (S("A", underline=True), S(" ", underline=True), S("c")),
        "<u>A </u>c",
        "<u>A </u>c",
    ),
    (
        "underlined blank",
        (S("Name:"), S("    ", underline=True, bold=True)),
        "Name:<u>    </u>",
        "Name:<strong><u>    </u></strong>",
    ),
    (
        "abbreviation after a space",
        (S("See"), S(" CIA", bold=True, expansion="Central Agency")),
        'See <abbr title="Central Agency">**CIA**</abbr>',
        'See <abbr title="Central Agency"><strong>CIA</strong></abbr>',
    ),
)
MARKER_RUN = re.compile(r"[*~]+")
REOPENED_TAG = re.compile(r"</(strong|em|del|u|mark|sup|sub)><\1>")


def styled_line(spans: tuple[TextSpan, ...]) -> TextLine:
    return TextLine("".join(span.text for span in spans), spans=spans)


def marker_on_space(markdown: str) -> str | None:
    """A run of emphasis markers with whitespace or the line's edge on both sides."""
    for match in MARKER_RUN.finditer(markdown):
        before = markdown[match.start() - 1] if match.start() else " "
        after = markdown[match.end()] if match.end() < len(markdown) else " "
        if before.isspace() and after.isspace():
            return match.group()
    return None


def check_cases() -> list[str]:
    failures: list[str] = []
    for name, spans, markdown, html in CASES:
        line = styled_line(spans)
        if (got := markdown_line(line)) != markdown:
            failures.append(f"{name}: Markdown {got!r}, expected {markdown!r}")
        if (got := html_line(line)) != html:
            failures.append(f"{name}: HTML {got!r}, expected {html!r}")
    # A list item's marker is cut from the spans before they render.
    item = styled_line((S("1. "), S("A", bold=True), S("B", bold=True)))
    if (got := markdown_line(item, start=3)) != "**AB**":
        failures.append(f"list item: Markdown {got!r}, expected '**AB**'")
    return failures


def check_line(line: TextLine) -> str | None:
    markdown = markdown_line(line)
    # Markdown output does not escape a literal * or ~, which then reads as a marker.
    if "*" not in line.text and "~" not in line.text:
        if "****" in markdown or "~~~~" in markdown:
            return f"Markdown joins markers: {markdown!r}"
        if (run := marker_on_space(markdown)) is not None:
            return f"Markdown {run!r} on whitespace: {markdown!r}"
    html = html_line(line)
    if REOPENED_TAG.search(html):
        return f"HTML reopens a tag: {html!r}"
    return None


def check_corpus(patterns: tuple[str, ...]) -> list[str]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern, recursive=True)})
    if not paths:
        print("no upstream test PDFs: run git submodule update --init --recursive --depth 1")
        return []
    failures: list[str] = []
    lines = styled = 0
    for path in paths:
        try:
            with PdfDocument(path) as pdf:
                pages = pdf.extract().pages
        except Exception:  # noqa: BLE001 - a PDF core cannot open has no lines to check
            continue
        for page in pages:
            for element in page.elements:
                if not isinstance(element, Block):
                    continue
                for line in element.lines:
                    lines += 1
                    styled += len(line.spans) > 1
                    if (failure := check_line(line)) is not None:
                        failures.append(f"{Path(path).name} page {page.page_number}: {failure}")
    for failure in failures[:20]:
        print(failure)
    print(
        f"{lines} lines ({styled} of several spans) in {len(paths)} upstream PDFs, "
        f"{len(failures)} with split markup"
    )
    return failures


def main(argv: list[str]) -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    failures = check_cases()
    for failure in failures:
        print(failure)
    print(f"{len(CASES) + 1} lines built here, {len(failures)} rendered wrong")
    corpus_failures = check_corpus(FULL_CORPUS if "--all" in argv else CORPUS)
    return 1 if failures or corpus_failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

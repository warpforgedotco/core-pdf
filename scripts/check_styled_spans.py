# SPDX-License-Identifier: AGPL-3.0-only

"""Check that Markdown and HTML keep markup open across spans that share it.

A line keeps a span per text run, and runs split at every font or glyph-run change, so
neighbouring spans often share their markup. render_styled_line keeps a layer open across
the spans that share it: two bold runs `U` and `N` are `**UN**`, never `**U****N**`, and a
bold highlighted run then a bold one is `**<mark>a</mark>b**`. This checks it on lines
built here for every layer the serializer writes (bold, italic, both, strikeout,
underline, highlight, superscript, subscript and an abbreviation's expansion), for spaces
between and around spans, and for neighbours that differ, then on the upstream test PDFs:
no HTML closes a tag only to reopen it (Markdown nests the same way, so `**U****N**` is
caught there; a `****` of layers that cross, bold over `a b` and italic over `b c` in
`**a*b****c*`, is valid), and no Markdown emphasis marker sits on whitespace. On both, an
HTML parser and a CommonMark parser (markdown-it-py, with strikethrough) must read back
the text and the markup each non-space character's span carries, a line's Markdown as one
paragraph on its own and after another line, and a block's as the heading, paragraph or
list it is, so text that reads as syntax must be escaped; a page's lists, each block's
apart from its neighbours', must read as tight lists of one item per line, and a page's
first block must start a line. Seeded random lines and blocks, of characters Markdown
reads as syntax, leading and trailing spaces and tabs, and random markup, go through the
same checks. A missing corpus file fails the check.

uv run --all-packages --group test python scripts/check_styled_spans.py [--all]

Without --all, the corpus scan reads the upstream PDFs that showed split markup; with it,
every PDF in the checkouts CI's core shard has.
"""

from __future__ import annotations

import glob
import html as html_module
import logging
import random
import re
import sys
import warnings
from collections import Counter
from collections.abc import Iterable
from html.parser import HTMLParser
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from core_pdf import PdfDocument
from core_pdf.impl.output.model import Block, BlockKind, Document, Page, TextLine, TextSpan
from core_pdf.impl.output.serialize import (
    block_to_markdown,
    document_to_markdown,
    html_line,
    markdown_line,
    markdown_list_item,
    page_to_markdown,
)

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
    # Layout gives an abbreviation one span, its runs joined, so two spans with one
    # expansion are two abbreviations.
    (
        "two abbreviations with one expansion",
        (S("CI", expansion="Central Agency"), S("A", expansion="Central Agency")),
        '<abbr title="Central Agency">CI</abbr><abbr title="Central Agency">A</abbr>',
        '<abbr title="Central Agency">CI</abbr><abbr title="Central Agency">A</abbr>',
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
        "a  **word**  *b*",
        "a  <strong>word</strong>  <em>b </em>",
    ),
    (
        "bold spaces at the end",
        (S("A ", bold=True), S("  ", bold=True)),
        "**A**",
        "<strong>A   </strong>",
    ),
    (
        "bold spaces at the start",
        (S("  ", bold=True), S(" A", bold=True)),
        "**A**",
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
    # A bold blank keeps a bold neighbour's markup only where the two share it, and
    # emphasis a crossing layer splits reopens after the blank, not before it.
    (
        "bold blank after italic bold",
        (S("x", italic=True), S("A", italic=True, bold=True), S(" ", bold=True), S("B", bold=True)),
        "*x**A*** **B**",
        "<em>x<strong>A</strong></em><strong> B</strong>",
    ),
    # Whitespace inside a layer that stays open stays inside it: an underlined form blank.
    (
        "underlined blank after an underlined label",
        (S("Name: ", underline=True), S("    ", underline=True)),
        "<u>Name:     </u>",
        "<u>Name:     </u>",
    ),
    # Markdown emphasis is written as HTML where CommonMark would not read its delimiters:
    # `</mark>**c` cannot close, and in `**a***b*` the run between closes and opens.
    (
        "bold closing after a tag before a letter",
        (S("a", bold=True), S("b", bold=True, mark=True), S("c")),
        "<strong>a<mark>b</mark></strong>c",
        "<strong>a<mark>b</mark></strong>c",
    ),
    (
        "bold punctuation inside a word",
        (S("a"), S("(b)", bold=True), S("c")),
        "a<strong>(b)</strong>c",
        "a<strong>(b)</strong>c",
    ),
    (
        "struck then bold",
        (S("a", strikeout=True), S("b", bold=True), S(" c", strikeout=True)),
        "~~a~~**b** ~~c~~",
        "<del>a</del><strong>b</strong> <del>c</del>",
    ),
    # Markdown escapes text CommonMark would read as markup.
    (
        "bold backslash",
        (S("x\\", bold=True), S("y")),
        "<strong>x\\\\</strong>y",
        "<strong>x\\</strong>y",
    ),
    (
        "literal markup characters",
        (S("a*b_c~d`e<f>g[h]i&amp; &j", bold=True),),
        "**a\\*b\\_c\\~d\\`e\\<f>g\\[h\\]i\\&amp; &j**",
        "<strong>a*b_c~d`e&lt;f&gt;g[h]i&amp;amp; &amp;j</strong>",
    ),
    ("heading at the line start", (S("# 1"),), "\\# 1", "# 1"),
    (
        "list item at the line start",
        (S("12"), S(". x"), S(" y", bold=True)),
        "12\\. x **y**",
        "12. x <strong>y</strong>",
    ),
    ("indented heading", (S("  #"), S(" x")), "\\# x", "  # x"),
    # Four columns of indentation would open a code block, two spaces at the end a line
    # break: Markdown drops a line's end spaces and tabs.
    ("indented code", (S("    # x"),), "\\# x", "    # x"),
    ("tab before a marker", (S("\t- x  "),), "\\- x", "\t- x  "),
    (
        "tab after spaces",
        (S("  \t"), S("1. x", bold=True)),
        "**1. x**",
        "  \t<strong>1. x</strong>",
    ),
    ("rule at the line start", (S("---"),), "\\---", "---"),
    ("number not starting a list", (S("1.5 x"),), "1.5 x", "1.5 x"),
    ("marker after markup", (S("-", bold=True), S(" x")), "**-** x", "<strong>-</strong> x"),
    (
        "abbreviation title to escape",
        (S("AT&T", expansion='"A" & <T>'),),
        '<abbr title="&quot;A&quot; &amp; &lt;T&gt;">AT&T</abbr>',
        '<abbr title="&quot;A&quot; &amp; &lt;T&gt;">AT&amp;T</abbr>',
    ),
    (
        "abbreviation after a space",
        (S("See"), S(" CIA", bold=True, expansion="Central Agency")),
        'See <abbr title="Central Agency">**CIA**</abbr>',
        'See <abbr title="Central Agency"><strong>CIA</strong></abbr>',
    ),
)
MARKER_RUN = re.compile(r"[*~]+")
ESCAPED_RE = re.compile(r"\\.")
REOPENED_TAG = re.compile(r"</(strong|em|del|u|mark|sup|sub)><\1>")


def styled_line(spans: tuple[TextSpan, ...]) -> TextLine:
    return TextLine("".join(span.text for span in spans), spans=spans)


def marker_on_space(markdown: str) -> str | None:
    """A run of emphasis markers with whitespace or the line's edge on both sides."""
    for match in MARKER_RUN.finditer(markdown):
        before = markdown[match.start() - 1] if match.start() else " "
        after = markdown[match.end()] if match.end() < len(markdown) else " "
        if SPACE_RE.match(before) and SPACE_RE.match(after):
            return match.group()
    return None


# The markup each Markdown and HTML marker stands for.
TAG_STYLES = {
    "strong": "bold",
    "em": "italic",
    "del": "strikeout",
    "u": "underline",
    "mark": "mark",
    "sup": "sup",
    "sub": "sub",
}
MARKDOWN_STYLES = {"strong": "bold", "em": "italic", "s": "strikeout"}
HTML_TAG_RE = re.compile(r'<(/?)(\w+)(?: title="([^"]*)")?>')
MARKDOWN = MarkdownIt("commonmark").enable("strikethrough")
PARAGRAPH = ["paragraph_open", "inline", "paragraph_close"]
HEADING = ["heading_open", "inline", "heading_close"]
LIST_TOKENS = frozenset(
    (
        "bullet_list_open",
        "bullet_list_close",
        "ordered_list_open",
        "ordered_list_close",
        "list_item_open",
        "list_item_close",
        *PARAGRAPH,
    )
)
# CommonMark's whitespace (2.1): Unicode Zs, tab, line feed, form feed and carriage return.
SPACE_RE = re.compile("[\t\n\f\r \u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")

type Styled = list[tuple[str, frozenset[str]]]


class StyleCounter:
    """The markup open at each point of a parsed line."""

    def __init__(self) -> None:
        self.open: Counter[str] = Counter()
        self.titles: list[str] = []
        self.styled: Styled = []

    def change(self, style: str, opening: bool) -> None:
        self.open[style] += 1 if opening else -1

    def tag(self, name: str, closing: bool, title: str | None) -> None:
        if name == "abbr":
            if closing:
                self.change(self.titles.pop() if self.titles else "abbr:", False)
            else:
                self.titles.append(f"abbr:{title}")
                self.change(self.titles[-1], True)
        elif name in TAG_STYLES:
            self.change(TAG_STYLES[name], not closing)

    def text(self, text: str) -> None:
        current = frozenset(style for style, count in self.open.items() if count > 0)
        self.styled.extend(
            (character, current) for character in text if not SPACE_RE.match(character)
        )


def markdown_styles(markdown: str) -> Styled | None:
    """The non-space characters CommonMark reads in a line, each with its markup, or None
    where the line, alone or after another, is not read as one paragraph."""
    if not markdown.strip(" "):
        return []
    blocks = MARKDOWN.parse(markdown)
    if [token.type for token in blocks] != PARAGRAPH or [
        token.type for token in MARKDOWN.parse(f"a\n{markdown}")
    ] != PARAGRAPH:
        return None
    return inline_styles(blocks[1].children or ())


def inline_styles(children: Iterable[Token]) -> Styled:
    counter = StyleCounter()
    for token in children:
        name, _, side = token.type.rpartition("_")
        if name in MARKDOWN_STYLES and side in {"open", "close"}:
            counter.change(MARKDOWN_STYLES[name], side == "open")
        elif token.type == "html_inline" and (match := HTML_TAG_RE.fullmatch(token.content)):
            title = match.group(3)
            counter.tag(
                match.group(2),
                bool(match.group(1)),
                None if title is None else html_module.unescape(title),
            )
        else:
            counter.text(token.content)
    return counter.styled


class HtmlStyles(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.counter = StyleCounter()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.counter.tag(tag, False, dict(attrs).get("title"))

    def handle_endtag(self, tag: str) -> None:
        self.counter.tag(tag, True, None)

    def handle_data(self, data: str) -> None:
        self.counter.text(data)


def html_styles(html: str) -> Styled:
    parser = HtmlStyles()
    parser.feed(html)
    parser.close()
    return parser.counter.styled


def span_styles(line: TextLine, start: int = 0) -> Styled:
    """The line's non-space characters from start, each with the markup of its span."""
    styled: Styled = []
    for span in line.styled_spans():
        styles = frozenset(
            style
            for style, flag in (
                ("bold", span.bold),
                ("italic", span.italic),
                ("strikeout", span.strikeout),
                ("underline", span.underline),
                ("mark", span.mark),
                ("sup", span.superscript),
                ("sub", span.subscript and not span.superscript),
                (f"abbr:{span.expansion}", span.expansion is not None),
            )
            if flag
        )
        styled.extend(
            (character, styles) for character in span.text if not SPACE_RE.match(character)
        )
    skipped = sum(not SPACE_RE.match(character) for character in line.text[:start])
    return styled[skipped:]


def structure_failure(line: TextLine, start: int = 0) -> str | None:
    """How the markup an HTML and a CommonMark parser read back differs from the spans'."""
    expected = span_styles(line, start)
    html = html_line(line, start=start)
    if html_styles(html) != expected:
        return f"HTML reads back other markup: {html!r}"
    markdown = markdown_line(line, start=start)
    if markdown_styles(markdown) != expected:
        return f"Markdown reads back other text or markup: {markdown!r}"
    return None


def block_failure(block: Block, level: int | None = None) -> str | None:
    """How the block CommonMark reads in a block's Markdown differs from the block, a
    heading's level checked where given."""
    markdown = block_to_markdown(block)
    tokens = MARKDOWN.parse(markdown)
    types = [token.type for token in tokens]
    starts = [0] * len(block.lines)
    if block.kind is BlockKind.HEADING:
        shape = types == HEADING and (level is None or tokens[0].tag == f"h{level}")
    elif block.kind is BlockKind.LIST:
        shape = LIST_TOKENS.issuperset(types) and types.count("list_item_open") == len(block.lines)
        starts = [markdown_list_item(line)[1] for line in block.lines]
    else:
        shape = types == PARAGRAPH or not types and not markdown.strip()
    if not shape:
        return f"{block.kind} Markdown reads as {types}: {markdown!r}"
    read = [
        styled
        for token in tokens
        if token.type == "inline"
        for styled in inline_styles(token.children or ())
    ]
    expected = [
        styled
        for line, start in zip(block.lines, starts, strict=True)
        for styled in span_styles(line, start)
    ]
    if read != expected:
        return f"{block.kind} Markdown reads back other text or markup: {markdown!r}"
    return None


def block_of(kind: BlockKind, *lines: TextLine, level: int | None = None, order: int = 0) -> Block:
    return Block(order, kind, lines=lines, level=level)


def lists_failure(markdown: str, items: int, lists: int | None = None) -> str | None:
    """How the lists CommonMark reads in a page's or document's Markdown differ from its
    list blocks: every list tight, one item per list line, and as many lists where given."""
    tokens = MARKDOWN.parse(markdown)
    depth = 0
    for token in tokens:
        if token.type in {"bullet_list_open", "ordered_list_open"}:
            depth += 1
        elif token.type in {"bullet_list_close", "ordered_list_close"}:
            depth -= 1
        elif token.type == "paragraph_open" and depth and not token.hidden:
            return f"Markdown reads a loose list: {markdown!r}"
    read = sum(token.type == "list_item_open" for token in tokens)
    if read != items:
        return f"Markdown reads {read} list items, not {items}: {markdown!r}"
    opened = sum(token.type in {"bullet_list_open", "ordered_list_open"} for token in tokens)
    if lists is not None and opened != lists:
        return f"Markdown reads {opened} lists, not {lists}: {markdown!r}"
    return None


def list_lines(page: Page) -> int:
    return sum(
        len(element.lines)
        for element in page.elements
        if isinstance(element, Block) and element.kind is BlockKind.LIST
    )


def check_pages() -> list[str]:
    """Neighbouring list blocks stay apart, and a page's first block starts a line."""
    bullets = (
        block_of(BlockKind.LIST, plain("• a"), plain("• b"), order=0),
        block_of(BlockKind.LIST, plain("• c"), order=1),
        block_of(BlockKind.LIST, plain("1. d"), plain("2) e"), order=2),
        block_of(BlockKind.LIST, plain("3. f"), order=3),
    )
    page = Page(1, blocks=bullets)
    failures: list[str] = []
    expected = "- a\n- b\n\n* c\n\n1. d\n2. e\n\n3) f"
    if (got := page_to_markdown(page)) != expected:
        failures.append(f"neighbouring lists: Markdown {got!r}, expected {expected!r}")
    if (failure := lists_failure(page_to_markdown(page), 6, lists=4)) is not None:
        failures.append(f"neighbouring lists: {failure}")
    second = Page(
        2,
        blocks=(
            block_of(BlockKind.HEADING, plain("Next"), level=2, order=0),
            block_of(BlockKind.LIST, plain("- g"), order=1),
        ),
    )
    document = Document(pages=(Page(1, blocks=bullets[:1]), second, Page(3, blocks=bullets[:1])))
    markdown = document_to_markdown(document)
    if (failure := lists_failure(markdown, 5, lists=3)) is not None:
        failures.append(f"page joins: {failure}")
    headings = [token.tag for token in MARKDOWN.parse(markdown) if token.type == "heading_open"]
    if headings != ["h2"]:
        failures.append(f"page joins: Markdown reads headings {headings}: {markdown!r}")
    return failures


def plain(text: str) -> TextLine:
    return TextLine(text)


# (name, block, Markdown, heading level CommonMark must read or None)
BLOCK_CASES: tuple[tuple[str, Block, str, int | None], ...] = (
    (
        "heading ending in a # run",
        block_of(BlockKind.HEADING, plain("Item #"), level=2),
        "## Item \\#",
        2,
    ),
    ("heading of a # only", block_of(BlockKind.HEADING, plain("#"), level=1), "# \\#", 1),
    (
        "heading of two lines",
        block_of(BlockKind.HEADING, plain("Annual  "), plain("  Report"), level=3),
        "### Annual Report",
        3,
    ),
    ("heading past level 6", block_of(BlockKind.HEADING, plain("x"), level=9), "###### x", 6),
    ("heading of level 0", block_of(BlockKind.HEADING, plain("x"), level=0), "# x", 1),
    ("heading of no level", block_of(BlockKind.HEADING, plain("x")), "## x", 2),
    (
        "paragraph lines with end spaces",
        block_of(BlockKind.PARAGRAPH, plain("one  "), plain("    two"), plain("   "), plain("3.")),
        "one\ntwo\n3\\.",
        None,
    ),
    (
        "setext underline",
        block_of(BlockKind.PARAGRAPH, plain("Title"), plain("====="), plain("---")),
        "Title\n\\=====\n\\---",
        None,
    ),
    (
        "bullet glyphs",
        block_of(BlockKind.LIST, plain("• one"), plain("▪ two"), plain("◦ three"), plain("- four")),
        "- one\n- two\n- three\n- four",
        None,
    ),
    (
        "letter markers",
        block_of(BlockKind.LIST, plain("a) one"), plain("b. two")),
        "- a) one\n- b. two",
        None,
    ),
    (
        "numbered markers",
        block_of(BlockKind.LIST, plain("1. one"), plain("2) two"), plain("10. # ten")),
        "1. one\n2. two\n10. \\# ten",
        None,
    ),
    (
        "markers without a space after them",
        block_of(BlockKind.LIST, plain("1.5 million"), plain("-5 degrees"), plain("•")),
        "- 1.5 million\n- \\-5 degrees\n- ",
        None,
    ),
    (
        "unmarked list line",
        block_of(BlockKind.LIST, plain("    plain *item*")),
        "- plain \\*item\\*",
        None,
    ),
)
FUZZ_PIECES = (
    "a", "Z", "7", "é", "—", "•", "▪", " ", "  ", "\t", "    ", "\u00a0", "\\", "*", "**",
    "_", "~", "~~", "`", "<", ">", "<u>", "[", "]", "&", "&amp;", "&#35;", "#", "##", "+",
    "-", "---", "=", "1.", "2)", "a)", "!", "(", ")", ".", ":", '"', "'",
)  # fmt: skip
FUZZ_STYLES = ("bold", "italic", "strikeout", "underline", "mark", "superscript", "subscript")


def fuzz_line(rng: random.Random) -> TextLine:
    spans = []
    for _ in range(rng.randint(1, 5)):
        text = "".join(rng.choice(FUZZ_PIECES) for _ in range(rng.randint(1, 4)))
        styles = {style: rng.random() < 0.25 for style in FUZZ_STYLES}
        expansion = rng.choice((None,) * 18 + ("x", '"A" & <B>'))
        spans.append(TextSpan(text, **styles, expansion=expansion))
    return styled_line(tuple(spans))


def check_fuzz(count: int, seed: int) -> list[str]:
    """Random lines, and blocks of them, through the line and block checks."""
    rng = random.Random(seed)
    failures: list[str] = []
    lines: list[TextLine] = []
    for index in range(count):
        line = fuzz_line(rng)
        lines.append(line)
        if (failure := check_line(line)) is not None:
            failures.append(f"random line {index}: {failure}")
        if len(lines) == 3:
            kind = rng.choice((BlockKind.PARAGRAPH, BlockKind.HEADING, BlockKind.LIST))
            block = block_of(kind, *lines, level=rng.randint(1, 7))
            if (failure := block_failure(block)) is not None:
                failures.append(f"random block {index}: {failure}")
            lines = []
    for failure in failures[:20]:
        print(failure)
    print(f"{count} random lines (seed {seed}), {len(failures)} misread")
    return failures


def check_cases() -> list[str]:
    failures: list[str] = []
    for name, spans, markdown, html in CASES:
        line = styled_line(spans)
        if (got := markdown_line(line)) != markdown:
            failures.append(f"{name}: Markdown {got!r}, expected {markdown!r}")
        if (got := html_line(line)) != html:
            failures.append(f"{name}: HTML {got!r}, expected {html!r}")
        if (failure := structure_failure(line)) is not None:
            failures.append(f"{name}: {failure}")
    # A list item's marker is cut from the spans before they render.
    item = styled_line((S("1. "), S("A", bold=True), S("B", bold=True)))
    if (got := markdown_line(item, start=3)) != "**AB**":
        failures.append(f"list item: Markdown {got!r}, expected '**AB**'")
    if (failure := structure_failure(item, start=3)) is not None:
        failures.append(f"list item: {failure}")
    for name, block, markdown, level in BLOCK_CASES:
        if (got := block_to_markdown(block)) != markdown:
            failures.append(f"{name}: Markdown {got!r}, expected {markdown!r}")
        if (failure := block_failure(block, level)) is not None:
            failures.append(f"{name}: {failure}")
    failures += check_pages()
    return failures


def check_line(line: TextLine) -> str | None:
    markdown = markdown_line(line)
    # A literal * or ~ is escaped: with escapes read as letters, only markers remain.
    if (run := marker_on_space(ESCAPED_RE.sub("x", markdown))) is not None:
        return f"Markdown {run!r} on whitespace: {markdown!r}"
    html = html_line(line)
    if REOPENED_TAG.search(html):
        return f"HTML reopens a tag: {html!r}"
    return structure_failure(line)


def check_corpus(patterns: tuple[str, ...]) -> list[str]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern, recursive=True)})
    missing = [pattern for pattern in patterns if not glob.glob(pattern, recursive=True)]
    if missing:
        for pattern in missing:
            print(f"missing {pattern}: run git submodule update --init --recursive --depth 1")
        return [f"missing {pattern}" for pattern in missing]
    failures: list[str] = []
    lines = styled = blocks = 0
    for path in paths:
        try:
            with PdfDocument(path) as pdf:
                pages = pdf.extract().pages
        except Exception:  # noqa: BLE001 - a PDF core cannot open has no lines to check
            continue
        for page in pages:
            if (failure := lists_failure(page_to_markdown(page), list_lines(page))) is not None:
                failures.append(f"{Path(path).name} page {page.page_number}: {failure}")
            for element in page.elements:
                if not isinstance(element, Block):
                    continue
                blocks += 1
                if (failure := block_failure(element)) is not None:
                    failures.append(f"{Path(path).name} page {page.page_number}: {failure}")
                for line in element.lines:
                    lines += 1
                    styled += len(line.spans) > 1
                    if (failure := check_line(line)) is not None:
                        failures.append(f"{Path(path).name} page {page.page_number}: {failure}")
    for failure in failures[:20]:
        print(failure)
    print(
        f"{blocks} blocks, {lines} lines ({styled} of several spans) in {len(paths)} "
        f"upstream PDFs, {len(failures)} with split or misread markup"
    )
    return failures


def main(argv: list[str]) -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    failures = check_cases()
    for failure in failures:
        print(failure)
    print(
        f"{len(CASES) + 1} lines and {len(BLOCK_CASES)} blocks built here, "
        f"{len(failures)} rendered wrong"
    )
    fuzz_failures = check_fuzz(3000, seed=20261120)
    corpus_failures = check_corpus(FULL_CORPUS if "--all" in argv else CORPUS)
    return 1 if failures or fuzz_failures or corpus_failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# SPDX-License-Identifier: AGPL-3.0-only
"""List markers and content use separate raw-text and styled-span boundaries."""

import pytest

from core_pdf.impl._impl.output.model import Block, BlockKind, TextLine, TextSpan
from core_pdf.impl._impl.output.serialize import block_to_html, block_to_markdown


@pytest.mark.parametrize("prefix", ["", "- ", "* ", "• ", "▪ ", "◦ ", "A) ", "12. ", "  23)\t"])
def test_list_content_retains_inline_styles_without_styling_its_marker(prefix: str) -> None:
    block = Block(
        0,
        BlockKind.LIST,
        (TextLine(f"{prefix}Important", spans=(TextSpan(f"{prefix}Important", bold=True),)),),
    )

    assert block_to_markdown(block) == f"{prefix or '- '}**Important**"
    assert block_to_html(block) == (
        '<ul data-block-kind="list"><li><strong>Important</strong></li></ul>'
    )


def test_list_prefix_can_cross_span_boundaries_before_html_escaping() -> None:
    block = Block(
        0,
        BlockKind.LIST,
        (
            TextLine(
                "  12) A&B <C>",
                spans=(
                    TextSpan("  1", bold=True),
                    TextSpan("2)", italic=True),
                    TextSpan(" A&B", underline=True),
                    TextSpan(" "),
                    TextSpan("<C>", strikeout=True),
                ),
            ),
        ),
    )

    assert block_to_markdown(block) == "  12) <u>A&B</u> ~~<C>~~"
    assert block_to_html(block) == (
        '<ul data-block-kind="list"><li><u>A&amp;B</u> <del>&lt;C&gt;</del></li></ul>'
    )


def test_unmarked_list_line_preserves_line_level_styles() -> None:
    block = Block(0, BlockKind.LIST, (TextLine("Note", bold=True, italic=True, mark=True),))

    assert block_to_markdown(block) == "- ***<mark>Note</mark>***"
    assert block_to_html(block) == (
        '<ul data-block-kind="list"><li><em><strong><mark>Note</mark></strong></em></li></ul>'
    )


def test_empty_list_item_does_not_render_marker_style_as_content() -> None:
    block = Block(0, BlockKind.LIST, (TextLine("- ", bold=True), TextLine("")))

    assert block_to_markdown(block) == "- \n- "
    assert block_to_html(block) == '<ul data-block-kind="list"><li></li><li></li></ul>'


def test_non_list_blocks_retain_text_that_looks_like_a_marker() -> None:
    line = TextLine("12. Important", bold=True)
    block = Block(0, BlockKind.PARAGRAPH, (line,))

    assert block_to_markdown(block) == "**12. Important**"
    assert block_to_html(block) == (
        '<p data-block-kind="paragraph"><strong>12. Important</strong></p>'
    )


@pytest.mark.parametrize("original", ["- raw", "123. raw", "raw"])
def test_normalized_list_text_wins_over_stale_span_text(original: str) -> None:
    line = TextLine("- repaired", italic=True, spans=(TextSpan(original, bold=True),))
    block = Block(0, BlockKind.LIST, (line,))

    assert block_to_markdown(block) == "- *repaired*"
    assert block_to_html(block) == '<ul data-block-kind="list"><li><em>repaired</em></li></ul>'


def test_unmarked_normalized_list_text_wins_over_stale_spans() -> None:
    line = TextLine("repaired", spans=(TextSpan("raw", bold=True),))
    block = Block(0, BlockKind.LIST, (line,))

    assert block_to_markdown(block) == "- repaired"
    assert block_to_html(block) == '<ul data-block-kind="list"><li>repaired</li></ul>'

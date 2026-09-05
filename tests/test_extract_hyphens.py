# SPDX-License-Identifier: AGPL-3.0-only
"""Literal line-end hyphens survive extraction and every output projection.

The generated PDFs were checked with qpdf 12.3.2 and Poppler 26.07.0
pdftotext -layout, which retains each hyphen and physical line boundary.
"""

import pytest

from core_pdf.impl._impl.output.serialize import line_to_json_dict, page_to_html, page_to_markdown
from tests.helpers.pdf_bytes import one_page_pdf, open_pdf


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Use a cost-", "effective solution."),
        ("See foo-", "bar for details."),
        ("Good gover-", "nance responds."),
    ],
)
def test_literal_line_end_hyphens_remain_in_text_words_and_serialization(
    first: str, second: str
) -> None:
    content = b"\n".join(
        f"BT /F1 12 Tf 40 {700 - index * 14} Td ({line}) Tj ET".encode()
        for index, line in enumerate((first, second))
    )
    with open_pdf(one_page_pdf(content)) as document:
        page = document.pages[0].extract()

    assert page.text == f"{first}\n{second}"
    assert len(page.blocks) == 1
    line = page.blocks[0].lines[0]
    assert line.words[-1].text == first.split()[-1]
    assert line.words[-1].bbox is not None
    assert line_to_json_dict(line)["text"] == first
    assert first in page_to_markdown(page)
    assert first in page_to_html(page)

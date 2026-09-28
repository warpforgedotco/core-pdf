# SPDX-License-Identifier: AGPL-3.0-only

import pytest

from core_pdf.impl.runs import TextRun


def make_run(text: str) -> TextRun:
    return TextRun(text, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 10.0, 3.0, 0, 0, 0)


@pytest.mark.parametrize("text", ["", " ", "Hi", " Hi ", "\tA\n", "a b"])
def test_derived_text_matches_strip(text: str) -> None:
    run = make_run(text)
    assert run.stripped_text == text.strip()
    assert run.has_text is bool(text.strip())


def test_set_text_refreshes_derived_text() -> None:
    run = make_run(" ")
    run.set_text(" word")
    assert (run.stripped_text, run.has_text, run.text_is_space) == ("word", True, False)
    assert make_run("x").replace(text="  ").has_text is False

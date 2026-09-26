"""TextRun.with_font_name is replace(font_name=...) and nothing else."""

from copy import copy

from core_pdf.impl.runs import TextRun


def test_with_font_name_matches_replace_in_every_field() -> None:
    run = TextRun(
        "word",
        1.0,
        2.0,
        30.0,
        14.0,
        1.0,
        2.0,
        12.0,
        3.0,
        4,
        5,
        1,
        "F5",
        False,
        90,
        True,
        True,
        True,
        7,
        (0.0, 0.5, 1.0),
        (1.0, 2.0, 30.0, 14.0),
        (1.5, 2.0, 29.0, 13.0),
        (1.0, 2.0, 30.0, 2.0),
        (("source", "native_text"),),
        0.75,
        (),
    )
    relabelled = run.with_font_name("TimesNewRomanPSMT")
    expected = run.replace(font_name="TimesNewRomanPSMT")
    assert type(relabelled) is TextRun
    assert repr(relabelled) == repr(expected)
    assert run.font_name == "F5"
    assert repr(copy(run)) == repr(run)

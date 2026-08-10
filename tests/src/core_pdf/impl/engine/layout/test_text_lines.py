from core_pdf.impl.engine.layout.text_lines import (
    repair_table_split_word_boundaries,
    should_join_plausible_split_word,
)


def test_table_word_fragments_can_join_when_the_gap_is_tight() -> None:
    assert should_join_plausible_split_word(
        "Vo",
        "lume",
        x_gap=1.5,
        height=10.2,
        space_width=5.0,
        prev_visible=True,
        visible=True,
        allow_short_prefix=True,
    )


def test_short_word_fragments_still_require_explicit_opt_in() -> None:
    assert not should_join_plausible_split_word(
        "Vo",
        "lume",
        x_gap=1.5,
        height=10.2,
        space_width=5.0,
        prev_visible=True,
        visible=True,
    )


def test_table_split_word_repair_joins_fragmented_form_labels() -> None:
    assert (
        repair_table_split_word_boundaries("SERVICE MODUL E Tempera ture Ox idizer We ight")
        == "SERVICE MODULE Temperature Oxidizer Weight"
    )


def test_table_split_word_repair_preserves_normal_word_boundaries() -> None:
    assert repair_table_split_word_boundaries("Primary Fuel Secondary Oxidizer") == (
        "Primary Fuel Secondary Oxidizer"
    )


def test_table_split_word_repair_removes_space_before_punctuation() -> None:
    assert repair_table_split_word_boundaries("Coef . Var . 105 .1") == "Coef. Var. 105.1"

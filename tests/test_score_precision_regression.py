"""Native tokenizer/scorer invariants for the SCORE-Bench evaluation."""

from runpy import run_path

from tests.helpers.paths import REPO_ROOT

tokenize = run_path(
    str(REPO_ROOT / "scripts/score_unstructured_bench.py"),
    run_name="score_precision_regression_tests",
)["tokenize"]


def test_tokenize_attaches_superscript_unit_into_single_token() -> None:
    # NFKC normalises the superscript-three (U+00B3) digit into an ASCII "3",
    # so a correctly reattached unit glyph "in³" becomes the single token
    # "in3" -- exactly what the reference text carries.  A space-decomposed
    # "in 3" instead yields two tokens and is counted as a miss.
    assert tokenize("in\u00b3") == ["in3"]
    assert tokenize("in 3") == ["in", "3"]


def test_tokenize_treats_orphan_symbol_as_own_token() -> None:
    # A lone non-alphanumeric character is its own token, so an orphan symbol
    # block (e.g. Braille "⠭") in the prediction is counted as an extra
    # token against precision. Emission preserves decoded symbols even when
    # a corpus reference omits them; spelling alone does not establish noise.
    assert tokenize("Amendment 110\n\u282d\n") == ["amendment", "110", "\u282d"]

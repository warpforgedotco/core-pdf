from pathlib import Path
from runpy import run_path

tokenize = run_path(
    str(Path(__file__).parents[1] / "scripts" / "score_score_bench.py"),
    run_name="score_score_bench_tests",
)["tokenize"]


def test_tokenize_normalizes_compatible_unicode_forms() -> None:
    assert tokenize("–12V in³ ‘quoted’") == tokenize("-12V in3 'quoted'")

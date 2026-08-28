"""Precision regression harness for the SCORE-Bench content evaluation.

These tests guard against regressions in extraction precision on the
SCORE-Bench corpus.  They are split into two layers:

* Fast unit tests (always run) pin down the tokenizer/scorer invariants that
  precision depends on -- e.g. that a correctly attached superscript unit
  (``in\u00b3``) collapses to a single ``in3`` token while a split
  ``in 3`` does not.

* A slow integration test that is gated behind the
  ``CORE_PDF_PRECISION_REGRESSION`` environment variable.  It re-runs the
  full SCORE-Bench evaluation, partitions the cases into the deterministic
  train/holdout split (``score_bench_partition``), and asserts that mean
  precision on each split stays above a safety floor.  The floors sit below
  the committed baseline (train 0.958 / holdout 0.970 as of this writing) so
  that environment variance does not flake CI, while still catching
  catastrophic precision regressions.

Enable the integration test with::

    CORE_PDF_PRECISION_REGRESSION=1 uv run pytest tests/test_score_precision_regression.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from runpy import run_path
from typing import Any, cast

import pytest

from tests.support.paths import REPO_ROOT

SCORE_SCRIPT = REPO_ROOT / "scripts" / "score_unstructured_bench.py"

_score_bench = run_path(
    str(SCORE_SCRIPT),
    run_name="score_precision_regression_tests",
)
tokenize = _score_bench["tokenize"]
score_bench_partition = _score_bench["score_bench_partition"]
SCORE_BENCH_ROOT = _score_bench["SCORE_BENCH_ROOT"]


# Committed baselines (mean precision over the deterministic train/holdout
# partition of the current SCORE-Bench fixture set, measured on Python 3.13
# with the current Tesseract).  The integration test floors below are set
# generously below these to tolerate environment variance while still
# detecting meaningful regressions.
TRAIN_PRECISION_BASELINE = 0.958
HOLDOUT_PRECISION_BASELINE = 0.970

# Safety floors that catch catastrophic regressions regardless of machine.
TRAIN_PRECISION_FLOOR = 0.93
HOLDOUT_PRECISION_FLOOR = 0.93
MIN_PRECISION_FLOOR = 0.30

SKIP_INTEGRATION = not os.environ.get("CORE_PDF_PRECISION_REGRESSION")


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
    # token against precision.  Dropping such blocks (see
    # ``internal_corrupt_native_block``) is therefore what protects precision.
    assert tokenize("Amendment 110\n\u282d\n") == ["amendment", "110", "\u282d"]


@pytest.mark.skipif(
    SKIP_INTEGRATION,
    reason=(
        "set CORE_PDF_PRECISION_REGRESSION=1 to run the full SCORE-Bench "
        "regression (slow, ~5 minutes)."
    ),
)
def test_precision_regression_partition(tmp_path: Path) -> None:
    if not SCORE_BENCH_ROOT.exists():
        cast(Any, pytest.skip)(f"SCORE-Bench fixtures not present at {SCORE_BENCH_ROOT}")

    json_output = tmp_path / "precision_regression.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCORE_SCRIPT),
            "--partition",
            "all",
            "--json-output",
            str(json_output),
            "--no-html-output",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=1200,
    )
    if result.returncode != 0:
        cast(Any, pytest.fail)(
            f"SCORE-Bench evaluation failed:\n{result.stderr[-4000:]}",
            pytrace=False,
        )

    cases = json.loads(json_output.read_text())
    scored = {record["stem"]: record for record in cases if record.get("status") == "ok"}
    if not scored:
        cast(Any, pytest.skip)("no scored cases produced")

    train = [r for r in scored.values() if score_bench_partition(r["stem"]) == "train"]
    holdout = [r for r in scored.values() if score_bench_partition(r["stem"]) == "holdout"]
    assert train, "train partition is empty"
    assert holdout, "holdout partition is empty"

    train_mean = sum(r["precision"] for r in train) / len(train)
    holdout_mean = sum(r["precision"] for r in holdout) / len(holdout)
    overall_min = min(r["precision"] for r in scored.values())

    assert train_mean >= TRAIN_PRECISION_FLOOR, (
        f"train precision {train_mean:.4f} below floor "
        f"{TRAIN_PRECISION_FLOOR} (baseline {TRAIN_PRECISION_BASELINE})"
    )
    assert holdout_mean >= HOLDOUT_PRECISION_FLOOR, (
        f"holdout precision {holdout_mean:.4f} below floor "
        f"{HOLDOUT_PRECISION_FLOOR} (baseline {HOLDOUT_PRECISION_BASELINE})"
    )
    assert overall_min >= MIN_PRECISION_FLOOR, (
        f"min precision {overall_min:.4f} below floor {MIN_PRECISION_FLOOR}"
    )

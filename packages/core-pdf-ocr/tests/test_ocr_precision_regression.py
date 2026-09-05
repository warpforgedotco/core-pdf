"""Optional OCR precision regression over the SCORE-Bench train/holdout split.

Tokenizer invariants live in the native suite. This integration test preserves
OCR extraction precision floors and runs the companion scorer. Enable it with::

    CORE_PDF_PRECISION_REGRESSION=1 uv run --all-packages pytest \
      packages/core-pdf-ocr/tests/test_ocr_precision_regression.py -q
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

from tests.helpers.paths import REPO_ROOT, require_fixture

ROOT = REPO_ROOT
SCORE_SCRIPT = ROOT / "packages/core-pdf-ocr/scripts/score_unstructured_bench.py"
SHARED_SCORE_SCRIPT = ROOT / "scripts/score_unstructured_bench.py"

_score_bench = run_path(
    str(SHARED_SCORE_SCRIPT),
    run_name="score_precision_regression_tests",
)
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


@pytest.mark.skipif(
    SKIP_INTEGRATION,
    reason=(
        "set CORE_PDF_PRECISION_REGRESSION=1 to run the full SCORE-Bench "
        "regression (slow, ~5 minutes)."
    ),
)
def test_precision_regression_partition(tmp_path: Path) -> None:
    require_fixture(SCORE_BENCH_ROOT, f"SCORE-Bench fixtures not present at {SCORE_BENCH_ROOT}")

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
        cwd=str(ROOT),
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

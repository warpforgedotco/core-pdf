# SPDX-License-Identifier: AGPL-3.0-only
"""Fixture locations, resolved once so test files do not count ``parents[n]``."""

from __future__ import annotations

from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_DIR.parent
FIXTURES = TESTS_DIR / "fixtures"
SNAPSHOTS = TESTS_DIR / "snapshots"
SCORE_BENCH = FIXTURES / "SCORE-Bench" / "src"
FONT_PROGRAMS = FIXTURES / "font_programs"


def require_fixture(path: Path, reason: str | None = None) -> Path:
    """Return ``path`` or skip the calling test when the fixture is absent."""
    if not path.exists():
        pytest.skip(reason or f"fixture not present: {path.relative_to(TESTS_DIR)}")
    return path


def score_bench_pdf(name: str) -> Path:
    """A SCORE-Bench corpus document, skipping when the corpus is not checked out."""
    return require_fixture(SCORE_BENCH / name)

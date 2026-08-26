#!/usr/bin/env python
"""Regenerate only the first-page raster golden manifest and references."""

from __future__ import annotations

import pathlib
import sys

internal_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(internal_ROOT))

from scripts.raster_golden import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

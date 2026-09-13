"""Measure the differential extraction workload with a fresh document each time.

Run under cProfile or Scalene for attribution; use unprofiled repetitions for
wall-clock comparisons. Select the reference separately to isolate engine cost.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--engine", choices=("core", "reference"), default="core")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    module = importlib.import_module(
        "core_pdf.api.compat.pdfplumber" if args.engine == "core" else "pdfplumber"
    )
    for path in args.pdfs:
        durations: list[float] = []
        for _ in range(args.repeat):
            start = perf_counter()
            with module.open(path) as document:
                for page in document.pages:
                    page.extract_text()
                    page.extract_words()
            durations.append(perf_counter() - start)
        print(
            json.dumps(
                {
                    "file": str(path),
                    "engine": args.engine,
                    "seconds": durations,
                    "median_seconds": median(durations),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

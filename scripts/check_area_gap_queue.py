# SPDX-License-Identifier: AGPL-3.0-only

"""Check that core-pdf-cythonized's AreaGapQueue pops pairs as the Python queue does.

The pdfminer facade's reading order takes the C queue when every box coordinate is a float
and the Python _PairQueue otherwise, so the two must agree pair for pair. This drives both
through the same random runs of pops, re-queues and merges, over boxes that include NaN,
infinite and repeated coordinates and heaps on both sides of the 2,500 items at which
CPython's heapq changes its heapify, and fails on the first pop that differs.

uv run --all-packages python scripts/check_area_gap_queue.py [runs]
"""

from __future__ import annotations

import math
import random
import sys

from core_pdf.impl.base.spatial import AreaGapQueue
from core_pdf_compat.pdfminer._layout import _PairQueue

SPECIAL = (math.nan, math.inf, -math.inf, 0.0, -0.0, 1.0, 2.5)
# 71 boxes make 2,485 pairs and 75 make 2,775: either side of heapq's 2,500.
SIZES = (2, 3, 5, 9, 17, 40, 71, 75, 100)


def coordinate(rng: random.Random, special: bool) -> float:
    if special and rng.random() < 0.08:
        return rng.choice(SPECIAL)
    return float(rng.choice((rng.randint(0, 40), rng.random() * 600, round(rng.random() * 50, 1))))


def run(seed: int, count: int, special: bool) -> str | None:
    rng = random.Random(seed)
    boxes = []
    for _ in range(count):
        x0, y0 = coordinate(rng, special), coordinate(rng, special)
        boxes.append((x0, y0, x0 + coordinate(rng, special), y0 + coordinate(rng, special)))
    python, kernel = _PairQueue(boxes), AreaGapQueue(boxes)
    corners = dict(enumerate(boxes))
    active = count
    next_id = count
    step = 0
    while True:
        expected, actual = python.pop(), kernel.pop()
        if expected != actual:
            return f"seed {seed}, {count} boxes, pop {step}: Python {expected}, C {actual}"
        if expected is None or active == 2:
            return None
        step += 1
        skip_between, first, second = expected
        if not skip_between and rng.random() < 0.3:
            python.push_skipped()
            kernel.push_skipped()
            continue
        a, b = corners[first], corners[second]
        union = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
        corners[next_id] = union
        python.merge(first, second, next_id, union)
        kernel.merge(first, second, next_id, union)
        next_id += 1
        active -= 1


def main(argv: list[str]) -> int:
    runs = int(argv[0]) if argv else 40
    trials = 0
    failures = []
    for seed in range(runs):
        for count in SIZES:
            for special in (False, True):
                trials += 1
                failure = run(seed * 1000 + count * 2 + special, count, special)
                if failure is not None:
                    failures.append(failure)
    print(f"{trials} runs, {len(failures)} differ")
    for failure in failures[:5]:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

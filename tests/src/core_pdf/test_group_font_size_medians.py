# SPDX-License-Identifier: AGPL-3.0-only

import numpy

from core_pdf.impl.array_views import finite_median
from core_pdf.impl.extract_block_layout import group_font_size_medians


def test_group_medians_match_finite_median() -> None:
    rng = numpy.random.default_rng(7)
    sizes = rng.uniform(-2.0, 20.0, 400).astype(numpy.float32)
    sizes[rng.integers(0, 400, 40)] = numpy.nan
    sizes[rng.integers(0, 400, 10)] = numpy.inf
    sizes[100:110] = numpy.nan  # a group with no usable size
    starts = numpy.array([0, 3, 4, 100, 110, 250, 399], dtype=numpy.int64)
    stops = numpy.append(starts[1:], len(sizes))
    medians = group_font_size_medians(sizes, starts, stops, len(starts))
    for (start, stop), median in zip(zip(starts, stops, strict=True), medians, strict=True):
        group = sizes[start:stop]
        usable = group[numpy.isfinite(group) & (group > 0)]
        assert median == (finite_median(usable) if len(usable) else None)

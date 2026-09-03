# SPDX-License-Identifier: AGPL-3.0-only
"""Glyph advance lookup, the hottest small leaf on the text path.

``width_for`` is called once per glyph per show operation -- about 70k times
across twelve corpus pages -- so it sits behind every text-heavy benchmark in
this suite. Measuring it directly gives a regression here a signal of its own
instead of a percent or two buried in ``test_interpret_text_page_benchmark``.

The two implementations are separated because they are separate code: simple
fonts hit a dict on ``SparseFontWidthMap``, CID fonts index a tuple by offset
on ``CompactCIDWidthMap``, and only the second pays a bounds check.
"""

from __future__ import annotations

from collections.abc import Callable
from random import Random

from core_pdf.impl.spec.s_09_fonts.cmap_widths import (
    CompactCIDWidthMap,
    SparseFontWidthMap,
)

DEFAULT_WIDTH = 500.0
LOOKUPS = 20_000

# Latin letter frequencies, so the sampled codes cluster the way real text does
# and the dict lookup sees a representative hit distribution rather than a
# uniform sweep over the whole table.
LETTER_FREQUENCIES = "etaoinshrdlcumwfgypbvkjxqz"


def internal_simple_codes() -> tuple[int, ...]:
    """Character codes shaped like English prose, with spaces and punctuation."""
    random = Random(20260903)
    population = [
        *(
            ord(letter)
            for index, letter in enumerate(LETTER_FREQUENCIES)
            for _ in range(26 - index)
        ),
        *(ord(" ") for _ in range(120)),
        *(ord(letter.upper()) for letter in LETTER_FREQUENCIES[:8]),
        *(ord(character) for character in ".,;:'\"()-"),
    ]
    return tuple(random.choice(population) for _ in range(LOOKUPS))


def internal_cid_codes() -> tuple[int, ...]:
    """CIDs across a mapped range, including misses that fall back to default."""
    random = Random(20260903)
    return tuple(random.randrange(0, 1400) for _ in range(LOOKUPS))


def internal_sum_widths(
    width_for: Callable[[int, float], float], codes: tuple[int, ...], default: float
) -> float:
    total = 0.0
    for code in codes:
        total += width_for(code, default)
    return total


def test_simple_font_width_lookup_benchmark(benchmark) -> None:
    """The simple-font path: a dict lookup per glyph, mostly hitting."""
    widths = SparseFontWidthMap({code: 250.0 + (code % 400) for code in range(32, 127)})
    codes = internal_simple_codes()

    total = benchmark(internal_sum_widths, widths.width_for, codes, DEFAULT_WIDTH)

    assert total > 0.0


def test_cid_font_width_lookup_benchmark(benchmark) -> None:
    """The CID path: an offset index with a bounds check, and real misses."""
    widths = CompactCIDWidthMap(1, tuple(240.0 + (index % 500) for index in range(1024)))
    codes = internal_cid_codes()

    total = benchmark(internal_sum_widths, widths.width_for, codes, DEFAULT_WIDTH)

    assert total > 0.0

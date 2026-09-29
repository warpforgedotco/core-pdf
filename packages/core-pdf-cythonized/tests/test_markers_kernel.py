# SPDX-License-Identifier: AGPL-3.0-only

import random

import pytest

from core_pdf_cythonized import markers_present, regions_with_markers

PIECES = (b"#", b"XRef", b"/W", b"/Size", b"X", b"R", b"/", b"S", b"ze", b"ef", b"W", b"crypt")


def regions_by_find(data, needle, starts, data_len):
    found = set()
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else data_len
        if data.find(needle, start, end) >= 0:
            found.add(index)
    return found


def test_one_pass_matches_a_find_per_needle_and_region():
    rng = random.Random(3)
    for _ in range(5000):
        data = b"".join(rng.choice(PIECES) for _ in range(rng.randint(0, 200)))
        needles = tuple(
            rng.sample([b"#", b"XRef", b"/W", b"/Size", b"crypt", b"ef"], rng.randint(1, 4))
        )
        assert markers_present(data, needles) == tuple(data.find(n) >= 0 for n in needles)
        starts = sorted({rng.randint(0, max(len(data) - 1, 0)) for _ in range(rng.randint(0, 20))})
        if not data:
            starts = []
        got = regions_with_markers(data, needles, starts, len(data))
        assert got == [regions_by_find(data, n, starts, len(data)) for n in needles]


def test_empty_needles_are_refused():
    with pytest.raises(ValueError):
        markers_present(b"abc", (b"",))
    with pytest.raises(ValueError):
        regions_with_markers(b"abc", (b"",), [0], 3)


def test_negative_starts_are_refused():
    with pytest.raises(ValueError, match="negative"):
        regions_with_markers(b"XRef", (b"XRef",), [-1], 4)

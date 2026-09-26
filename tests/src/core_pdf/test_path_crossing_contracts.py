import pytest

from core_pdf.impl.render import paths


@pytest.mark.parametrize(
    ("crossings", "nonzero", "evenodd"),
    [
        ([], [], []),
        ([(0, 1), (10, -1)], list(range(10)), list(range(10))),
        ([(5, 1), (5, -1)], [], []),
        ([(0, 1), (2, -1), (8, 1), (10, -1)], [0, 1, 8, 9], [0, 1, 8, 9]),
        ([(0, 1), (2, 1), (8, -1), (10, -1)], list(range(10)), [0, 1, 8, 9]),
        ([(0, 1), (5, -1), (5, 1), (10, -1)], list(range(10)), list(range(10))),
        ([(0, 1), (0, -1), (10, 1), (10, -1)], [], []),
    ],
)
@pytest.mark.parametrize("rule", ["nonzero", "evenodd"])
@pytest.mark.parametrize("reverse", [False, True])
def test_crossing_spans_distinguish_holes_overlaps_and_shared_boundaries(
    crossings, nonzero, evenodd, rule, reverse
):
    values = list(reversed(crossings)) if reverse else list(crossings)
    spans = paths.fill_path_crossing_spans(values, rule)
    occupied = [x for x in range(10) if any(start <= x + 0.5 < end for start, end in spans)]
    assert occupied == (nonzero if rule == "nonzero" else evenodd)
    assert all(start < end for start, end in spans)

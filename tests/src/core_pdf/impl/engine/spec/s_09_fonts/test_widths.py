from typing import Any

from core_pdf.impl.engine.spec.s_09_fonts.cmap_widths import (
    CompactCIDWidthMap,
    parse_cid_widths,
)
from core_pdf.impl.engine.spec.s_09_fonts.widths import parse_font_widths


def test_cid_width_range_is_clipped_to_valid_cid_domain() -> None:
    widths = parse_cid_widths([-(10**12), 10**12, 321])

    assert len(widths) == 65536
    assert widths[0] == 321.0
    assert widths[65535] == 321.0
    assert widths.get(-1) is None
    assert widths.get(65536) is None


def test_compact_cid_width_array_keeps_only_overlapping_codes() -> None:
    widths = parse_cid_widths([-2, [100, 200, 300, 400]])

    assert isinstance(widths, CompactCIDWidthMap)
    assert widths.start == 0
    assert tuple(widths.values()) == (300.0, 400.0)


def test_cid_width_array_ignores_codes_outside_valid_domain() -> None:
    widths = parse_cid_widths([65535, [500, "ignored", 700]])

    assert dict(widths.items()) == {65535: 500.0}


def test_vertical_width_range_is_clipped_to_valid_cid_domain() -> None:
    font: dict[str, Any] = {
        "Subtype": "Type0",
        "DescendantFonts": [
            {
                "Subtype": "CIDFontType0",
                "W2": [-(10**12), 10**12, -700, 25, 800],
            }
        ],
    }

    metrics = parse_font_widths(font, "Type0")

    assert len(metrics.vertical_metrics) == 65536
    assert metrics.vertical_metrics[0] == (-700.0, 25.0, 800.0)
    assert metrics.vertical_metrics[65535] == (-700.0, 25.0, 800.0)
    assert -1 not in metrics.vertical_metrics
    assert 65536 not in metrics.vertical_metrics


def test_vertical_width_array_keeps_only_valid_cids_without_shifting_metrics() -> None:
    font: dict[str, Any] = {
        "Subtype": "Type0",
        "DescendantFonts": [
            {
                "Subtype": "CIDFontType0",
                "W2": [
                    -1,
                    [-500, 10, 20, -600, 30, 40],
                    65535,
                    [-700, 50, 60, -800, 70, 80],
                ],
            }
        ],
    }

    metrics = parse_font_widths(font, "Type0")

    assert metrics.vertical_metrics == {
        0: (-600.0, 30.0, 40.0),
        65535: (-700.0, 50.0, 60.0),
    }

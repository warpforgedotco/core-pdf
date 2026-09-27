from core_pdf.impl.extract_block_order import (
    REORDER_PASSES,
    column_major_prose,
    interleave_columnar_blocks,
    transpose_numeric_table_blocks,
)
from core_pdf.impl.extract_layout_rules import LAYOUT_RULES, LayoutRules, PageRegionRules


def test_reorder_passes_run_in_their_established_order() -> None:
    assert (
        interleave_columnar_blocks,
        transpose_numeric_table_blocks,
        column_major_prose,
    ) == REORDER_PASSES


def test_layout_rules_group_the_reading_order_thresholds() -> None:
    assert LayoutRules() == LAYOUT_RULES
    assert LAYOUT_RULES.reading_order.full_width_ratio == 0.70
    assert LAYOUT_RULES.xy_cut.horizontal_gap_ratio == 0.90
    assert LAYOUT_RULES.xy_cut.horizontal_preference == 1.15
    assert LAYOUT_RULES.classification.heading_size_ratio == 1.2
    regions = LAYOUT_RULES.page_regions
    assert (regions.header_top_ratio, regions.footer_bottom_ratio) == (0.88, 0.12)
    assert (regions.caption_min_overlap, regions.caption_gap, regions.caption_height_ratio) == (
        0.3,
        24.0,
        2.5,
    )
    assert LayoutRules(page_regions=PageRegionRules(caption_gap=12.0)) != LAYOUT_RULES

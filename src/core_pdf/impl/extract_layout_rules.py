# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core_pdf.impl.types import GeneratedRecord, frozen_setattr


class ReadingOrderRules(GeneratedRecord):
    full_width_ratio: float = 0.70
    column_band_overlap: float = 0.50
    stacking_tolerance: float = 2.0
    side_by_side_overlap: float = 4.0
    stacked_overlap_ratio: float = 0.45
    repeated_columns_min_blocks: int = 6
    repeated_columns_top_ratio: float = 0.55
    repeated_columns_min_top_blocks: int = 6
    repeated_columns_gap: float = 16.0
    repeated_columns_min_members: int = 3
    repeated_columns_min_columns: int = 3
    interleave_min_lines: int = 20
    interleave_min_blocks: int = 3
    interleave_left_spread: float = 20.0
    interleave_right_spread: float = 30.0
    transpose_min_lines: int = 300
    transpose_column_gap: float = 8.0
    transpose_min_columns: int = 20
    transpose_min_digit_ratio: float = 0.25
    column_major_min_lines: int = 80
    column_major_column_gap: float = 40.0
    column_major_min_columns: int = 3
    column_major_min_alpha_ratio: float = 0.45
    column_major_min_transition_ratio: float = 0.25


class XyCutRules(GeneratedRecord):
    max_depth: int = 32
    row_band_floor: float = 1.0
    row_band_ratio: float = 0.5
    horizontal_gap_floor: float = 3.0
    horizontal_gap_ratio: float = 0.90
    horizontal_preference: float = 1.15
    column_gap_floor: float = 12.0
    column_gap_width_ratio: float = 0.05
    narrow_gap_floor: float = 6.0
    narrow_gap_cap: float = 12.0
    narrow_gap_width_ratio: float = 0.08
    narrow_gap_height_ratio: float = 0.9
    column_alignment_min_boxes: int = 4
    column_alignment_tolerance: float = 3.0
    column_alignment_ratio: float = 0.6
    column_width_balance: float = 0.5
    gutter_min_boxes: int = 8
    gutter_samples: int = 256
    gutter_crossing_divisor: int = 20
    peel_min_boxes: int = 4
    peel_band_floor: float = 1.0
    peel_band_ratio: float = 0.5
    peel_min_column_boxes: int = 18


class ClassificationRules(GeneratedRecord):
    heading_size_ratio: float = 1.2
    heading_max_lines: int = 3
    heading_max_characters: int = 240
    heading_max_level: int = 3


class PageRegionRules(GeneratedRecord):
    header_top_ratio: float = 0.88
    footer_bottom_ratio: float = 0.12
    band_height_ratio: float = 0.08
    band_max_characters: int = 240
    caption_min_overlap: float = 0.3
    caption_gap: float = 24.0
    caption_height_ratio: float = 2.5


class LayoutRules(GeneratedRecord):
    reading_order: ReadingOrderRules
    xy_cut: XyCutRules
    classification: ClassificationRules
    page_regions: PageRegionRules

    def __init__(
        self,
        reading_order: ReadingOrderRules | None = None,
        xy_cut: XyCutRules | None = None,
        classification: ClassificationRules | None = None,
        page_regions: PageRegionRules | None = None,
    ) -> None:
        frozen_setattr(
            self, "reading_order", ReadingOrderRules() if reading_order is None else reading_order
        )
        frozen_setattr(self, "xy_cut", XyCutRules() if xy_cut is None else xy_cut)
        frozen_setattr(
            self,
            "classification",
            ClassificationRules() if classification is None else classification,
        )
        frozen_setattr(
            self, "page_regions", PageRegionRules() if page_regions is None else page_regions
        )


LAYOUT_RULES = LayoutRules()

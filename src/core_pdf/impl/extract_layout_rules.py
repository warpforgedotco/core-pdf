# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import ClassVar

from core_pdf.impl.types import Record, frozen_setattr


class ReadingOrderRules(Record):
    __slots__ = (
        "full_width_ratio",
        "column_band_overlap",
        "stacking_tolerance",
        "side_by_side_overlap",
        "stacked_overlap_ratio",
        "repeated_columns_min_blocks",
        "repeated_columns_top_ratio",
        "repeated_columns_min_top_blocks",
        "repeated_columns_gap",
        "repeated_columns_min_members",
        "repeated_columns_min_columns",
        "interleave_min_lines",
        "interleave_min_blocks",
        "interleave_left_spread",
        "interleave_right_spread",
        "transpose_min_lines",
        "transpose_column_gap",
        "transpose_min_columns",
        "transpose_min_digit_ratio",
        "column_major_min_lines",
        "column_major_column_gap",
        "column_major_min_columns",
        "column_major_min_alpha_ratio",
        "column_major_min_transition_ratio",
    )

    full_width_ratio: float
    column_band_overlap: float
    stacking_tolerance: float
    side_by_side_overlap: float
    stacked_overlap_ratio: float
    repeated_columns_min_blocks: int
    repeated_columns_top_ratio: float
    repeated_columns_min_top_blocks: int
    repeated_columns_gap: float
    repeated_columns_min_members: int
    repeated_columns_min_columns: int
    interleave_min_lines: int
    interleave_min_blocks: int
    interleave_left_spread: float
    interleave_right_spread: float
    transpose_min_lines: int
    transpose_column_gap: float
    transpose_min_columns: int
    transpose_min_digit_ratio: float
    column_major_min_lines: int
    column_major_column_gap: float
    column_major_min_columns: int
    column_major_min_alpha_ratio: float
    column_major_min_transition_ratio: float

    __fields__: ClassVar[tuple[str, ...]] = (
        "full_width_ratio",
        "column_band_overlap",
        "stacking_tolerance",
        "side_by_side_overlap",
        "stacked_overlap_ratio",
        "repeated_columns_min_blocks",
        "repeated_columns_top_ratio",
        "repeated_columns_min_top_blocks",
        "repeated_columns_gap",
        "repeated_columns_min_members",
        "repeated_columns_min_columns",
        "interleave_min_lines",
        "interleave_min_blocks",
        "interleave_left_spread",
        "interleave_right_spread",
        "transpose_min_lines",
        "transpose_column_gap",
        "transpose_min_columns",
        "transpose_min_digit_ratio",
        "column_major_min_lines",
        "column_major_column_gap",
        "column_major_min_columns",
        "column_major_min_alpha_ratio",
        "column_major_min_transition_ratio",
    )
    __match_args__ = (
        "full_width_ratio",
        "column_band_overlap",
        "stacking_tolerance",
        "side_by_side_overlap",
        "stacked_overlap_ratio",
        "repeated_columns_min_blocks",
        "repeated_columns_top_ratio",
        "repeated_columns_min_top_blocks",
        "repeated_columns_gap",
        "repeated_columns_min_members",
        "repeated_columns_min_columns",
        "interleave_min_lines",
        "interleave_min_blocks",
        "interleave_left_spread",
        "interleave_right_spread",
        "transpose_min_lines",
        "transpose_column_gap",
        "transpose_min_columns",
        "transpose_min_digit_ratio",
        "column_major_min_lines",
        "column_major_column_gap",
        "column_major_min_columns",
        "column_major_min_alpha_ratio",
        "column_major_min_transition_ratio",
    )

    def __init__(
        self,
        full_width_ratio: float = 0.70,
        column_band_overlap: float = 0.50,
        stacking_tolerance: float = 2.0,
        side_by_side_overlap: float = 4.0,
        stacked_overlap_ratio: float = 0.45,
        repeated_columns_min_blocks: int = 6,
        repeated_columns_top_ratio: float = 0.55,
        repeated_columns_min_top_blocks: int = 6,
        repeated_columns_gap: float = 16.0,
        repeated_columns_min_members: int = 3,
        repeated_columns_min_columns: int = 3,
        interleave_min_lines: int = 20,
        interleave_min_blocks: int = 3,
        interleave_left_spread: float = 20.0,
        interleave_right_spread: float = 30.0,
        transpose_min_lines: int = 300,
        transpose_column_gap: float = 8.0,
        transpose_min_columns: int = 20,
        transpose_min_digit_ratio: float = 0.25,
        column_major_min_lines: int = 80,
        column_major_column_gap: float = 40.0,
        column_major_min_columns: int = 3,
        column_major_min_alpha_ratio: float = 0.45,
        column_major_min_transition_ratio: float = 0.25,
    ) -> None:
        frozen_setattr(self, "full_width_ratio", full_width_ratio)
        frozen_setattr(self, "column_band_overlap", column_band_overlap)
        frozen_setattr(self, "stacking_tolerance", stacking_tolerance)
        frozen_setattr(self, "side_by_side_overlap", side_by_side_overlap)
        frozen_setattr(self, "stacked_overlap_ratio", stacked_overlap_ratio)
        frozen_setattr(self, "repeated_columns_min_blocks", repeated_columns_min_blocks)
        frozen_setattr(self, "repeated_columns_top_ratio", repeated_columns_top_ratio)
        frozen_setattr(self, "repeated_columns_min_top_blocks", repeated_columns_min_top_blocks)
        frozen_setattr(self, "repeated_columns_gap", repeated_columns_gap)
        frozen_setattr(self, "repeated_columns_min_members", repeated_columns_min_members)
        frozen_setattr(self, "repeated_columns_min_columns", repeated_columns_min_columns)
        frozen_setattr(self, "interleave_min_lines", interleave_min_lines)
        frozen_setattr(self, "interleave_min_blocks", interleave_min_blocks)
        frozen_setattr(self, "interleave_left_spread", interleave_left_spread)
        frozen_setattr(self, "interleave_right_spread", interleave_right_spread)
        frozen_setattr(self, "transpose_min_lines", transpose_min_lines)
        frozen_setattr(self, "transpose_column_gap", transpose_column_gap)
        frozen_setattr(self, "transpose_min_columns", transpose_min_columns)
        frozen_setattr(self, "transpose_min_digit_ratio", transpose_min_digit_ratio)
        frozen_setattr(self, "column_major_min_lines", column_major_min_lines)
        frozen_setattr(self, "column_major_column_gap", column_major_column_gap)
        frozen_setattr(self, "column_major_min_columns", column_major_min_columns)
        frozen_setattr(self, "column_major_min_alpha_ratio", column_major_min_alpha_ratio)
        frozen_setattr(self, "column_major_min_transition_ratio", column_major_min_transition_ratio)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.full_width_ratio == other.full_width_ratio
            and self.column_band_overlap == other.column_band_overlap
            and self.stacking_tolerance == other.stacking_tolerance
            and self.side_by_side_overlap == other.side_by_side_overlap
            and self.stacked_overlap_ratio == other.stacked_overlap_ratio
            and self.repeated_columns_min_blocks == other.repeated_columns_min_blocks
            and self.repeated_columns_top_ratio == other.repeated_columns_top_ratio
            and self.repeated_columns_min_top_blocks == other.repeated_columns_min_top_blocks
            and self.repeated_columns_gap == other.repeated_columns_gap
            and self.repeated_columns_min_members == other.repeated_columns_min_members
            and self.repeated_columns_min_columns == other.repeated_columns_min_columns
            and self.interleave_min_lines == other.interleave_min_lines
            and self.interleave_min_blocks == other.interleave_min_blocks
            and self.interleave_left_spread == other.interleave_left_spread
            and self.interleave_right_spread == other.interleave_right_spread
            and self.transpose_min_lines == other.transpose_min_lines
            and self.transpose_column_gap == other.transpose_column_gap
            and self.transpose_min_columns == other.transpose_min_columns
            and self.transpose_min_digit_ratio == other.transpose_min_digit_ratio
            and self.column_major_min_lines == other.column_major_min_lines
            and self.column_major_column_gap == other.column_major_column_gap
            and self.column_major_min_columns == other.column_major_min_columns
            and self.column_major_min_alpha_ratio == other.column_major_min_alpha_ratio
            and self.column_major_min_transition_ratio == other.column_major_min_transition_ratio
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.full_width_ratio,
                self.column_band_overlap,
                self.stacking_tolerance,
                self.side_by_side_overlap,
                self.stacked_overlap_ratio,
                self.repeated_columns_min_blocks,
                self.repeated_columns_top_ratio,
                self.repeated_columns_min_top_blocks,
                self.repeated_columns_gap,
                self.repeated_columns_min_members,
                self.repeated_columns_min_columns,
                self.interleave_min_lines,
                self.interleave_min_blocks,
                self.interleave_left_spread,
                self.interleave_right_spread,
                self.transpose_min_lines,
                self.transpose_column_gap,
                self.transpose_min_columns,
                self.transpose_min_digit_ratio,
                self.column_major_min_lines,
                self.column_major_column_gap,
                self.column_major_min_columns,
                self.column_major_min_alpha_ratio,
                self.column_major_min_transition_ratio,
            )
        )


class XyCutRules(Record):
    __slots__ = (
        "max_depth",
        "row_band_floor",
        "row_band_ratio",
        "horizontal_gap_floor",
        "horizontal_gap_ratio",
        "horizontal_preference",
        "column_gap_floor",
        "column_gap_width_ratio",
        "narrow_gap_floor",
        "narrow_gap_cap",
        "narrow_gap_width_ratio",
        "narrow_gap_height_ratio",
        "column_alignment_min_boxes",
        "column_alignment_tolerance",
        "column_alignment_ratio",
        "column_width_balance",
        "gutter_min_boxes",
        "gutter_samples",
        "gutter_crossing_divisor",
        "peel_min_boxes",
        "peel_band_floor",
        "peel_band_ratio",
        "peel_min_column_boxes",
    )

    max_depth: int
    row_band_floor: float
    row_band_ratio: float
    horizontal_gap_floor: float
    horizontal_gap_ratio: float
    horizontal_preference: float
    column_gap_floor: float
    column_gap_width_ratio: float
    narrow_gap_floor: float
    narrow_gap_cap: float
    narrow_gap_width_ratio: float
    narrow_gap_height_ratio: float
    column_alignment_min_boxes: int
    column_alignment_tolerance: float
    column_alignment_ratio: float
    column_width_balance: float
    gutter_min_boxes: int
    gutter_samples: int
    gutter_crossing_divisor: int
    peel_min_boxes: int
    peel_band_floor: float
    peel_band_ratio: float
    peel_min_column_boxes: int

    __fields__: ClassVar[tuple[str, ...]] = (
        "max_depth",
        "row_band_floor",
        "row_band_ratio",
        "horizontal_gap_floor",
        "horizontal_gap_ratio",
        "horizontal_preference",
        "column_gap_floor",
        "column_gap_width_ratio",
        "narrow_gap_floor",
        "narrow_gap_cap",
        "narrow_gap_width_ratio",
        "narrow_gap_height_ratio",
        "column_alignment_min_boxes",
        "column_alignment_tolerance",
        "column_alignment_ratio",
        "column_width_balance",
        "gutter_min_boxes",
        "gutter_samples",
        "gutter_crossing_divisor",
        "peel_min_boxes",
        "peel_band_floor",
        "peel_band_ratio",
        "peel_min_column_boxes",
    )
    __match_args__ = (
        "max_depth",
        "row_band_floor",
        "row_band_ratio",
        "horizontal_gap_floor",
        "horizontal_gap_ratio",
        "horizontal_preference",
        "column_gap_floor",
        "column_gap_width_ratio",
        "narrow_gap_floor",
        "narrow_gap_cap",
        "narrow_gap_width_ratio",
        "narrow_gap_height_ratio",
        "column_alignment_min_boxes",
        "column_alignment_tolerance",
        "column_alignment_ratio",
        "column_width_balance",
        "gutter_min_boxes",
        "gutter_samples",
        "gutter_crossing_divisor",
        "peel_min_boxes",
        "peel_band_floor",
        "peel_band_ratio",
        "peel_min_column_boxes",
    )

    def __init__(
        self,
        max_depth: int = 32,
        row_band_floor: float = 1.0,
        row_band_ratio: float = 0.5,
        horizontal_gap_floor: float = 3.0,
        horizontal_gap_ratio: float = 0.90,
        horizontal_preference: float = 1.15,
        column_gap_floor: float = 12.0,
        column_gap_width_ratio: float = 0.05,
        narrow_gap_floor: float = 6.0,
        narrow_gap_cap: float = 12.0,
        narrow_gap_width_ratio: float = 0.08,
        narrow_gap_height_ratio: float = 0.9,
        column_alignment_min_boxes: int = 4,
        column_alignment_tolerance: float = 3.0,
        column_alignment_ratio: float = 0.6,
        column_width_balance: float = 0.5,
        gutter_min_boxes: int = 8,
        gutter_samples: int = 256,
        gutter_crossing_divisor: int = 20,
        peel_min_boxes: int = 4,
        peel_band_floor: float = 1.0,
        peel_band_ratio: float = 0.5,
        peel_min_column_boxes: int = 18,
    ) -> None:
        frozen_setattr(self, "max_depth", max_depth)
        frozen_setattr(self, "row_band_floor", row_band_floor)
        frozen_setattr(self, "row_band_ratio", row_band_ratio)
        frozen_setattr(self, "horizontal_gap_floor", horizontal_gap_floor)
        frozen_setattr(self, "horizontal_gap_ratio", horizontal_gap_ratio)
        frozen_setattr(self, "horizontal_preference", horizontal_preference)
        frozen_setattr(self, "column_gap_floor", column_gap_floor)
        frozen_setattr(self, "column_gap_width_ratio", column_gap_width_ratio)
        frozen_setattr(self, "narrow_gap_floor", narrow_gap_floor)
        frozen_setattr(self, "narrow_gap_cap", narrow_gap_cap)
        frozen_setattr(self, "narrow_gap_width_ratio", narrow_gap_width_ratio)
        frozen_setattr(self, "narrow_gap_height_ratio", narrow_gap_height_ratio)
        frozen_setattr(self, "column_alignment_min_boxes", column_alignment_min_boxes)
        frozen_setattr(self, "column_alignment_tolerance", column_alignment_tolerance)
        frozen_setattr(self, "column_alignment_ratio", column_alignment_ratio)
        frozen_setattr(self, "column_width_balance", column_width_balance)
        frozen_setattr(self, "gutter_min_boxes", gutter_min_boxes)
        frozen_setattr(self, "gutter_samples", gutter_samples)
        frozen_setattr(self, "gutter_crossing_divisor", gutter_crossing_divisor)
        frozen_setattr(self, "peel_min_boxes", peel_min_boxes)
        frozen_setattr(self, "peel_band_floor", peel_band_floor)
        frozen_setattr(self, "peel_band_ratio", peel_band_ratio)
        frozen_setattr(self, "peel_min_column_boxes", peel_min_column_boxes)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.max_depth == other.max_depth
            and self.row_band_floor == other.row_band_floor
            and self.row_band_ratio == other.row_band_ratio
            and self.horizontal_gap_floor == other.horizontal_gap_floor
            and self.horizontal_gap_ratio == other.horizontal_gap_ratio
            and self.horizontal_preference == other.horizontal_preference
            and self.column_gap_floor == other.column_gap_floor
            and self.column_gap_width_ratio == other.column_gap_width_ratio
            and self.narrow_gap_floor == other.narrow_gap_floor
            and self.narrow_gap_cap == other.narrow_gap_cap
            and self.narrow_gap_width_ratio == other.narrow_gap_width_ratio
            and self.narrow_gap_height_ratio == other.narrow_gap_height_ratio
            and self.column_alignment_min_boxes == other.column_alignment_min_boxes
            and self.column_alignment_tolerance == other.column_alignment_tolerance
            and self.column_alignment_ratio == other.column_alignment_ratio
            and self.column_width_balance == other.column_width_balance
            and self.gutter_min_boxes == other.gutter_min_boxes
            and self.gutter_samples == other.gutter_samples
            and self.gutter_crossing_divisor == other.gutter_crossing_divisor
            and self.peel_min_boxes == other.peel_min_boxes
            and self.peel_band_floor == other.peel_band_floor
            and self.peel_band_ratio == other.peel_band_ratio
            and self.peel_min_column_boxes == other.peel_min_column_boxes
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.max_depth,
                self.row_band_floor,
                self.row_band_ratio,
                self.horizontal_gap_floor,
                self.horizontal_gap_ratio,
                self.horizontal_preference,
                self.column_gap_floor,
                self.column_gap_width_ratio,
                self.narrow_gap_floor,
                self.narrow_gap_cap,
                self.narrow_gap_width_ratio,
                self.narrow_gap_height_ratio,
                self.column_alignment_min_boxes,
                self.column_alignment_tolerance,
                self.column_alignment_ratio,
                self.column_width_balance,
                self.gutter_min_boxes,
                self.gutter_samples,
                self.gutter_crossing_divisor,
                self.peel_min_boxes,
                self.peel_band_floor,
                self.peel_band_ratio,
                self.peel_min_column_boxes,
            )
        )


class ClassificationRules(Record):
    __slots__ = (
        "heading_size_ratio",
        "heading_max_lines",
        "heading_max_characters",
        "heading_max_level",
    )

    heading_size_ratio: float
    heading_max_lines: int
    heading_max_characters: int
    heading_max_level: int

    __fields__: ClassVar[tuple[str, ...]] = (
        "heading_size_ratio",
        "heading_max_lines",
        "heading_max_characters",
        "heading_max_level",
    )
    __match_args__ = (
        "heading_size_ratio",
        "heading_max_lines",
        "heading_max_characters",
        "heading_max_level",
    )

    def __init__(
        self,
        heading_size_ratio: float = 1.2,
        heading_max_lines: int = 3,
        heading_max_characters: int = 240,
        heading_max_level: int = 3,
    ) -> None:
        frozen_setattr(self, "heading_size_ratio", heading_size_ratio)
        frozen_setattr(self, "heading_max_lines", heading_max_lines)
        frozen_setattr(self, "heading_max_characters", heading_max_characters)
        frozen_setattr(self, "heading_max_level", heading_max_level)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.heading_size_ratio == other.heading_size_ratio
            and self.heading_max_lines == other.heading_max_lines
            and self.heading_max_characters == other.heading_max_characters
            and self.heading_max_level == other.heading_max_level
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.heading_size_ratio,
                self.heading_max_lines,
                self.heading_max_characters,
                self.heading_max_level,
            )
        )


class PageRegionRules(Record):
    __slots__ = (
        "header_top_ratio",
        "footer_bottom_ratio",
        "band_height_ratio",
        "band_max_characters",
        "caption_min_overlap",
        "caption_gap",
        "caption_height_ratio",
    )

    header_top_ratio: float
    footer_bottom_ratio: float
    band_height_ratio: float
    band_max_characters: int
    caption_min_overlap: float
    caption_gap: float
    caption_height_ratio: float

    __fields__: ClassVar[tuple[str, ...]] = (
        "header_top_ratio",
        "footer_bottom_ratio",
        "band_height_ratio",
        "band_max_characters",
        "caption_min_overlap",
        "caption_gap",
        "caption_height_ratio",
    )
    __match_args__ = (
        "header_top_ratio",
        "footer_bottom_ratio",
        "band_height_ratio",
        "band_max_characters",
        "caption_min_overlap",
        "caption_gap",
        "caption_height_ratio",
    )

    def __init__(
        self,
        header_top_ratio: float = 0.88,
        footer_bottom_ratio: float = 0.12,
        band_height_ratio: float = 0.08,
        band_max_characters: int = 240,
        caption_min_overlap: float = 0.3,
        caption_gap: float = 24.0,
        caption_height_ratio: float = 2.5,
    ) -> None:
        frozen_setattr(self, "header_top_ratio", header_top_ratio)
        frozen_setattr(self, "footer_bottom_ratio", footer_bottom_ratio)
        frozen_setattr(self, "band_height_ratio", band_height_ratio)
        frozen_setattr(self, "band_max_characters", band_max_characters)
        frozen_setattr(self, "caption_min_overlap", caption_min_overlap)
        frozen_setattr(self, "caption_gap", caption_gap)
        frozen_setattr(self, "caption_height_ratio", caption_height_ratio)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.header_top_ratio == other.header_top_ratio
            and self.footer_bottom_ratio == other.footer_bottom_ratio
            and self.band_height_ratio == other.band_height_ratio
            and self.band_max_characters == other.band_max_characters
            and self.caption_min_overlap == other.caption_min_overlap
            and self.caption_gap == other.caption_gap
            and self.caption_height_ratio == other.caption_height_ratio
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.header_top_ratio,
                self.footer_bottom_ratio,
                self.band_height_ratio,
                self.band_max_characters,
                self.caption_min_overlap,
                self.caption_gap,
                self.caption_height_ratio,
            )
        )


class LayoutRules(Record):
    __slots__ = (
        "reading_order",
        "xy_cut",
        "classification",
        "page_regions",
    )

    reading_order: ReadingOrderRules
    xy_cut: XyCutRules
    classification: ClassificationRules
    page_regions: PageRegionRules

    __fields__: ClassVar[tuple[str, ...]] = (
        "reading_order",
        "xy_cut",
        "classification",
        "page_regions",
    )
    __match_args__ = (
        "reading_order",
        "xy_cut",
        "classification",
        "page_regions",
    )

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

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.reading_order == other.reading_order
            and self.xy_cut == other.xy_cut
            and self.classification == other.classification
            and self.page_regions == other.page_regions
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.reading_order,
                self.xy_cut,
                self.classification,
                self.page_regions,
            )
        )


LAYOUT_RULES = LayoutRules()

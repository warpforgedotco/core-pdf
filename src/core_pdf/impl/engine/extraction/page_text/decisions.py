# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass

from core_pdf.impl.engine.extraction.common import page_profile


@dataclass(frozen=True)
class PageExtractionDecision:
    route: str
    reason: str
    recommended_strategy: str
    skip_text: bool
    ocr_enabled: bool


def page_extraction_decision(
    profile: page_profile.PageProfile,
    *,
    ocr_enabled: bool,
) -> PageExtractionDecision:
    if profile.can_skip_all_text:
        return PageExtractionDecision(
            route="profile_skip",
            reason="no_text_or_drawn_content",
            recommended_strategy=profile.recommended_strategy,
            skip_text=True,
            ocr_enabled=ocr_enabled,
        )
    if profile.can_skip_native_text and not ocr_enabled:
        return PageExtractionDecision(
            route="profile_skip",
            reason="no_native_text_and_ocr_disabled",
            recommended_strategy=profile.recommended_strategy,
            skip_text=True,
            ocr_enabled=ocr_enabled,
        )
    if profile.recommended_strategy == "native_text":
        return PageExtractionDecision(
            route="native_text_fast",
            reason="profile_native_text",
            recommended_strategy=profile.recommended_strategy,
            skip_text=False,
            ocr_enabled=ocr_enabled,
        )
    if profile.recommended_strategy == "text_table":
        return PageExtractionDecision(
            route="text_table_layout",
            reason="profile_text_table",
            recommended_strategy=profile.recommended_strategy,
            skip_text=False,
            ocr_enabled=ocr_enabled,
        )
    return PageExtractionDecision(
        route="generic",
        reason=f"profile_{profile.recommended_strategy}",
        recommended_strategy=profile.recommended_strategy,
        skip_text=False,
        ocr_enabled=ocr_enabled,
    )


__all__ = (
    "PageExtractionDecision",
    "page_extraction_decision",
)

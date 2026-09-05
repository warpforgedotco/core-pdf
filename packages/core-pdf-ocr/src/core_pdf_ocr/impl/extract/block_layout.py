# SPDX-License-Identifier: AGPL-3.0-only
"""Apply recognition source labels and geometric order to shared block layout."""

from __future__ import annotations

import unicodedata

import numpy

from core_pdf.impl.extract import block_layout as native_layout
from core_pdf.impl.extract.contracts import (
    ObservationBatch,
    ParsedBlock,
    ReadingOrderEvidence,
)
from core_pdf.impl.records import TextWord
from core_pdf_ocr.impl.extract.contracts import ObservationSource

internal_SOURCE_LABELS = {
    int(ObservationSource.NATIVE): "native",
    int(ObservationSource.OCR): "ocr",
}
internal_OCR_SOURCE = numpy.uint8(ObservationSource.OCR)


def internal_group_order(observations: ObservationBatch, indexes: numpy.ndarray) -> numpy.ndarray:
    if not bool((observations.source[indexes] == internal_OCR_SOURCE).any()):
        return indexes
    rotation = int(observations.rotation[indexes[0]]) % 360
    boxes = observations.bbox[indexes]
    if rotation == 90:
        positions = (boxes[:, 1] + boxes[:, 3]) * 0.5
    elif rotation == 180:
        positions = -(boxes[:, 0] + boxes[:, 2]) * 0.5
    elif rotation == 270:
        positions = -(boxes[:, 1] + boxes[:, 3]) * 0.5
    else:
        positions = (boxes[:, 0] + boxes[:, 2]) * 0.5
    # One pass over the text counts both directions; ASCII text can only
    # contribute L characters, and only letters carry a strong class.
    rtl = 0
    ltr = 0
    bidirectional = unicodedata.bidirectional
    for index in indexes:
        observation_text = observations.text[index]
        if observation_text.isascii():
            ltr += sum(map(str.isalpha, observation_text))
            continue
        for character in observation_text:
            direction_class = bidirectional(character)
            if direction_class == "L":
                ltr += 1
            elif direction_class in {"R", "AL", "AN"}:
                rtl += 1
    order = numpy.argsort(-positions if rtl > ltr else positions, kind="stable")
    return indexes[order]


def internal_group_text_and_words(
    observations: ObservationBatch,
    indexes: numpy.ndarray,
    *,
    may_contain_ocr: bool = True,
) -> tuple[str, tuple[TextWord, ...]]:
    if may_contain_ocr:
        indexes = internal_group_order(observations, indexes)
    return native_layout.internal_group_text_and_words(observations, indexes)


def internal_build_lines(observations: ObservationBatch) -> native_layout.internal_BuiltLines:
    return native_layout.internal_build_lines(
        observations, source_labels=internal_SOURCE_LABELS, group_order=internal_group_order
    )


def layout_blocks(
    observations: ObservationBatch,
    *,
    obstacles: tuple[tuple[float, float, float, float], ...] = (),
    use_xy_cut: bool = True,
    rotation: int = 0,
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> tuple[ParsedBlock, ...]:
    return native_layout.layout_blocks(
        observations,
        obstacles=obstacles,
        use_xy_cut=use_xy_cut,
        rotation=rotation,
        page_width=page_width,
        page_height=page_height,
        source_labels=internal_SOURCE_LABELS,
        group_order=internal_group_order,
    )


def layout_blocks_with_evidence(
    observations: ObservationBatch,
    *,
    obstacles: tuple[tuple[float, float, float, float], ...] = (),
    use_xy_cut: bool = True,
    rotation: int = 0,
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> tuple[tuple[ParsedBlock, ...], ReadingOrderEvidence]:
    return native_layout.layout_blocks_with_evidence(
        observations,
        obstacles=obstacles,
        use_xy_cut=use_xy_cut,
        rotation=rotation,
        page_width=page_width,
        page_height=page_height,
        source_labels=internal_SOURCE_LABELS,
        group_order=internal_group_order,
    )

# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import unicodedata

import numpy

from core_pdf.impl.extract_block_layout import LayoutHooks
from core_pdf.impl.extract_contracts import ObservationBatch, ObservationSource, PageFrame

SOURCE_LABELS = {
    int(ObservationSource.NATIVE): "native",
    int(ObservationSource.OCR): "ocr",
}
OCR_SOURCE = numpy.uint8(ObservationSource.OCR)


def group_order(observations: ObservationBatch, indexes: numpy.ndarray) -> numpy.ndarray:
    if not bool((observations.source[indexes] == OCR_SOURCE).any()):
        return indexes
    positions = PageFrame.reading_axis_positions(
        observations.bbox[indexes], int(observations.rotation[indexes[0]])
    )
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


OCR_LAYOUT_HOOKS = LayoutHooks(SOURCE_LABELS, group_order)

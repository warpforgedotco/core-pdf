# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import unicodedata
from functools import partial

import numpy

from core_pdf.impl import extract_block_layout as native_layout
from core_pdf.impl.extract_contracts import ObservationBatch
from core_pdf_ocr.impl.extract.contracts import ObservationSource

SOURCE_LABELS = {
    int(ObservationSource.NATIVE): "native",
    int(ObservationSource.OCR): "ocr",
}
OCR_SOURCE = numpy.uint8(ObservationSource.OCR)


def group_order(observations: ObservationBatch, indexes: numpy.ndarray) -> numpy.ndarray:
    if not bool((observations.source[indexes] == OCR_SOURCE).any()):
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


# The native layout with OCR observations labelled and ordered as OCR reads them.
layout_blocks_with_evidence = partial(
    native_layout.layout_blocks_with_evidence,
    source_labels=SOURCE_LABELS,
    group_order=group_order,
)

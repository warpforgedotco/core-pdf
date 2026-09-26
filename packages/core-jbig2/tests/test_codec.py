# SPDX-License-Identifier: AGPL-3.0-only

import pytest

from core_jbig2.codec import (
    JBIG2_IMMEDIATE_TEXT_REGION,
    JBIG2_PAGE_INFO,
    JBIG2PageDecoder,
    Jbig2ParseError,
    JBIG2Segment,
    Jbig2UnsupportedError,
)


def test_canvas_growth_preserves_pixels_and_finish_does_not_mutate_them() -> None:
    decoder = JBIG2PageDecoder()
    decoder.ensure_image(8, 1)
    assert decoder.image is not None
    decoder.image.data[0] = 0x80
    decoder.ensure_image(16, 2)
    assert (decoder.image.width, decoder.image.height) == (16, 2)
    assert decoder.finish() == b"\x80\x00\x00\x00"
    assert decoder.finish() == b"\x80\x00\x00\x00"


def test_finish_without_a_page_is_unsupported_not_empty() -> None:
    with pytest.raises(Jbig2UnsupportedError, match="produced no image"):
        JBIG2PageDecoder().finish()


def test_every_segment_type_is_read_by_default() -> None:
    reserved = JBIG2Segment(0, 1, 0, 1, b"")
    with pytest.raises(Jbig2ParseError, match="reserved JBIG2 segment type 1"):
        JBIG2PageDecoder().decode_segment(reserved)


def test_a_decoder_skips_the_segment_types_it_does_not_support() -> None:
    class PageOnly(JBIG2PageDecoder):
        supported_segment_types = frozenset({JBIG2_PAGE_INFO})

    decoder = PageOnly()
    decoder.decode_segment(JBIG2Segment(0, 1, 0, 1, b""))
    decoder.decode_segment(JBIG2Segment(1, JBIG2_IMMEDIATE_TEXT_REGION, 0, 1, b""))
    assert decoder.image is None

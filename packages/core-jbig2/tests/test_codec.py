# SPDX-License-Identifier: AGPL-3.0-only
"""T.88 page decoding keeps a growing canvas and reports its native polarity."""

import pytest

from core_jbig2.codec import JBIG2PageDecoder, Jbig2UnsupportedError


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

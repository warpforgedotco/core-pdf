# SPDX-License-Identifier: AGPL-3.0-only
"""Reader approximations for JBIG2 features without a supported decoder."""

from __future__ import annotations

from core_pdf_spec.s_07_filters.jbig2.bitmap_kernels import compose_packed_bitmap_data
from core_pdf_spec.s_07_filters.jbig2.codec import (
    JBIG2GenericRegionHeader,
    JBIG2PageDecoder,
    Jbig2ParseError,
    JBIG2Region,
    JBIG2Segment,
    compose_packed_bitmap_region,
)


class RecoveryJBIG2PageDecoder(JBIG2PageDecoder):
    """Preserve the reader's historical partial images and raw-region guesses."""

    def decode_segment(self, segment: JBIG2Segment) -> None:
        if segment.segment_type in (48, 6, 38, 39):
            super().decode_segment(segment)

    def decode_text_region(self, region: JBIG2Region) -> None:
        image = self.image
        if image is None:
            return
        if len(region.raw) < 20:
            raise Jbig2ParseError("truncated JBIG2 text region")
        bitmap = region.raw[20:]
        row_bytes = max(1, (region.width + 7) // 8)
        compose_packed_bitmap_data(
            bitmap,
            min(region.height, len(bitmap) // row_bytes),
            region.width,
            region.x,
            region.y,
            image.width,
            image.height,
            image.stride,
            image.data,
            0,
        )

    def decode_generic_region(self, header: JBIG2GenericRegionHeader) -> None:
        region = header.region
        if region.width <= 0 or region.height <= 0:
            return
        if header.mmr:
            if self.image is not None:
                compose_packed_bitmap_region(
                    region,
                    region.raw[header.bitmap_start :],
                    self.image,
                    self.page_info,
                )
            return
        super().decode_generic_region(header)

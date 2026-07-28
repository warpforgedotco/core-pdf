# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_jpeg.impl.codecs.jpx.boxes import Jp2Parser
from core_jpeg.impl.codecs.jpx.codestream import JpxImage
from core_jpeg.impl.codecs.jpx.color import (
    apply_jp2_container_transforms,
    apply_jp2_embedded_color_transforms,
    jp2_color_space_kind,
)
from core_jpeg.impl.codecs.jpx.output import (
    decoded_jpx_image_from_interleaved,
    decoded_jpx_native_components,
    jp2_decode_component_mode,
    jp2_preserves_native_component_output,
)
from core_jpeg.impl.errors import JpegUnsupportedError
from core_jpeg.impl.models import DecodedJpxImage


def decode_jpx_image(
    data: bytes,
    *,
    apply_embedded_color: bool = True,
) -> DecodedJpxImage:
    jp2 = Jp2Parser(data).parse()
    img = JpxImage()
    if img.parse(jp2.codestream):
        component_mode = jp2_decode_component_mode(
            jp2,
            apply_embedded_color=apply_embedded_color,
        )
        color_mode = jp2_color_space_kind(jp2.color_specification) is not None
        native_components = decoded_jpx_native_components(img)
        components = (
            native_components
            if jp2_preserves_native_component_output(
                img,
                jp2,
                apply_embedded_color=apply_embedded_color,
            )
            else None
        )
        raw = img.to_raw(component_mode=component_mode)
        raw = apply_jp2_container_transforms(raw, img.width, img.height, jp2)
        if apply_embedded_color or color_mode:
            raw = apply_jp2_embedded_color_transforms(raw, img.width, img.height, jp2)
        return decoded_jpx_image_from_interleaved(
            raw,
            img,
            jp2,
            component_mode=component_mode,
            components=components,
            native_components=native_components,
        )
    raise JpegUnsupportedError("JPXDecode failed to parse codestream")


def decode_jpx(data: bytes, *, apply_embedded_color: bool = True) -> bytes:
    return decode_jpx_image(
        data,
        apply_embedded_color=apply_embedded_color,
    ).interleaved

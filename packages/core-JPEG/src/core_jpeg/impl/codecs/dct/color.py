# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

SCALEBITS = 16
ONE_HALF = 1 << (SCALEBITS - 1)
FIX_1_40200 = 91881
FIX_1_77200 = 116130
FIX_0_34414 = 22554
FIX_0_71414 = 46802

CB_TO_B = tuple(((FIX_1_77200 * (i - 128) + ONE_HALF) >> SCALEBITS) for i in range(256))
CB_TO_G = tuple((-FIX_0_34414 * (i - 128)) + ONE_HALF for i in range(256))
CR_TO_R = tuple(((FIX_1_40200 * (i - 128) + ONE_HALF) >> SCALEBITS) for i in range(256))
CR_TO_G = tuple((-FIX_0_71414 * (i - 128)) for i in range(256))


def clamp_u8(value: int) -> int:
    if value < 0:
        return 0
    if value > 255:
        return 255
    return value


def ycbcr_to_rgb_channels(y_value: int, cb_value: int, cr_value: int) -> tuple[int, int, int]:
    red = y_value + CR_TO_R[cr_value]
    green = y_value + ((CB_TO_G[cb_value] + CR_TO_G[cr_value]) >> SCALEBITS)
    blue = y_value + CB_TO_B[cb_value]
    return clamp_u8(red), clamp_u8(green), clamp_u8(blue)


def cmyk_to_rgb_channels(cyan: int, magenta: int, yellow: int, black: int) -> tuple[int, int, int]:
    return (
        255 - min(255, cyan + black),
        255 - min(255, magenta + black),
        255 - min(255, yellow + black),
    )


def inverted_cmyk_to_rgb_channels(
    cyan: int, magenta: int, yellow: int, black: int
) -> tuple[int, int, int]:
    return cmyk_to_rgb_channels(
        255 - cyan,
        255 - magenta,
        255 - yellow,
        255 - black,
    )


def ycck_to_rgb_channels(
    y_value: int, cb_value: int, cr_value: int, black: int
) -> tuple[int, int, int]:
    cyan, magenta, yellow = ycbcr_to_rgb_channels(y_value, cb_value, cr_value)
    return inverted_cmyk_to_rgb_channels(cyan, magenta, yellow, black)

# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import struct
from typing import cast

from core_jpeg.impl.codecs.jpx.structures import SubBand, TileComponent
from core_jpeg.impl.codecs.jpx.tier1_state import (
    T1_ORIENT_HH,
    T1_ORIENT_HL,
    T1_ORIENT_LH,
    T1_ORIENT_LL,
)
from core_jpeg.impl.errors import JpegParseError

_FLOAT32 = struct.Struct("f")


def float32(value: int | float) -> float:
    return _FLOAT32.unpack(_FLOAT32.pack(float(value)))[0]


def idwt_53(
    samples: list[int],
    offset: int,
    length: int,
    stride: int,
    low_count: int | None = None,
    *,
    low_on_even: bool = True,
) -> None:
    if length <= 0:
        return
    low_count = (length + 1) // 2 if low_count is None else low_count
    if low_count < 0 or low_count > length:
        raise ValueError("invalid JPX low-pass sample count")
    high_count = length - low_count
    low = [samples[offset + i * stride] for i in range(low_count)]
    high = [samples[offset + (low_count + i) * stride] for i in range(high_count)]
    out = [0] * length

    if length == 1:
        if low_on_even:
            out[0] = low[0]
        else:
            out[0] = trunc_divide_by_two(high[0])
    elif length == 2:
        if low_on_even:
            out[0] = low[0] - ((high[0] + 1) >> 1)
            out[1] = high[0] + out[0]
        else:
            out[1] = low[0] - ((high[0] + 1) >> 1)
            out[0] = high[0] + out[1]
    elif low_on_even:
        if low_count == 0:
            return
        s1n = low[0]
        d1n = high[0]
        s0n = s1n - ((d1n + 1) >> 1)
        i = 0
        j = 1
        out[0] = s0n
        while i < (length - 3):
            d1c = d1n
            s0c = s0n
            s1n = low[j]
            d1n = high[j]
            s0n = s1n - ((d1c + d1n + 2) >> 2)
            out[i + 1] = d1c + ((s0c + s0n) >> 1)
            out[i + 2] = s0n
            i += 2
            j += 1
        out[i] = s0n
        if length & 1:
            out[length - 1] = low[(length - 1) // 2] - ((d1n + 1) >> 1)
            out[length - 2] = d1n + ((s0n + out[length - 1]) >> 1)
        else:
            out[length - 1] = d1n + s0n
    else:
        if low_count == 0:
            return
        s1 = high[1] if high_count > 1 else high[0]
        dc = low[0] - ((high[0] + s1 + 2) >> 2)
        out[0] = high[0] + dc
        i = 1
        j = 1
        limit = length - 2 - (0 if length & 1 else 1)
        while i < limit:
            s2 = high[j + 1] if j + 1 < high_count else high[-1]
            dn = low[j] - ((s1 + s2 + 2) >> 2)
            out[i] = dc
            out[i + 1] = s1 + ((dn + dc) >> 1)
            dc = dn
            s1 = s2
            i += 2
            j += 1
        out[i] = dc
        if not (length & 1):
            dn = low[length // 2 - 1] - ((s1 + 1) >> 1)
            out[length - 2] = s1 + ((dn + dc) >> 1)
            out[length - 1] = dn
        else:
            out[length - 1] = s1 + dc

    for i, value in enumerate(out):
        samples[offset + i * stride] = value


def idwt_97(
    samples: list[float],
    offset: int,
    length: int,
    stride: int,
    low_count: int | None = None,
    *,
    low_on_even: bool = True,
) -> None:
    if length <= 1:
        return
    low_count = (length + 1) // 2 if low_count is None else low_count
    if low_count < 0 or low_count > length:
        raise ValueError("invalid JPX low-pass sample count")
    high_count = length - low_count
    if low_count == 0:
        return
    low = [samples[offset + i * stride] for i in range(low_count)]
    high = [samples[offset + (low_count + i) * stride] for i in range(high_count)]

    alpha = -1.586134342059924
    beta = -0.052980118572961
    gamma = 0.882911075530934
    delta = 0.443506852043971
    gain = 1.230174104914001
    # Match OpenJPEG's historical irreversible inverse DWT scaling.
    high_gain = 1.625732422
    f_alpha = float32(alpha)
    f_beta = float32(beta)
    f_gamma = float32(gamma)
    f_delta = float32(delta)
    f_gain = float32(gain)
    f_high_gain = float32(high_gain)

    for i, value in enumerate(low):
        low[i] = float32(float32(value) * f_gain)
    for i, value in enumerate(high):
        high[i] = float32(float32(value) * f_high_gain)

    if not low_on_even:
        full: list[float] = [0.0] * length
        low_i = 0
        high_i = 0
        for i in range(length):
            if i % 2:
                full[i] = low[low_i]
                low_i += 1
            else:
                full[i] = high[high_i]
                high_i += 1
        if high_count:
            idwt_97_update_full(full, low_on_even=False, coefficient=f_delta)
            idwt_97_update_full(full, low_on_even=True, coefficient=f_gamma)
            idwt_97_update_full(full, low_on_even=False, coefficient=f_beta)
            idwt_97_update_full(full, low_on_even=True, coefficient=f_alpha)
        for i, value in enumerate(full):
            samples[offset + i * stride] = value
        return

    for i in range(low_count):
        low[i] = float32(
            low[i] - float32(idwt_sample(high, i - 1) + idwt_sample(high, i)) * f_delta
        )
    for i in range(high_count):
        high[i] = float32(
            high[i] - float32(idwt_sample(low, i) + idwt_sample(low, i + 1)) * f_gamma
        )
    for i in range(low_count):
        low[i] = float32(low[i] - float32(idwt_sample(high, i - 1) + idwt_sample(high, i)) * f_beta)
    for i in range(high_count):
        high[i] = float32(
            high[i] - float32(idwt_sample(low, i) + idwt_sample(low, i + 1)) * f_alpha
        )

    for i in range(length):
        value = low[i // 2] if i % 2 == 0 else high[i // 2]
        samples[offset + i * stride] = value


def idwt_97_update_full(
    values: list[float],
    *,
    low_on_even: bool,
    coefficient: float,
) -> None:
    target_start = 0 if low_on_even else 1
    for i in range(target_start, len(values), 2):
        left = values[i - 1] if i > 0 else values[i + 1]
        right = values[i + 1] if i + 1 < len(values) else values[i - 1]
        values[i] = float32(values[i] - float32(left + right) * coefficient)


def idwt_sample(values: list[float], index: int) -> float:
    if not values:
        return 0.0
    if index < 0:
        return values[0]
    if index >= len(values):
        return values[-1]
    return values[index]


def inverse_reversible_mct(
    y_samples: list[int | float],
    db_samples: list[int | float],
    dr_samples: list[int | float],
) -> tuple[list[int | float], list[int | float], list[int | float]]:
    if len(y_samples) != len(db_samples) or len(y_samples) != len(dr_samples):
        raise ValueError("JPX MCT component sizes do not match")
    red: list[int | float] = []
    green: list[int | float] = []
    blue: list[int | float] = []
    for raw_y, raw_db, raw_dr in zip(y_samples, db_samples, dr_samples, strict=True):
        y = int(raw_y)
        db = int(raw_db)
        dr = int(raw_dr)
        g = y - ((db + dr) >> 2)
        red.append(dr + g)
        green.append(g)
        blue.append(db + g)
    return red, green, blue


def inverse_irreversible_mct(
    y_samples: list[int | float],
    cb_samples: list[int | float],
    cr_samples: list[int | float],
) -> tuple[list[float], list[float], list[float]]:
    if len(y_samples) != len(cb_samples) or len(y_samples) != len(cr_samples):
        raise ValueError("JPX MCT component sizes do not match")
    red: list[float] = []
    green: list[float] = []
    blue: list[float] = []
    f_cr_to_r = float32(1.402)
    f_cb_to_g = float32(0.34413)
    f_cr_to_g = float32(0.71414)
    f_cb_to_b = float32(1.772)
    for y, cb, cr in zip(y_samples, cb_samples, cr_samples, strict=True):
        f_y = float32(y)
        f_cb = float32(cb)
        f_cr = float32(cr)
        red.append(float32(f_y + f_cr * f_cr_to_r))
        green.append(float32(float32(f_y - f_cb * f_cb_to_g) - f_cr * f_cr_to_g))
        blue.append(float32(f_y + f_cb * f_cb_to_b))
    return red, green, blue


def inverse_mct(
    components: list[list[int | float]], reversible: bool
) -> tuple[list[int | float], list[int | float], list[int | float]]:
    if len(components) < 3:
        raise ValueError("JPX MCT requires at least three components")
    if reversible:
        return inverse_reversible_mct(components[0], components[1], components[2])
    return inverse_irreversible_mct(components[0], components[1], components[2])


def subband_gain(orientation: int) -> int:
    if orientation == T1_ORIENT_LL:
        return 0
    if orientation in (T1_ORIENT_LH, T1_ORIENT_HL):
        return 1
    if orientation == T1_ORIENT_HH:
        return 2
    raise ValueError("invalid JPX subband orientation")


def subband_quant_index(total_levels: int, subband: SubBand) -> int:
    if subband.orientation == T1_ORIENT_LL:
        return 0
    orientation_offset = {
        T1_ORIENT_HL: 0,
        T1_ORIENT_LH: 1,
        T1_ORIENT_HH: 2,
    }[subband.orientation]
    level_offset = total_levels - subband.level
    if level_offset < 0:
        raise ValueError("invalid JPX subband level")
    return 1 + level_offset * 3 + orientation_offset


def quant_step_size(
    mantissa: int,
    exponent: int,
    precision: int,
    gain: int,
    reversible: bool,
) -> float:
    if reversible:
        return 1.0
    rb = precision + gain
    return (1.0 + mantissa / 2048.0) * (2.0 ** (rb - exponent))


def quant_num_bitplanes(exponent: int, guard_bits: int) -> int:
    return max(0, exponent + guard_bits - 1)


def resolve_quant_step(
    quant_style: int,
    quant_steps: list[tuple[int, int]],
    total_levels: int,
    subband: SubBand,
) -> tuple[int, int]:
    if not quant_steps:
        raise JpegParseError("missing JPX quantization step")
    index = subband_quant_index(total_levels, subband)
    if quant_style == 1:
        if len(quant_steps) != 1:
            raise JpegParseError("invalid JPX derived quantization table")
        mantissa, base_exponent = quant_steps[0]
        # ISO equation E-5, clamped like OpenJPEG for malformed negative exponents.
        exponent = (
            base_exponent
            if subband.orientation == T1_ORIENT_LL
            else max(0, base_exponent - (total_levels - subband.level))
        )
        return mantissa, exponent
    if index >= len(quant_steps):
        raise JpegParseError("missing JPX quantization step")
    return quant_steps[index]


def trunc_divide_by_two(value: int) -> int:
    return value // 2 if value >= 0 else -((-value) // 2)


def dequantize_samples(
    samples: list[int],
    step_size: float,
    *,
    reversible: bool,
) -> list[int | float]:
    if reversible:
        return [trunc_divide_by_two(value) for value in samples]
    half_step = float32(float32(0.5) * float32(step_size))
    return [float32(value * half_step) for value in samples]


def apply_roi_shift_samples(samples: list[int], roi_shift: int) -> list[int]:
    if roi_shift <= 0:
        return list(samples)
    threshold = 1 << roi_shift
    shifted: list[int] = []
    for value in samples:
        magnitude = abs(value)
        if magnitude >= threshold:
            magnitude >>= roi_shift
            shifted.append(-magnitude if value < 0 else magnitude)
        else:
            shifted.append(value)
    return shifted


def apply_roi_shift_subband(subband: SubBand, roi_shift: int) -> None:
    if roi_shift > 0:
        subband.samples = cast(
            list[int | float],
            apply_roi_shift_samples(
                cast(list[int], subband.samples),
                roi_shift,
            ),
        )


def dequantize_subband(
    subband: SubBand,
    quant_steps: list[tuple[int, int]],
    quant_style: int,
    total_levels: int,
    precision: int,
    reversible: bool,
) -> None:
    mantissa, exponent = resolve_quant_step(
        quant_style,
        quant_steps,
        total_levels,
        subband,
    )
    step = quant_step_size(
        mantissa,
        exponent,
        precision,
        0 if not reversible else subband_gain(subband.orientation),
        reversible,
    )
    subband.samples = dequantize_samples(
        cast(list[int], subband.samples),
        step,
        reversible=reversible,
    )


def component_coefficients(
    component: TileComponent,
) -> tuple[list[int | float], int, int]:
    if not component.resolutions:
        return [], 0, 0
    canvas: list[int | float] = [0] * (component.width * component.height)
    ll = component.resolutions[0].subbands[0]
    copy_band(canvas, component.width, component.height, 0, 0, ll)
    for res_index in range(1, len(component.resolutions)):
        res = component.resolutions[res_index]
        prev = component.resolutions[res_index - 1]
        x0 = prev.width
        y0 = prev.height
        for subband in res.subbands:
            if subband.orientation == T1_ORIENT_LH:
                copy_band(canvas, component.width, component.height, 0, y0, subband)
            elif subband.orientation == T1_ORIENT_HL:
                copy_band(canvas, component.width, component.height, x0, 0, subband)
            elif subband.orientation == T1_ORIENT_HH:
                copy_band(canvas, component.width, component.height, x0, y0, subband)
            elif subband.orientation == T1_ORIENT_LL:
                copy_band(canvas, component.width, component.height, 0, 0, subband)
    return canvas, component.width, component.height


def synthesize_component(
    component: TileComponent,
    reversible: bool,
) -> tuple[list[int | float], int, int]:
    samples, width, height = component_coefficients(component)
    if width <= 0 or height <= 0:
        return samples, width, height
    if not reversible:
        float_samples = [float(value) for value in samples]
        for res_index in range(1, len(component.resolutions)):
            previous = component.resolutions[res_index - 1]
            active_width = component.resolutions[res_index].width
            active_height = component.resolutions[res_index].height
            inverse_dwt_97_region(
                float_samples,
                width,
                active_width,
                active_height,
                low_width=previous.width,
                low_height=previous.height,
                low_on_even_x=component.resolutions[res_index].x0 % 2 == 0,
                low_on_even_y=component.resolutions[res_index].y0 % 2 == 0,
            )
        return float_samples, width, height
    int_samples = cast(list[int], samples)
    for res_index in range(1, len(component.resolutions)):
        previous = component.resolutions[res_index - 1]
        active_width = component.resolutions[res_index].width
        active_height = component.resolutions[res_index].height
        inverse_dwt_region(
            int_samples,
            width,
            active_width,
            active_height,
            reversible,
            low_width=previous.width,
            low_height=previous.height,
            low_on_even_x=component.resolutions[res_index].x0 % 2 == 0,
            low_on_even_y=component.resolutions[res_index].y0 % 2 == 0,
        )
    return cast(list[int | float], int_samples), width, height


def inverse_dwt_region(
    samples: list[int],
    stride: int,
    width: int,
    height: int,
    reversible: bool,
    *,
    low_width: int | None = None,
    low_height: int | None = None,
    low_on_even_x: bool = True,
    low_on_even_y: bool = True,
) -> None:
    if width <= 0 or height <= 0:
        return
    low_width = (width + 1) // 2 if low_width is None else low_width
    low_height = (height + 1) // 2 if low_height is None else low_height
    if reversible:
        for y in range(height):
            idwt_53(
                samples,
                y * stride,
                width,
                1,
                low_count=low_width,
                low_on_even=low_on_even_x,
            )
        for x in range(width):
            idwt_53(
                samples,
                x,
                height,
                stride,
                low_count=low_height,
                low_on_even=low_on_even_y,
            )
        return

    float_samples = [float(value) for value in samples]
    inverse_dwt_97_region(
        float_samples,
        stride,
        width,
        height,
        low_width=low_width,
        low_height=low_height,
        low_on_even_x=low_on_even_x,
        low_on_even_y=low_on_even_y,
    )
    for y in range(height):
        row = y * stride
        for x in range(width):
            index = row + x
            samples[index] = int(round(float_samples[index]))


def inverse_dwt_97_region(
    samples: list[float],
    stride: int,
    width: int,
    height: int,
    *,
    low_width: int | None = None,
    low_height: int | None = None,
    low_on_even_x: bool = True,
    low_on_even_y: bool = True,
) -> None:
    if width <= 0 or height <= 0:
        return
    low_width = (width + 1) // 2 if low_width is None else low_width
    low_height = (height + 1) // 2 if low_height is None else low_height
    for y in range(height):
        idwt_97(
            samples,
            y * stride,
            width,
            1,
            low_count=low_width,
            low_on_even=low_on_even_x,
        )
    for x in range(width):
        idwt_97(
            samples,
            x,
            height,
            stride,
            low_count=low_height,
            low_on_even=low_on_even_y,
        )


def copy_band(
    canvas: list[int | float],
    canvas_width: int,
    canvas_height: int,
    x_offset: int,
    y_offset: int,
    subband: SubBand,
) -> None:
    if x_offset >= canvas_width or y_offset >= canvas_height:
        return
    width = min(subband.width, canvas_width - x_offset)
    height = min(subband.height, canvas_height - y_offset)
    if not any(subband.samples):
        return
    for y in range(height):
        dst = (y_offset + y) * canvas_width + x_offset
        src = y * subband.width
        row = subband.samples[src : src + width]
        if any(row):
            canvas[dst : dst + width] = row

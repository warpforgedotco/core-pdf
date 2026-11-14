# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 11.3.5: blend modes.

The reference implementation of the sixteen standard blend functions B(cb, cs): the twelve
separable modes of Table 134, applied to each component, and the four non-separable modes of
Table 135, applied to an RGB colour. Components are additive, in [0, 1]. Each formula is
written once per form (one float, or numpy arrays of them) with the same operations in the
same order, so both give the same float64 bits, and the compiled kernels in
core_pdf_cythonized._pixel_blend repeat them in that order too.

Where the clauses leave a floating point choice open, this resolves it so that the result is
the exact mathematical one:

- SetSat gives every component equal to the maximum s and every one equal to the minimum 0;
  only a component strictly between them takes the middle formula. Read literally, a tie for
  the maximum makes one of the tied components the "middle" one, whose formula
  (Cmax - Cmin) * s / (Cmax - Cmin) is s in exact arithmetic but may round away from it.
- ClipColor leaves a colour unchanged where its denominator, l - n or x - l, is zero: that
  happens only when every component equals the luminosity, and scaling the colour about its
  luminosity then changes nothing. Its maximum x is the one on entry, as the clause has it.
"""

from __future__ import annotations

from math import sqrt
from typing import Any, Literal

lazy import numpy

from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.exceptions import PdfUnsupportedError
from core_pdf.impl.spec.standards import PdfVersion, SemanticContext, require_recognized_version

BlendMode = Literal[
    "Normal",
    "Multiply",
    "Screen",
    "ColorDodge",
    "ColorBurn",
    "Overlay",
    "Darken",
    "Lighten",
    "HardLight",
    "SoftLight",
    "Difference",
    "Exclusion",
    "Hue",
    "Saturation",
    "Color",
    "Luminosity",
]
type BlendSamples = numpy.ndarray[Any, numpy.dtype[numpy.float64]]
type Rgb = tuple[float, float, float]
type RgbSamples = tuple[BlendSamples, BlendSamples, BlendSamples]

# Every standard blend mode, in the order of core's blend codes (core_pdf.impl.render.blend
# BlendOp and the compiled kernels' MODE_ constants): a mode's code is its index here.
BLEND_MODES: tuple[BlendMode, ...] = (
    "Normal",
    "Multiply",
    "Screen",
    "ColorDodge",
    "ColorBurn",
    "Overlay",
    "Darken",
    "Lighten",
    "HardLight",
    "SoftLight",
    "Difference",
    "Exclusion",
    "Hue",
    "Saturation",
    "Color",
    "Luminosity",
)
SEPARABLE_BLEND_MODES: frozenset[BlendMode] = frozenset(BLEND_MODES[:12])
NON_SEPARABLE_BLEND_MODES: frozenset[BlendMode] = frozenset(BLEND_MODES[12:])
# Table 134 also lists Compatible, deprecated in PDF 2.0, which is the same as Normal.
RECOGNIZED_BLEND_MODES: frozenset[str] = frozenset((*BLEND_MODES, "Compatible"))


@cites("ISO 32000-2", "11.3.5.2", since="2.0", rule="revised ColorDodge and ColorBurn")
@cites(
    "ISO 32000-2",
    "11.3.5.2",
    extension="ADBE/1.7/5",
    rule="revised ColorDodge and ColorBurn in PDF 1.7",
)
def revised_blending(context: SemanticContext | None) -> bool:
    if context is None:
        return True
    version = require_recognized_version(
        context, "blend semantics require a recognized PDF version"
    )
    if version == PdfVersion(2, 0):
        return True
    return version == PdfVersion(1, 7) and any(
        extension.prefix == "ADBE"
        and extension.base_version == PdfVersion(1, 7)
        and extension.extension_level == 5
        and extension.extension_revision is None
        for extension in context.extensions
    )


@cites("ISO 32000-2", "11.6.3", rule="an array of blend modes selects the first one recognized")
def first_recognized_blend_mode(names: list[str | None]) -> str:
    """The blend mode a deprecated array of names selects: the first name the reader
    recognizes, or Normal if it recognizes none."""
    for name in names:
        if name in RECOGNIZED_BLEND_MODES:
            return name
    return "Normal"


def blend_revision(mode: str, context: SemanticContext | None) -> bool:
    # Only ColorDodge and ColorBurn depend on the version; the rest never ask for it.
    return revised_blending(context) if mode in {"ColorDodge", "ColorBurn"} else True


def color_dodge(cb: float, cs: float, revised: bool) -> float:
    if revised and cb == 0.0:
        return 0.0
    if cb >= 1.0 - cs:
        return 1.0
    return cb / (1.0 - cs)


def color_burn(cb: float, cs: float, revised: bool) -> float:
    if revised and cb == 1.0:
        return 1.0
    if 1.0 - cb >= cs:
        return 0.0
    return 1.0 - (1.0 - cb) / cs


def hard_light(cb: float, cs: float) -> float:
    # Multiply(cb, 2 cs) up to one half, Screen(cb, 2 cs - 1) above it.
    if cs <= 0.5:
        return cb * (2.0 * cs)
    return 1.0 - (1.0 - cb) * (1.0 - (2.0 * cs - 1.0))


def soft_light_d(x: float) -> float:
    if x <= 0.25:
        return ((16.0 * x - 12.0) * x + 4.0) * x
    return sqrt(x)


def soft_light(cb: float, cs: float) -> float:
    if cs <= 0.5:
        return cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb)
    return cb + (2.0 * cs - 1.0) * (soft_light_d(cb) - cb)


@cites("ISO 32000-2", "11.3.5.2", introduced="1.4", rule="the standard separable blend modes")
def separable_blend(cb: float, cs: float, mode: str, revised: bool = True) -> float:
    """B(cb, cs) of a separable mode on one component, unchecked: Table 134's formulas."""
    if mode == "Normal":
        return cs
    if mode == "Multiply":
        return cb * cs
    if mode == "Screen":
        return 1.0 - (1.0 - cb) * (1.0 - cs)
    if mode == "Overlay":
        return hard_light(cs, cb)
    if mode == "Darken":
        return min(cb, cs)
    if mode == "Lighten":
        return max(cb, cs)
    if mode == "ColorDodge":
        return color_dodge(cb, cs, revised)
    if mode == "ColorBurn":
        return color_burn(cb, cs, revised)
    if mode == "HardLight":
        return hard_light(cb, cs)
    if mode == "SoftLight":
        return soft_light(cb, cs)
    if mode == "Difference":
        return abs(cb - cs)
    if mode == "Exclusion":
        return cb + cs - 2.0 * cb * cs
    raise PdfUnsupportedError(f"unsupported component blend mode: {mode}")


def lum(r: float, g: float, b: float) -> float:
    return 0.3 * r + 0.59 * g + 0.11 * b


def clip_color(r: float, g: float, b: float) -> Rgb:
    l = lum(r, g, b)  # noqa: E741 (the clause's name)
    n = min(r, g, b)
    x = max(r, g, b)
    if n < 0.0:
        low = l - n
        if low != 0.0:
            r = l + ((r - l) * l) / low
            g = l + ((g - l) * l) / low
            b = l + ((b - l) * l) / low
    if x > 1.0:
        high = x - l
        if high != 0.0:
            r = l + ((r - l) * (1.0 - l)) / high
            g = l + ((g - l) * (1.0 - l)) / high
            b = l + ((b - l) * (1.0 - l)) / high
    return r, g, b


def set_lum(r: float, g: float, b: float, l: float) -> Rgb:  # noqa: E741
    d = l - lum(r, g, b)
    return clip_color(r + d, g + d, b + d)


def sat(r: float, g: float, b: float) -> float:
    return max(r, g, b) - min(r, g, b)


def set_sat(r: float, g: float, b: float, s: float) -> Rgb:
    high = max(r, g, b)
    low = min(r, g, b)
    if not high > low:
        return 0.0, 0.0, 0.0
    span = high - low
    return (
        s if r == high else 0.0 if r == low else ((r - low) * s) / span,
        s if g == high else 0.0 if g == low else ((g - low) * s) / span,
        s if b == high else 0.0 if b == low else ((b - low) * s) / span,
    )


@cites("ISO 32000-2", "11.3.5.3", introduced="1.4", rule="the standard non-separable blend modes")
def non_separable_blend(backdrop: Rgb, source: Rgb, mode: str) -> Rgb:
    """B(Cb, Cs) of a non-separable mode on an RGB colour, unchecked: Table 135's formulas."""
    if mode == "Hue":
        return set_lum(*set_sat(*source, sat(*backdrop)), lum(*backdrop))
    if mode == "Saturation":
        return set_lum(*set_sat(*backdrop, sat(*source)), lum(*backdrop))
    if mode == "Color":
        return set_lum(*source, lum(*backdrop))
    if mode == "Luminosity":
        return set_lum(*backdrop, lum(*source))
    raise PdfUnsupportedError(f"unsupported blend mode: {mode}")


def check_unit(*values: float) -> None:
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError("blend components must be finite and between zero and one")


def blend_component(
    backdrop: float, source: float, mode: BlendMode, *, context: SemanticContext | None = None
) -> float:
    """B(cb, cs) of a separable mode."""
    revised = blend_revision(mode, context)
    check_unit(backdrop, source)
    if mode not in SEPARABLE_BLEND_MODES:
        raise PdfUnsupportedError(f"unsupported component blend mode: {mode}")
    return separable_blend(backdrop, source, mode, revised)


def blend_color(
    backdrop: Rgb, source: Rgb, mode: BlendMode, *, context: SemanticContext | None = None
) -> Rgb:
    """B(Cb, Cs) of any standard mode on an RGB colour."""
    revised = blend_revision(mode, context)
    check_unit(*backdrop, *source)
    if mode in NON_SEPARABLE_BLEND_MODES:
        return non_separable_blend(backdrop, source, mode)
    if mode not in SEPARABLE_BLEND_MODES:
        raise PdfUnsupportedError(f"unsupported blend mode: {mode}")
    return (
        separable_blend(backdrop[0], source[0], mode, revised),
        separable_blend(backdrop[1], source[1], mode, revised),
        separable_blend(backdrop[2], source[2], mode, revised),
    )


def unit_samples(*values: numpy.ndarray | float) -> tuple[BlendSamples, ...]:
    arrays = numpy.broadcast_arrays(*(numpy.asarray(v, dtype=numpy.float64) for v in values))
    for array in arrays:
        if not numpy.all((array >= 0.0) & (array <= 1.0)):
            raise ValueError("blend components must be finite and between zero and one")
    return arrays


def hard_light_samples(cb: BlendSamples, cs: BlendSamples) -> BlendSamples:
    return numpy.where(cs <= 0.5, cb * (2.0 * cs), 1.0 - (1.0 - cb) * (1.0 - (2.0 * cs - 1.0)))


def separable_samples(
    cb: BlendSamples, cs: BlendSamples, mode: str, revised: bool = True
) -> BlendSamples:
    """separable_blend on arrays of components, unchecked."""
    if mode == "Normal":
        return cs.copy()
    if mode == "Multiply":
        return cb * cs
    if mode == "Screen":
        return 1.0 - (1.0 - cb) * (1.0 - cs)
    if mode == "Overlay":
        return hard_light_samples(cs, cb)
    if mode == "Darken":
        return numpy.where(cs < cb, cs, cb)
    if mode == "Lighten":
        return numpy.where(cs > cb, cs, cb)
    if mode == "ColorDodge":
        result = numpy.ones(cb.shape, dtype=numpy.float64)
        numpy.divide(cb, 1.0 - cs, out=result, where=cb < 1.0 - cs)
        if revised:
            numpy.copyto(result, 0.0, where=cb == 0.0)
        return result
    if mode == "ColorBurn":
        result = numpy.ones(cb.shape, dtype=numpy.float64)
        numpy.divide(1.0 - cb, cs, out=result, where=1.0 - cb < cs)
        numpy.subtract(1.0, result, out=result)
        if revised:
            numpy.copyto(result, 1.0, where=cb == 1.0)
        return result
    if mode == "HardLight":
        return hard_light_samples(cb, cs)
    if mode == "SoftLight":
        d = numpy.where(cb <= 0.25, ((16.0 * cb - 12.0) * cb + 4.0) * cb, numpy.sqrt(cb))
        return numpy.where(
            cs <= 0.5, cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb), cb + (2.0 * cs - 1.0) * (d - cb)
        )
    if mode == "Difference":
        return numpy.abs(cb - cs)
    if mode == "Exclusion":
        return cb + cs - 2.0 * cb * cs
    raise PdfUnsupportedError(f"unsupported component blend mode: {mode}")


def lum_samples(r: BlendSamples, g: BlendSamples, b: BlendSamples) -> BlendSamples:
    return 0.3 * r + 0.59 * g + 0.11 * b


def clip_color_samples(r: BlendSamples, g: BlendSamples, b: BlendSamples) -> RgbSamples:
    l = lum_samples(r, g, b)  # noqa: E741
    n = numpy.minimum(numpy.minimum(r, g), b)
    x = numpy.maximum(numpy.maximum(r, g), b)
    low = l - n
    lift = (n < 0.0) & (low != 0.0)
    low = numpy.where(lift, low, 1.0)
    r = numpy.where(lift, l + ((r - l) * l) / low, r)
    g = numpy.where(lift, l + ((g - l) * l) / low, g)
    b = numpy.where(lift, l + ((b - l) * l) / low, b)
    high = x - l
    lower = (x > 1.0) & (high != 0.0)
    high = numpy.where(lower, high, 1.0)
    r = numpy.where(lower, l + ((r - l) * (1.0 - l)) / high, r)
    g = numpy.where(lower, l + ((g - l) * (1.0 - l)) / high, g)
    b = numpy.where(lower, l + ((b - l) * (1.0 - l)) / high, b)
    return r, g, b


def set_lum_samples(
    r: BlendSamples,
    g: BlendSamples,
    b: BlendSamples,
    l: BlendSamples,  # noqa: E741
) -> RgbSamples:
    d = l - lum_samples(r, g, b)
    return clip_color_samples(r + d, g + d, b + d)


def sat_samples(r: BlendSamples, g: BlendSamples, b: BlendSamples) -> BlendSamples:
    return numpy.maximum(numpy.maximum(r, g), b) - numpy.minimum(numpy.minimum(r, g), b)


def set_sat_samples(
    r: BlendSamples, g: BlendSamples, b: BlendSamples, s: BlendSamples
) -> RgbSamples:
    high = numpy.maximum(numpy.maximum(r, g), b)
    low = numpy.minimum(numpy.minimum(r, g), b)
    spread = high > low
    span = numpy.where(spread, high - low, 1.0)

    def component(c: BlendSamples) -> BlendSamples:
        middle = numpy.where(c == low, 0.0, ((c - low) * s) / span)
        return numpy.where(spread, numpy.where(c == high, s, middle), 0.0)

    return component(r), component(g), component(b)


def non_separable_samples(backdrop: RgbSamples, source: RgbSamples, mode: str) -> RgbSamples:
    """non_separable_blend on arrays of colours, unchecked."""
    if mode == "Hue":
        return set_lum_samples(
            *set_sat_samples(*source, sat_samples(*backdrop)), lum_samples(*backdrop)
        )
    if mode == "Saturation":
        return set_lum_samples(
            *set_sat_samples(*backdrop, sat_samples(*source)), lum_samples(*backdrop)
        )
    if mode == "Color":
        return set_lum_samples(*source, lum_samples(*backdrop))
    if mode == "Luminosity":
        return set_lum_samples(*backdrop, lum_samples(*source))
    raise PdfUnsupportedError(f"unsupported blend mode: {mode}")


def blend_components(
    backdrop: numpy.ndarray | float,
    source: numpy.ndarray | float,
    mode: BlendMode,
    *,
    context: SemanticContext | None = None,
) -> BlendSamples:
    """B(cb, cs) of a separable mode on arrays of components."""
    revised = blend_revision(mode, context)
    cb, cs = unit_samples(backdrop, source)
    if mode not in SEPARABLE_BLEND_MODES:
        raise PdfUnsupportedError(f"unsupported component blend mode: {mode}")
    return separable_samples(cb, cs, mode, revised)


def blend_colors(
    backdrop: tuple[numpy.ndarray | float, numpy.ndarray | float, numpy.ndarray | float],
    source: tuple[numpy.ndarray | float, numpy.ndarray | float, numpy.ndarray | float],
    mode: BlendMode,
    *,
    context: SemanticContext | None = None,
) -> RgbSamples:
    """B(Cb, Cs) of any standard mode on arrays of RGB colours."""
    revised = blend_revision(mode, context)
    br, bg, bb, sr, sg, sb = unit_samples(*backdrop, *source)
    if mode in NON_SEPARABLE_BLEND_MODES:
        return non_separable_samples((br, bg, bb), (sr, sg, sb), mode)
    if mode not in SEPARABLE_BLEND_MODES:
        raise PdfUnsupportedError(f"unsupported blend mode: {mode}")
    return (
        separable_samples(br, sr, mode, revised),
        separable_samples(bg, sg, mode, revised),
        separable_samples(bb, sb, mode, revised),
    )


__all__ = (
    "BLEND_MODES",
    "NON_SEPARABLE_BLEND_MODES",
    "RECOGNIZED_BLEND_MODES",
    "SEPARABLE_BLEND_MODES",
    "BlendMode",
    "BlendSamples",
    "blend_color",
    "blend_colors",
    "blend_component",
    "blend_components",
    "first_recognized_blend_mode",
    "non_separable_blend",
    "revised_blending",
    "separable_blend",
)

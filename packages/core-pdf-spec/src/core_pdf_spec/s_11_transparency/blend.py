# SPDX-License-Identifier: AGPL-3.0-only
"""Version-sensitive separable colour blending, before alpha compositing.

ISO 32000-1:2008, 11.3.5/Table 136 defines the original ColorDodge and
ColorBurn functions. Adobe's 1.7 ExtensionLevel 5 supplement, 3.1, and
ISO 32000-2:2020, 11.3.5/Table 134 revise their singular corner values.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.standards import PdfVersion, SemanticContext

BlendMode = Literal["ColorDodge", "ColorBurn"]
BlendSamples = numpy.ndarray[Any, numpy.dtype[numpy.float64]]


def internal_revised_blending(context: SemanticContext | None) -> bool:
    if context is None:
        return True
    version = context.version
    if version is None or not version.recognized:
        raise PdfUnsupportedError("blend semantics require a recognized PDF version")
    if version == PdfVersion(2, 0):
        return True
    # Exact audited identity; an arbitrary higher extension number is not
    # evidence that this particular extension's semantics have been selected.
    return version == PdfVersion(1, 7) and any(
        extension.prefix == "ADBE"
        and extension.base_version == PdfVersion(1, 7)
        and extension.extension_level == 5
        and extension.extension_revision is None
        for extension in context.extensions
    )


def blend_component(
    backdrop: float,
    source: float,
    mode: BlendMode,
    *,
    context: SemanticContext | None = None,
) -> float:
    """Blend one unit-range component, without changing its alpha.

    A recognized explicit context selects the historical or revised equation;
    no context selects the PDF 2.0 equation. Inputs must be finite and in [0, 1].
    This implements these two modes, not transparency conformance validation.
    """
    revised = internal_revised_blending(context)
    if not 0.0 <= backdrop <= 1.0 or not 0.0 <= source <= 1.0:
        raise ValueError("blend components must be finite and between zero and one")
    if mode == "ColorDodge":
        if revised and backdrop == 0.0:
            return 0.0
        if backdrop >= 1.0 - source:
            return 1.0
        return backdrop / (1.0 - source)
    if mode == "ColorBurn":
        if revised and backdrop == 1.0:
            return 1.0
        if 1.0 - backdrop >= source:
            return 0.0
        return 1.0 - (1.0 - backdrop) / source
    raise PdfUnsupportedError(f"unsupported component blend mode: {mode}")


def blend_components(
    backdrop: numpy.ndarray | float,
    source: numpy.ndarray | float,
    mode: BlendMode,
    *,
    context: SemanticContext | None = None,
) -> BlendSamples:
    """The scalar equation over broadcastable arrays, returning new float64 data.

    Arrays may have any shape and are not mutated. Values must be finite and
    unit-range, as for ``blend_component``. Singular branches never divide by
    zero, including when NumPy floating-point exceptions are enabled.
    """
    revised = internal_revised_blending(context)
    cb, cs = numpy.broadcast_arrays(
        numpy.asarray(backdrop, dtype=numpy.float64), numpy.asarray(source, dtype=numpy.float64)
    )
    if not numpy.all((cb >= 0.0) & (cb <= 1.0)) or not numpy.all((cs >= 0.0) & (cs <= 1.0)):
        raise ValueError("blend components must be finite and between zero and one")
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
    raise PdfUnsupportedError(f"unsupported component blend mode: {mode}")


__all__ = ("BlendMode", "BlendSamples", "blend_component", "blend_components")

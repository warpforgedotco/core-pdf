"""The compiled non-isolated composite against core_pdf_spec, which keeps the
backdrop-removal algorithm it mirrors.

remove_group_backdrop stays in spec as the definition. The kernel carries a
copy of its arithmetic, so these tests pin the copy to it: directly, over
samples, and composed, against the gather, spec call, clip and
blend_visible_pixels that core ran before the kernel.
"""

from typing import Any

import numpy
import pytest

from core_pdf.impl.render_blend import BlendOp, blend_op, blend_visible_pixels
from core_pdf.impl.render_target import composite_nonisolated_group
from core_pdf_cythonized import remove_group_backdrop_samples
from core_pdf_spec.s_11_transparency.groups import remove_group_backdrop
from core_pdf_spec.standards import PdfVersion, SemanticContext

REVISED = SemanticContext(PdfVersion(2, 0))
LEGACY = SemanticContext(PdfVersion(1, 4))


def sample_rows(rng: numpy.random.Generator, count: int) -> tuple[numpy.ndarray, ...]:
    def unit(*shape: int) -> numpy.ndarray:
        values = rng.integers(0, 256, shape).astype(numpy.float64) / 255.0
        edge = rng.integers(0, 6, shape)
        return numpy.where(edge == 0, 0.0, numpy.where(edge == 1, 1.0, values))

    accumulated = rng.random(count).astype(numpy.float32).astype(numpy.float64)
    accumulated[rng.integers(0, 5, count) == 0] = 0.0
    return unit(count, 3), unit(count), unit(count, 3), unit(count), accumulated


@pytest.mark.parametrize("seed", range(8))
def test_kernel_backdrop_removal_matches_spec_bitwise(seed: int) -> None:
    components, alpha, backdrop, backdrop_alpha, accumulated = sample_rows(
        numpy.random.default_rng(seed), 4096
    )
    expected, expected_alpha = remove_group_backdrop(
        components, alpha, backdrop, backdrop_alpha, accumulated
    )
    actual = remove_group_backdrop_samples(components, alpha, backdrop, backdrop_alpha, accumulated)
    assert actual.tobytes() == expected.tobytes()
    assert expected_alpha.tobytes() == accumulated.tobytes()


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_kernel_backdrop_removal_matches_spec_outside_the_unit_range() -> None:
    # Core calls spec with validate=False, so out-of-range and non-finite
    # samples are part of the contract too.
    rng = numpy.random.default_rng(99)
    components = rng.normal(0.5, 2.0, (512, 3))
    alpha = rng.normal(0.5, 2.0, 512)
    backdrop = rng.normal(0.5, 2.0, (512, 3))
    backdrop_alpha = rng.normal(0.5, 2.0, 512)
    accumulated = rng.normal(0.5, 2.0, 512)
    accumulated[:8] = [0.0, -0.0, numpy.nan, numpy.inf, -numpy.inf, 1e-300, 1e300, 1.0]
    expected, _ = remove_group_backdrop(
        components, alpha, backdrop, backdrop_alpha, accumulated, validate=False
    )
    actual = remove_group_backdrop_samples(components, alpha, backdrop, backdrop_alpha, accumulated)
    assert actual.tobytes() == expected.tobytes()


def reference_composite(
    destination: numpy.ndarray,
    rendered: numpy.ndarray,
    source_alpha: numpy.ndarray,
    opacity: float,
    blend_mode: str | None,
    semantic_context: SemanticContext,
    mask_alpha: numpy.ndarray | None,
) -> numpy.ndarray[Any, Any]:
    """The numpy general branch the kernel replaced, calling spec itself."""
    mode = blend_op(blend_mode, casefold=True)
    normal = mode is None or mode is BlendOp.NORMAL
    scaled_alpha = source_alpha.astype(numpy.float64) * opacity * 255.0
    if mask_alpha is not None:
        scaled_alpha *= mask_alpha
    effective_alpha = numpy.rint(scaled_alpha).astype(numpy.uint8)
    visible = effective_alpha > 0
    if not numpy.any(visible):
        return effective_alpha
    if opacity == 1.0 and normal:
        unchanged_alpha = visible & (mask_alpha == 1.0)
        destination[unchanged_alpha] = rendered[unchanged_alpha]
        visible &= ~unchanged_alpha
        if not numpy.any(visible):
            return effective_alpha
    backdrop = destination[visible].astype(numpy.float64)
    result = rendered[visible].astype(numpy.float64) / 255.0
    colors, _ = remove_group_backdrop(
        result[..., :3],
        result[..., 3],
        backdrop[..., :3] / 255.0,
        backdrop[..., 3] / 255.0,
        source_alpha[visible],
        validate=False,
    )
    colors = numpy.clip(colors, 0.0, 1.0)
    blend_visible_pixels(
        destination,
        visible,
        colors[..., 0],
        colors[..., 1],
        colors[..., 2],
        effective_alpha[visible].astype(numpy.float64) / 255.0,
        mode,
        semantic_context=semantic_context,
    )
    return effective_alpha


@pytest.mark.parametrize(
    "blend_mode", [None, "Normal", "Multiply", "Screen", "ColorDodge", "ColorBurn", "Darken"]
)
@pytest.mark.parametrize("semantic_context", [REVISED, LEGACY], ids=["pdf-2.0", "pdf-1.4"])
@pytest.mark.parametrize("opacity", [0.3, 1.0])
@pytest.mark.parametrize("masked", [False, True])
def test_group_composite_matches_the_spec_composition(
    blend_mode: str | None, semantic_context: SemanticContext, opacity: float, masked: bool
) -> None:
    if opacity == 1.0 and blend_mode in {None, "Normal"} and not masked:
        pytest.skip("composite_elementary_normal's route, not the backdrop removal")
    rng = numpy.random.default_rng(len(str(blend_mode)) * 7 + int(opacity * 10) + masked)
    height, width = 11, 13
    page = rng.integers(0, 256, (height + 2, width + 3, 4), dtype=numpy.uint8)
    page[..., 3] = numpy.where(rng.integers(0, 3, page.shape[:2]) == 0, 0, page[..., 3])
    rendered = rng.integers(0, 256, (height, width, 4), dtype=numpy.uint8)
    source_plane = rng.random((height + 2, width + 3), dtype=numpy.float32)
    source_alpha = source_plane[1 : 1 + height, 2 : 2 + width]
    mask_alpha = None
    if masked:
        mask_alpha = rng.random((height, width), dtype=numpy.float32)
        mask_alpha[::3] = 1.0
    expected_page = page.copy()
    expected_alpha = reference_composite(
        expected_page[1 : 1 + height, 2 : 2 + width],
        rendered,
        source_alpha,
        opacity,
        blend_mode,
        semantic_context,
        mask_alpha,
    )
    actual_alpha = composite_nonisolated_group(
        page[1 : 1 + height, 2 : 2 + width],
        rendered,
        source_alpha,
        opacity,
        blend_mode,
        semantic_context=semantic_context,
        mask_alpha=mask_alpha,
    )
    assert numpy.array_equal(actual_alpha, expected_alpha)
    assert page.tobytes() == expected_page.tobytes()


@pytest.mark.parametrize("palette", [3, 40, 256])
@pytest.mark.parametrize("blend_mode", ["Multiply", "ColorBurn", None])
def test_large_planes_with_repeating_pixels_match_the_spec_composition(
    palette: int, blend_mode: str | None
) -> None:
    # The kernel caches results by input; planes drawing from a small palette
    # repeat inputs in runs and across rows, and a full palette collides.
    rng = numpy.random.default_rng(palette)
    height, width = 120, 170
    colors = rng.integers(0, 256, (palette, 4), dtype=numpy.uint8)
    destination = colors[rng.integers(0, palette, (height, width))]
    destination = numpy.repeat(destination[:, ::2], 2, axis=1)[:, :width].copy()
    rendered = colors[rng.integers(0, palette, (height, width))]
    levels = numpy.linspace(0.0, 1.0, max(2, palette // 4), dtype=numpy.float32)
    source_alpha = levels[rng.integers(0, len(levels), (height, width))]
    expected = destination.copy()
    expected_alpha = reference_composite(
        expected, rendered, source_alpha, 0.7, blend_mode, REVISED, None
    )
    actual_alpha = composite_nonisolated_group(
        destination, rendered, source_alpha, 0.7, blend_mode, semantic_context=REVISED
    )
    assert numpy.array_equal(actual_alpha, expected_alpha)
    assert destination.tobytes() == expected.tobytes()

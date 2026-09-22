import numpy as np
import pytest

from core_pdf.impl.capture.program import CapturedProgram
from core_pdf.impl.capture.records import CapturedDrawing, CapturedPath, CapturedSoftMask
from core_pdf.impl.render.target import internal_resolve_soft_mask
from tests.src.core_pdf.test_pattern_rendering import internal_target


def internal_mask_program() -> CapturedProgram:
    path = CapturedPath()
    path.rect(0, 0, 2, 2)
    return CapturedProgram(
        drawings=(CapturedDrawing(0, (1, 0, 0), 1, path=path, bbox=(0, 0, 2, 2)),)
    )


@pytest.mark.parametrize("backdrop_alpha", [0, 255])
@pytest.mark.parametrize("offset", [(0, 0), (1, 1)])
@pytest.mark.parametrize("invert", [False, True])
def test_mask_pixels_ignore_destination_clip_and_backdrop(
    backdrop_alpha: int, offset: tuple[int, int], invert: bool
) -> None:
    target = internal_target(4, 4)
    target.pixels[:] = bytes([20, 40, 60, backdrop_alpha]) * 16
    original_pixels = bytes(target.pixels)
    clip = CapturedPath()
    clip.rect(3, 3, 1, 1)
    target.clip.push(clip, "nonzero")
    samples: list[float] = []

    def transfer(alpha: float) -> tuple[float, ...]:
        samples.append(alpha)
        return (1 - alpha,)

    mask = CapturedSoftMask(internal_mask_program(), transfer if invert else None, offset)
    result = internal_resolve_soft_mask(target, mask)
    assert result is not None
    expected = np.zeros((4, 4), dtype=np.float32)
    x, y = offset
    expected[2 - y : 4 - y, x : x + 2] = 1
    np.testing.assert_array_equal(result, 1 - expected if invert else expected)
    assert result.dtype == np.float32
    assert not result.flags.writeable
    assert bytes(target.pixels) == original_pixels
    assert target.clip.depth == 1
    assert not target.active_soft_masks
    assert samples == ([0, 1] if invert else [])
    assert internal_resolve_soft_mask(target, mask) is result
    assert samples == ([0, 1] if invert else [])


def test_sibling_reuses_mask_cache_without_sharing_paint_state() -> None:
    target = internal_target(4, 4)
    mask = CapturedSoftMask(internal_mask_program())
    result = internal_resolve_soft_mask(target, mask)
    assert result is not None
    sibling, pixels = target.blank_sibling()
    assert internal_resolve_soft_mask(sibling, mask) is result
    assert sibling.prepared_image_cache is target.prepared_image_cache
    assert sibling.tiling_cell_cache is target.tiling_cell_cache
    assert sibling.active_soft_masks is target.active_soft_masks
    sibling.blend_normal_pixel(0, 255, 0, 0, 255)
    np.testing.assert_array_equal(pixels[0, 0], [255, 0, 0, 255])
    assert not any(target.pixels)


def test_transfer_failure_is_cached_and_does_not_poison_other_masks() -> None:
    target = internal_target(4, 4)
    calls = 0

    def transfer(alpha: float) -> tuple[float, ...]:
        nonlocal calls
        calls += 1
        raise ValueError("invalid mask transfer")

    program = internal_mask_program()
    mask = CapturedSoftMask(program, transfer)
    assert internal_resolve_soft_mask(target, mask) is None
    assert not target.active_soft_masks
    assert internal_resolve_soft_mask(target, mask) is None
    assert calls == 1
    valid = internal_resolve_soft_mask(target, CapturedSoftMask(program))
    assert valid is not None
    assert np.count_nonzero(valid) == 4
    assert not target.active_soft_masks

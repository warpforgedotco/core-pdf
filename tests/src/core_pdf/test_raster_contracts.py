import numpy as np
import pytest

from core_pdf.impl._impl.capture.records import CapturedPath
from core_pdf.impl._impl.render import target as target_module
from core_pdf.impl._impl.render.model import DisplayListItem
from core_pdf.impl._impl.render.target import internal_RasterTarget
from tests.src.core_pdf.test_pattern_rendering import internal_target


@pytest.mark.parametrize(
    "quad",
    [
        ((0, 0), (4, 0), (0, 4), (4, 4)),
        ((4, 0), (0, 0), (4, 4), (0, 4)),
        ((0, 4), (0, 0), (4, 4), (4, 0)),
        ((0, 0), (3, 1), (1, 3), (4, 4)),
    ],
)
@pytest.mark.parametrize("opacity", [0, 0.5, 1])
@pytest.mark.parametrize("mask_kind", [None, "alpha", "soft"])
@pytest.mark.parametrize("clipped", [False, True])
def test_normal_blend_matches_optimized_sampler(quad, opacity, mask_kind, clipped):
    results = []
    source = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    mask = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    for blend in (None, "Normal"):
        target = internal_target(4, 4)
        target.pixels[:] = bytes([20, 40, 60, 255]) * 16
        if clipped:
            path = CapturedPath()
            path.rect(1, 1, 2, 2)
            target.clip.push(path, "nonzero")
        target.blit_affine_image(
            quad,
            source.reshape(-1),
            2,
            2,
            3,
            opacity,
            blend,
            source_alpha=mask if mask_kind == "alpha" else None,
            soft_mask=mask if mask_kind == "soft" else None,
        )
        results.append(np.frombuffer(target.pixels, dtype=np.uint8).astype(int))
    if mask_kind is None and opacity in {0, 1}:
        np.testing.assert_array_equal(results[0], results[1])
    else:
        np.testing.assert_allclose(results[0], results[1], atol=1, rtol=0)
    if opacity == 0:
        np.testing.assert_array_equal(results[0].reshape(-1, 4), [[20, 40, 60, 255]] * 16)


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("knockout", [False, True])
@pytest.mark.parametrize("opacity", [0, 0.5, 1])
def test_group_compositing_applies_opacity_once(isolated, knockout, opacity):
    target = internal_target(2, 2)
    original_buffer = target.pixels
    target.pixels[:] = bytes([10, 20, 30, 255]) * 4
    target.push_group(bytearray(16), opacity, None, isolated=isolated, knockout=knockout)
    target.blit_affine_image(((0, 0), (2, 0), (0, 2), (2, 2)), bytes([255, 0, 0]), 1, 1, 3, 1, None)
    target.composite_group(target.pop_group())
    assert target.pixels is original_buffer
    assert len(target.buffer_stack) == 1
    expected = [10 + 245 * opacity, 20 * (1 - opacity), 30 * (1 - opacity), 255]
    pixels = np.frombuffer(target.pixels, dtype=np.uint8).reshape(4, 4)
    np.testing.assert_allclose(pixels, [expected] * 4, atol=1, rtol=0)


@pytest.mark.parametrize("failure", ["prepare", "paint", "composite"])
def test_paint_failure_restores_shape_and_group_state(monkeypatch, failure):
    target = internal_target(2, 2)
    original_buffer = target.pixels
    target.paint_alpha_is_shape, target.shape_alpha = False, 0.25

    def fail(*args, **kwargs):
        raise RuntimeError("injected rendering failure")

    monkeypatch.setattr(target_module, "internal_graphics_soft_mask", lambda item: object())
    monkeypatch.setattr(
        target_module,
        "internal_resolve_soft_mask",
        fail if failure == "prepare" else lambda *a: np.ones((2, 2), dtype=np.float32),
    )
    if failure == "paint":
        monkeypatch.setattr(internal_RasterTarget, "internal_paint_item", fail)
    elif failure == "composite":
        monkeypatch.setattr(internal_RasterTarget, "composite_group", fail)
    with pytest.raises(RuntimeError, match="injected"):
        target.paint_item(DisplayListItem("shading", 0, {"alpha_is_shape": True}))
    assert (target.paint_alpha_is_shape, target.shape_alpha) == (False, 0.25)
    assert target.pixels is original_buffer
    assert len(target.buffer_stack) == 1
    assert target.group_source_alpha is None
    assert target.group_source_shape is None


@pytest.mark.parametrize("failure", ["clip", "paint", "composite"])
def test_scope_failure_restores_clip_and_unwinds_all_groups(monkeypatch, failure):
    target = internal_target(2, 2)
    original_buffer, original_clip_stack = target.pixels, target.clip_stack
    clip = CapturedPath()
    clip.rect(0, 0, 1, 1)

    def fail(*args, **kwargs):
        raise RuntimeError("injected scope failure")

    if failure == "clip":
        monkeypatch.setattr(type(target.clip), "push", fail)
    elif failure == "composite":
        monkeypatch.setattr(internal_RasterTarget, "composite_group", fail)

    def items():
        yield DisplayListItem("group-begin", 0)
        yield DisplayListItem("group-begin", 1)
        if failure == "paint":
            fail()

    with pytest.raises(RuntimeError, match="injected"):
        target.paint_items(items(), translation=(0, 0), clip_path=clip)
    assert target.pixels is original_buffer
    assert len(target.buffer_stack) == 1
    assert target.scope_stack == []
    assert target.clip_stack is original_clip_stack
    assert target.clip.depth == 0
    assert target.clip_floor == 0
    assert target.group_floor == 1


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("blend", [None, "Multiply"])
def test_tiling_patterns_repeat_inside_the_path_clip(nested, blend):
    from core_pdf.impl._impl.capture.program import CapturedProgram
    from core_pdf.impl._impl.capture.records import CapturedDrawing, TilingPattern
    from core_pdf.impl._impl.render.display import DisplayList

    cell_path = CapturedPath()
    cell_path.rect(0, 0, 1, 1)
    cell = CapturedDrawing(0, (1, 0, 0), 1, path=cell_path, bbox=(0, 0, 1, 1))
    pattern = TilingPattern((0, 0, 1, 1), 1, 1, CapturedProgram(drawings=(cell,)))
    if nested:
        cell = CapturedDrawing(0, None, 1, path=cell_path, bbox=(0, 0, 1, 1), fill_pattern=pattern)
        pattern = TilingPattern((0, 0, 1, 1), 1, 1, CapturedProgram(drawings=(cell,)))
    path = CapturedPath()
    path.rect(1, 1, 2, 2)
    display = DisplayList(4, 4)
    display.append_captured_drawing(
        CapturedDrawing(
            0, None, 1, path=path, bbox=(1, 1, 3, 3), fill_pattern=pattern, blend_mode=blend
        )
    )
    target = internal_target(4, 4)
    target.pixels[:] = bytes([200, 200, 200, 255]) * 16
    target.paint_items(display.items)
    actual = np.frombuffer(target.pixels, dtype=np.uint8).reshape(4, 4, 4)
    expected = np.full((4, 4, 4), [200, 200, 200, 255], dtype=np.uint8)
    expected[1:3, 1:3] = [255 if blend is None else 200, 0, 0, 255]
    np.testing.assert_array_equal(actual, expected)
    assert len(target.buffer_stack) == 1
    assert target.clip.depth == 0
    assert target.scope_stack == []

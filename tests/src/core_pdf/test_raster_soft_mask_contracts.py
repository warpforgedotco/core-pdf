from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from core_pdf.impl.capture_program import CapturedProgram
from core_pdf.impl.capture_records import CapturedDrawing, CapturedPath, CapturedSoftMask
from core_pdf.impl.render_target import resolve_soft_mask
from tests.src.core_pdf.raster_support import make_target

if TYPE_CHECKING:
    from core_pdf.impl.capture_recording import TextState
    from core_pdf.impl.capture_tolerant_state import RecoveringTextState
    from core_pdf_spec.s_11_transparency.soft_masks import SoftMask


def mask_program() -> CapturedProgram:
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
    target = make_target(4, 4)
    target.pixels[:] = bytes([20, 40, 60, backdrop_alpha]) * 16
    original_pixels = bytes(target.pixels)
    clip = CapturedPath()
    clip.rect(3, 3, 1, 1)
    target.clip.push(clip, "nonzero")
    samples: list[float] = []

    def transfer(alpha: float) -> tuple[float, ...]:
        samples.append(alpha)
        return (1 - alpha,)

    mask = CapturedSoftMask(mask_program(), transfer if invert else None, offset)
    result = resolve_soft_mask(target, mask)
    assert result is not None
    expected = np.zeros((4, 4), dtype=np.float32)
    x, y = offset
    expected[2 - y : 4 - y, x : x + 2] = 1
    plane = result[...]
    np.testing.assert_array_equal(plane, 1 - expected if invert else expected)
    assert plane.dtype == np.float32
    assert not result.alpha.flags.writeable
    np.testing.assert_array_equal(result[1:3, 2:4], plane[1:3, 2:4])
    assert bytes(target.pixels) == original_pixels
    assert target.clip.depth == 1
    assert not target.resources.active_soft_masks
    assert samples == ([0, 1] if invert else [])
    assert resolve_soft_mask(target, mask) is result
    assert samples == ([0, 1] if invert else [])


def test_sibling_reuses_mask_cache_without_sharing_paint_state() -> None:
    target = make_target(4, 4)
    mask = CapturedSoftMask(mask_program())
    result = resolve_soft_mask(target, mask)
    assert result is not None
    sibling, pixels = target.blank_sibling()
    assert resolve_soft_mask(sibling, mask) is result
    assert sibling.resources is target.resources
    sibling.blend_px(0, (255, 0, 0, 255), None)
    np.testing.assert_array_equal(pixels[0, 0], [255, 0, 0, 255])
    assert not any(target.pixels)


def test_transfer_runs_only_on_alphas_the_mask_holds() -> None:
    target = make_target(4, 4)
    samples: list[float] = []

    def transfer(alpha: float) -> tuple[float, ...]:
        if alpha not in (0.0, 1.0):
            raise ValueError("undefined here")
        samples.append(alpha)
        return (0.25 + alpha / 2,)

    result = resolve_soft_mask(target, CapturedSoftMask(mask_program(), transfer))
    assert result is not None
    assert samples == [0.0, 1.0]
    assert sorted(set(result[...].ravel().tolist())) == [0.25, 0.75]
    assert np.count_nonzero(result[...] == np.float32(0.75)) == 4


def test_transfer_failure_is_cached_and_does_not_poison_other_masks() -> None:
    target = make_target(4, 4)
    calls = 0

    def transfer(alpha: float) -> tuple[float, ...]:
        nonlocal calls
        calls += 1
        raise ValueError("invalid mask transfer")

    program = mask_program()
    mask = CapturedSoftMask(program, transfer)
    assert resolve_soft_mask(target, mask) is None
    assert not target.resources.active_soft_masks
    assert resolve_soft_mask(target, mask) is None
    assert calls == 1
    valid = resolve_soft_mask(target, CapturedSoftMask(program))
    assert valid is not None
    assert np.count_nonzero(valid[...]) == 4
    assert not target.resources.active_soft_masks


REPEATED_SOFT_MASK_PDF = (
    Path(__file__).resolve().parents[3] / "tests/fixtures/PyMuPDF/tests/resources/test_4942.pdf"
)


@pytest.fixture
def state(text_pdf_bytes: bytes) -> Iterator[TextState]:
    from core_pdf import PdfDocument
    from core_pdf.impl.capture_recording import TextState

    with PdfDocument(text_pdf_bytes) as document:
        yield TextState(document)


@pytest.fixture
def parses(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    from core_pdf.impl import capture_tolerant_state as tolerant_state

    reached: list[object] = []
    monkeypatch.setattr(
        tolerant_state,
        "parse_soft_mask",
        lambda value, resolver, *, ctm, compile_function: reached.append(value),
    )
    return reached


def test_a_repeated_soft_mask_is_parsed_once_per_distinct_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core_pdf import PdfDocument
    from core_pdf.impl import capture_tolerant_state as tolerant_state

    calls = 0
    produced: list[int] = []
    original = tolerant_state.RecoveringTextState.resolve_soft_mask

    def counting(self: RecoveringTextState, value: object) -> SoftMask | None:
        nonlocal calls
        calls += 1
        mask = original(self, value)
        if mask is not None:
            produced.append(id(mask))
        return mask

    monkeypatch.setattr(tolerant_state.RecoveringTextState, "resolve_soft_mask", counting)
    with PdfDocument(REPEATED_SOFT_MASK_PDF.read_bytes()) as document:
        document.pages[0].extract()

    assert calls > 0
    assert len(set(produced)) <= calls // 2


def test_the_same_source_under_the_same_state_parses_once(
    state: TextState, parses: list[object]
) -> None:
    value = {"S": "Alpha"}
    assert state.resolve_soft_mask(value) is None
    assert state.resolve_soft_mask(value) is None
    assert state.resolve_soft_mask(value) is None
    assert len(parses) == 1
    assert len(state.caches.parsed_soft_masks) == 1


def test_the_parse_cache_separates_masks_by_transform(
    state: TextState, parses: list[object]
) -> None:
    from core_pdf_spec.s_08_graphics.matrix import Matrix

    value = {"S": "Alpha"}
    state.resolve_soft_mask(value)
    state.graphics.ctm = Matrix(2.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    state.resolve_soft_mask(value)
    assert len(parses) == 2


def test_the_parse_cache_separates_masks_by_resource_scope(
    state: TextState, parses: list[object]
) -> None:
    value = {"S": "Alpha"}
    state.resolve_soft_mask(value)
    state.resources = dict(state.resources)
    state.resolve_soft_mask(value)
    assert len(parses) == 2


def test_the_parse_cache_is_bounded(state: TextState, parses: list[object]) -> None:
    state.caches.parsed_soft_masks.limit = 3
    for index in range(4):
        state.resolve_soft_mask({"S": "Alpha", "n": index})
    assert len(state.caches.parsed_soft_masks) <= 3


def test_a_nested_capture_shares_the_parse_cache(state: TextState) -> None:
    assert state.nested_capture_state().caches.parsed_soft_masks is state.caches.parsed_soft_masks


def make_plane(megabytes: float) -> np.ndarray:
    return np.zeros(int(megabytes * 1e6 // 4), dtype=np.float32)


def plane_key(index: int) -> tuple[int, int, tuple[float, float]]:
    return (index, 0, (0.0, 0.0))


def make_mask() -> CapturedSoftMask:
    return CapturedSoftMask(mask_program())


def store_plane(cache, key, plane) -> None:
    cache.store(key, (make_mask(), plane), 0 if plane is None else plane.nbytes)


def test_the_plane_cache_evicts_oldest_once_the_budget_is_spent() -> None:
    from core_pdf.impl.caches import ByteBudgetCache

    cache = ByteBudgetCache(budget=10_000_000)
    for index in range(4):
        store_plane(cache, plane_key(index), make_plane(3))
    assert cache.size <= cache.budget
    assert cache.get(plane_key(0)) is None
    assert cache.get(plane_key(3)) is not None


def test_a_plane_larger_than_the_budget_is_not_cached_at_all() -> None:
    from core_pdf.impl.caches import ByteBudgetCache

    cache = ByteBudgetCache(budget=1_000_000)
    store_plane(cache, plane_key(1), make_plane(5))
    assert cache.get(plane_key(1)) is None
    assert cache.size == 0


def test_restoring_a_key_does_not_double_count_its_bytes() -> None:
    from core_pdf.impl.caches import ByteBudgetCache

    cache = ByteBudgetCache(budget=10_000_000)
    store_plane(cache, plane_key(1), make_plane(2))
    first = cache.size
    store_plane(cache, plane_key(1), make_plane(2))
    assert cache.size == first


def test_a_cached_none_plane_costs_nothing() -> None:
    from core_pdf.impl.caches import ByteBudgetCache

    cache = ByteBudgetCache(budget=1_000)
    store_plane(cache, plane_key(1), None)
    assert cache.size == 0
    assert cache.get(plane_key(1)) is not None

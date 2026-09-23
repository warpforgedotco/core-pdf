from pathlib import Path

import numpy as np
import pytest

from core_pdf.impl.capture.program import CapturedProgram
from core_pdf.impl.capture.records import CapturedDrawing, CapturedPath, CapturedSoftMask
from core_pdf.impl.render.target import resolve_soft_mask
from tests.src.core_pdf.test_pattern_rendering import make_target


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
    np.testing.assert_array_equal(result, 1 - expected if invert else expected)
    assert result.dtype == np.float32
    assert not result.flags.writeable
    assert bytes(target.pixels) == original_pixels
    assert target.clip.depth == 1
    assert not target.active_soft_masks
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
    assert sibling.prepared_image_cache is target.prepared_image_cache
    assert sibling.tiling_cell_cache is target.tiling_cell_cache
    assert sibling.active_soft_masks is target.active_soft_masks
    sibling.blend_px(0, (255, 0, 0, 255), None)
    np.testing.assert_array_equal(pixels[0, 0], [255, 0, 0, 255])
    assert not any(target.pixels)


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
    assert not target.active_soft_masks
    assert resolve_soft_mask(target, mask) is None
    assert calls == 1
    valid = resolve_soft_mask(target, CapturedSoftMask(program))
    assert valid is not None
    assert np.count_nonzero(valid) == 4
    assert not target.active_soft_masks


# A page dense in repeated soft masks; the parse cache exists for this shape.
REPEATED_SOFT_MASK_PDF = (
    Path(__file__).resolve().parents[3] / "tests/fixtures/PyMuPDF/tests/resources/test_4942.pdf"
)


@pytest.fixture
def state(text_pdf_bytes):
    """A capture state over a trivial document.

    The cache tests stub parse_soft_mask out, so nothing here reads the
    document's content and a real mask would only slow them down.
    """
    from core_pdf import PdfDocument
    from core_pdf.impl.capture.recording import TextState

    with PdfDocument(text_pdf_bytes) as document:
        yield TextState(document)


@pytest.fixture
def parses(monkeypatch):
    """Counts reaching the real parse, standing in for its result."""
    from core_pdf.impl.capture import tolerant_state

    reached = []
    monkeypatch.setattr(
        tolerant_state,
        "parse_soft_mask",
        lambda value, resolver, *, ctm, compile_function: reached.append(value),
    )
    return reached


def test_a_repeated_soft_mask_is_parsed_once_per_distinct_state(monkeypatch) -> None:
    # Every gs used to re-parse the ExtGState soft mask into a fresh SoftMask
    # carrying a freshly compiled transfer closure, and every cache downstream
    # is keyed by one of those identities, so none of them could ever hit.
    from core_pdf import PdfDocument
    from core_pdf.impl.capture import tolerant_state

    calls = 0
    produced: list[int] = []
    original = tolerant_state.RecoveringTextState.resolve_soft_mask

    def counting(self, value):
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


def test_the_same_source_under_the_same_state_parses_once(state, parses) -> None:
    value = {"S": "Alpha"}
    assert state.resolve_soft_mask(value) is None
    assert state.resolve_soft_mask(value) is None
    assert state.resolve_soft_mask(value) is None
    # A None result is the commonest case -- an Identity or absent mask -- so
    # it has to be cached too, not treated as a miss.
    assert len(parses) == 1
    assert len(state.parsed_soft_masks) == 1


def test_the_parse_cache_separates_masks_by_transform(state, parses) -> None:
    # The ctm is baked into the parsed mask, so the same source under a
    # different transform must not be served from the cache.
    from core_pdf_spec.s_08_graphics.matrix import Matrix

    value = {"S": "Alpha"}
    state.resolve_soft_mask(value)
    state.graphics.ctm = Matrix(2.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    state.resolve_soft_mask(value)
    assert len(parses) == 2


def test_the_parse_cache_separates_masks_by_resource_scope(state, parses) -> None:
    # A mask reached through different resources stays a separate object, so a
    # program captured under one scope is never reused under another.
    value = {"S": "Alpha"}
    state.resolve_soft_mask(value)
    state.resources = dict(state.resources)
    state.resolve_soft_mask(value)
    assert len(parses) == 2


def test_the_parse_cache_is_bounded(state, parses, monkeypatch) -> None:
    from core_pdf.impl.capture import tolerant_state

    monkeypatch.setattr(tolerant_state, "SOFT_MASK_CACHE_LIMIT", 3)
    for index in range(4):
        state.resolve_soft_mask({"S": "Alpha", "n": index})
    assert len(state.parsed_soft_masks) <= 3


def test_a_nested_capture_shares_the_parse_cache(state) -> None:
    # Patterns and mask groups each capture a whole nested content stream. If
    # the nested state started with an empty cache, every gs inside one would
    # mint a fresh mask identity again and miss every cache downstream.
    assert state.nested_capture_state().parsed_soft_masks is state.parsed_soft_masks


def make_plane(megabytes: float) -> np.ndarray:
    return np.zeros(int(megabytes * 1e6 // 4), dtype=np.float32)


def plane_key(index: int) -> tuple[int, int, tuple[float, float]]:
    return (index, 0, (0.0, 0.0))


def make_mask() -> CapturedSoftMask:
    return CapturedSoftMask(mask_program())


def test_the_plane_cache_evicts_oldest_once_the_budget_is_spent() -> None:
    # A resolved plane covers the whole page in float32, so a page using many
    # distinct masks kept far more than it could afford: one corpus page held
    # 1,598 planes totalling 3,208MB, 91% of its peak resident set.
    from core_pdf.impl.render.target import SoftMaskCache

    cache = SoftMaskCache(budget=10_000_000)
    for index in range(4):
        cache.store(plane_key(index), make_mask(), make_plane(3))
    assert cache.size <= cache.budget
    # The oldest went first, the newest are still there.
    assert cache.get(plane_key(0)) is None
    assert cache.get(plane_key(3)) is not None


def test_a_plane_larger_than_the_budget_keeps_only_the_negative_result() -> None:
    from core_pdf.impl.render.target import SoftMaskCache

    cache = SoftMaskCache(budget=1_000_000)
    mask = make_mask()
    cache.store(plane_key(1), mask, make_plane(5))
    entry = cache.get(plane_key(1))
    # Cached, so the mask is not resolved again, but the plane that would not
    # fit is dropped rather than blowing the budget.
    assert entry is not None
    assert entry[0] is mask
    assert entry[1] is None
    assert cache.size == 0


def test_restoring_a_key_does_not_double_count_its_bytes() -> None:
    from core_pdf.impl.render.target import SoftMaskCache

    cache = SoftMaskCache(budget=10_000_000)
    cache.store(plane_key(1), make_mask(), make_plane(2))
    first = cache.size
    cache.store(plane_key(1), make_mask(), make_plane(2))
    assert cache.size == first


def test_a_cached_none_plane_costs_nothing() -> None:
    from core_pdf.impl.render.target import SoftMaskCache

    cache = SoftMaskCache(budget=1_000)
    cache.store(plane_key(1), make_mask(), None)
    assert cache.size == 0
    assert cache.get(plane_key(1)) is not None

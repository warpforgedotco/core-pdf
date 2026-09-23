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


SOFT_MASK_FIXTURE = "tests/fixtures/PyMuPDF/tests/resources/test_4942.pdf"


def capture_soft_mask_parses(data: bytes) -> tuple[int, int]:
    """Extract page one, reporting (parse calls, distinct masks produced)."""
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

    tolerant_state.RecoveringTextState.resolve_soft_mask = counting
    try:
        with PdfDocument(data) as document:
            document.pages[0].extract()
    finally:
        tolerant_state.RecoveringTextState.resolve_soft_mask = original
    return calls, len(set(produced))


def test_a_repeated_soft_mask_is_parsed_once_per_distinct_state() -> None:
    # Every gs used to re-parse the ExtGState soft mask into a fresh SoftMask
    # carrying a freshly compiled transfer closure. Each cache downstream is
    # keyed by one of those identities -- the captured program by id(mask), the
    # rendered plane by id(mask.program) -- so none of them could ever hit and
    # the mask was rasterized again for every operator that referenced it.
    data = Path(SOFT_MASK_FIXTURE).read_bytes()
    calls, distinct = capture_soft_mask_parses(data)
    assert calls > distinct, "the parse cache never hit"
    assert distinct <= calls // 2


def make_state():
    from core_pdf import PdfDocument
    from core_pdf.impl.capture.interpreter import TextState

    document = PdfDocument(Path(SOFT_MASK_FIXTURE).read_bytes())
    return document, TextState(document)


class ParseCounter:
    """Stands in for parse_soft_mask so the cache can be tested on its own.

    A soft mask that really parses needs a Form XObject; what matters here is
    how often the parse is reached, not what it returns.
    """

    def __init__(self, monkeypatch) -> None:
        from core_pdf.impl.capture import tolerant_state

        self.calls = 0
        monkeypatch.setattr(tolerant_state, "parse_soft_mask", self)

    def __call__(self, value, resolver, *, ctm, compile_function):
        self.calls += 1
        return None


def test_the_same_source_under_the_same_state_parses_once(monkeypatch) -> None:
    document, state = make_state()
    try:
        parses = ParseCounter(monkeypatch)
        value = {"S": "Alpha"}
        assert state.resolve_soft_mask(value) is None
        assert state.resolve_soft_mask(value) is None
        assert state.resolve_soft_mask(value) is None
        assert parses.calls == 1
        assert len(state.parsed_soft_masks) == 1
    finally:
        document.close()


def test_a_cached_none_result_is_served_from_the_cache(monkeypatch) -> None:
    # An Identity or absent mask parses to None, which is worth caching too:
    # otherwise the commonest case re-parses on every gs.
    document, state = make_state()
    try:
        parses = ParseCounter(monkeypatch)
        for _ in range(10):
            assert state.resolve_soft_mask("None") is None
        assert parses.calls == 1
    finally:
        document.close()


def test_the_parse_cache_separates_masks_by_transform(monkeypatch) -> None:
    # The ctm is baked into the parsed mask, so the same source under a
    # different transform must not be served from the cache.
    from core_pdf_spec.s_08_graphics.matrix import Matrix

    document, state = make_state()
    try:
        parses = ParseCounter(monkeypatch)
        value = {"S": "Alpha"}
        state.resolve_soft_mask(value)
        state.graphics.ctm = Matrix(2.0, 0.0, 0.0, 2.0, 0.0, 0.0)
        state.resolve_soft_mask(value)
        assert parses.calls == 2
        assert len(state.parsed_soft_masks) == 2
    finally:
        document.close()


def test_the_parse_cache_separates_masks_by_resource_scope(monkeypatch) -> None:
    # A mask reached through different resources stays a separate object, so a
    # program captured under one scope is never reused under another.
    document, state = make_state()
    try:
        parses = ParseCounter(monkeypatch)
        value = {"S": "Alpha"}
        state.resolve_soft_mask(value)
        state.resources = dict(state.resources)
        state.resolve_soft_mask(value)
        assert parses.calls == 2
    finally:
        document.close()


def test_the_parse_cache_holds_its_keys_alive(monkeypatch) -> None:
    # The cache is keyed by id(), so the source and the resources are stored
    # with the result; without that a collected source could hand its address
    # to an unrelated object and the cache would answer for the wrong mask.
    document, state = make_state()
    try:
        ParseCounter(monkeypatch)
        state.resolve_soft_mask({"S": "Alpha"})
        key, entry = next(iter(state.parsed_soft_masks.items()))
        assert len(entry) == 3
        assert id(entry[0]) == key[0]
        assert id(entry[1]) == key[2]
    finally:
        document.close()


def test_the_parse_cache_is_bounded(monkeypatch) -> None:
    from core_pdf.impl.capture.tolerant_state import SOFT_MASK_CACHE_LIMIT

    document, state = make_state()
    try:
        ParseCounter(monkeypatch)
        sources = [{"S": "Alpha", "n": index} for index in range(SOFT_MASK_CACHE_LIMIT + 5)]
        for source in sources:
            state.resolve_soft_mask(source)
        assert len(state.parsed_soft_masks) <= SOFT_MASK_CACHE_LIMIT
    finally:
        document.close()

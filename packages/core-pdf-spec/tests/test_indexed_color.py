import pytest

from core_pdf_spec.s_08_graphics.color import indexed_color_components
from core_pdf_spec.s_08_graphics.color_spec import ColorSpace, parse_color_space


@pytest.mark.parametrize(
    ("value", "index"),
    [(-10, 0), (0, 0), (0.49, 0), (0.5, 1), (0.51, 1), (1.5, 2), (2.49, 2), (2.5, 3), (10, 3)],
)
def test_indexed_palette_rounds_half_up_and_clamps(value: float, index: int) -> None:
    palette = bytes([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255])
    spec = ColorSpace(
        "Indexed", ((0.0, float(3)),), base=parse_color_space("DeviceRGB"), hival=3, lookup=palette
    )
    expected = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    assert indexed_color_components(spec, value) == expected[index]


def test_indexed_palette_hival_limits_lookup_even_when_more_entries_exist() -> None:
    spec = ColorSpace(
        "Indexed",
        ((0.0, float(0)),),
        base=parse_color_space("DeviceGray"),
        hival=0,
        lookup=bytes([127, 255]),
    )
    assert indexed_color_components(spec, 2.5) == (127 / 255,)


@pytest.mark.parametrize(
    ("lookup", "components"), [(None, 1), (b"\0", 0), (b"\0", 3), (b"\0\0\0", 3)]
)
def test_indexed_palette_rejects_missing_or_short_entries(
    lookup: bytes | None, components: int
) -> None:
    spec = ColorSpace(
        "Indexed",
        ((0.0, float(1)),),
        base=ColorSpace("Test", ((0, 1),) * components),
        hival=1,
        lookup=lookup,
    )
    with pytest.raises(ValueError, match="invalid Indexed color lookup"):
        indexed_color_components(spec, 0.5)

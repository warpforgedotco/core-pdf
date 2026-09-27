import zlib

import pytest

from core_pdf.impl import graphics_image_samples
from core_pdf.impl.graphics_image_samples import ImageHeader
from core_pdf.impl.graphics_images import decode_pdf_image


def counted_parser(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    calls: list[object] = []
    original = graphics_image_samples.parse_color_space

    def parse(value: object, active: set[int] | None = None) -> object:
        calls.append(value)
        return original(value, active)

    monkeypatch.setattr(graphics_image_samples, "parse_color_space", parse)
    return calls


@pytest.mark.parametrize(
    ("space", "samples"),
    [("DeviceRGB", bytes(range(12))), ("DeviceGray", bytes(range(4))), ("DeviceCMYK", bytes(16))],
)
@pytest.mark.parametrize("compressed", [False, True])
def test_an_image_parses_its_color_space_at_most_once(
    monkeypatch: pytest.MonkeyPatch, space: str, samples: bytes, compressed: bool
) -> None:
    calls = counted_parser(monkeypatch)
    dictionary: dict[str, object] = {
        "Width": 2,
        "Height": 2,
        "BitsPerComponent": 8,
        "ColorSpace": space,
    }
    raw = samples
    if compressed:
        dictionary["Filter"] = "FlateDecode"
        raw = zlib.compress(samples)
    decoded = decode_pdf_image(raw, dictionary)
    assert decoded is not None
    assert calls in ([], [space])


def test_a_color_space_error_is_raised_again_without_reparsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = counted_parser(monkeypatch)
    header = ImageHeader({"ColorSpace": ["Indexed", "DeviceRGB", -1, b""]})
    for ignored in range(2):
        with pytest.raises(ValueError, match="hival"):
            header.space()
    assert len(calls) == 1


def test_the_header_reads_dimensions_and_depth_leniently() -> None:
    header = ImageHeader({"Width": "3", "Height": 2.0, "Mask": [0, 1]})
    assert (header.width, header.height, header.bits) == (3, 0, 8)
    assert header.has_color_key_mask
    assert header.filter is None

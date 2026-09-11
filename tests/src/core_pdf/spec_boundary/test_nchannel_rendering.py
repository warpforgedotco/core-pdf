# SPDX-License-Identifier: AGPL-3.0-only
"""NChannel process values reach pixels (ISO 32000-2, 8.6.6.5, Tables 70-71)."""

from dataclasses import dataclass
from typing import Any

import imagecodecs
import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.graphics.device_profiles import default_cmyk_transform


@dataclass(frozen=True)
class internal_ProcessCase:
    name: str
    space: bytes
    colorants: tuple[str, ...]
    components: tuple[str, ...]
    values: tuple[float, ...]
    mapped: tuple[float, ...]


CASES = (
    internal_ProcessCase(
        "rgb-natural",
        b"/DeviceRGB",
        ("Red", "Green", "Blue"),
        ("Red", "Green", "Blue"),
        (0.2, 0.4, 0.8),
        (0.2, 0.4, 0.8),
    ),
    internal_ProcessCase(
        "rgb-custom-names",
        b"/DeviceRGB",
        ("A", "B", "C"),
        ("A", "B", "C"),
        (1, 0, 0),
        (1, 0, 0),
    ),
    internal_ProcessCase(
        "cmyk-reordered",
        b"/DeviceCMYK",
        ("Black", "Yellow", "Cyan", "Magenta"),
        ("Cyan", "Magenta", "Yellow", "Black"),
        (0.2, 0.8, 0.4, 0),
        (0.4, 0, 0.8, 0.2),
    ),
    internal_ProcessCase(
        "cmyk-subset",
        b"/DeviceCMYK",
        ("Yellow", "Cyan"),
        ("Cyan", "Magenta", "Yellow", "Black"),
        (1, 1),
        (1, 0, 1, 0),
    ),
    internal_ProcessCase(
        "icc-cmyk-subset",
        b"[/ICCBased 7 0 R]",
        ("Black",),
        ("Cyan", "Magenta", "Yellow", "Black"),
        (1,),
        (0, 0, 0, 1),
    ),
)
PAINTS = (
    "path",
    "image8",
    "image16",
    "colored-pattern",
    "uncolored-pattern",
    "indexed-path",
    "indexed-image",
    "shading",
)


def internal_stream(raw: bytes, entries: bytes = b"") -> bytes:
    return b"<< " + entries + f" /Length {len(raw)} >>\nstream\n".encode() + raw + b"\nendstream"


def internal_names(values: tuple[str, ...]) -> bytes:
    return b"[" + b" ".join(b"/" + value.encode() for value in values) + b"]"


def internal_numbers(values: tuple[float, ...]) -> bytes:
    return b" ".join(f"{value:.12g}".encode() for value in values)


def internal_nchannel(
    case: internal_ProcessCase, subtype: str = "NChannel", *, spot: bool = False
) -> bytes:
    names = (*case.colorants, "Spot") if spot else case.colorants
    return (
        b"[/DeviceN "
        + internal_names(names)
        + b" /DeviceRGB 8 0 R << /Subtype /"
        + subtype.encode()
        + b" /Process << /ColorSpace "
        + case.space
        + b" /Components "
        + internal_names(case.components)
        + b" >> "
        + (b"/Colorants << /Spot [/Separation /Spot /DeviceRGB 9 0 R] >> " if spot else b"")
        + b">>]"
    )


def internal_document(
    space: bytes,
    values: tuple[float, ...],
    paint: str = "path",
    *,
    repeated: bool = False,
    initial: bool = False,
    image_intent: bool = False,
) -> bytes:
    """Use a deliberately blue global tint, regardless of the supplied values."""
    original_count = len(values)
    if paint.startswith("indexed-"):
        lookup = bytes(round(value * 255) for value in values).hex().encode()
        space = b"[/Indexed " + space + b" 0 <" + lookup + b">]"
        values = (0,)
    operands = internal_numbers(values)
    resource = b""
    extra = b"<< >>"
    operation = b"/C cs " + (b"" if initial else operands + b" scn ") + b"0 0 10 10 re f"
    if paint in {"image8", "image16", "indexed-image"}:
        depth = 16 if paint == "image16" else 8
        maximum = (1 << depth) - 1
        raw = b"".join(round(value * maximum).to_bytes(depth // 8, "big") for value in values)
        extra = internal_stream(
            raw,
            f"/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent {depth} ".encode()
            + b"/ColorSpace 5 0 R "
            + (b"/Intent /Perceptual" if image_intent else b""),
        )
        resource = b"/XObject << /Im 6 0 R >>"
        operation = b"10 0 0 10 0 0 cm /Im Do"
    elif paint in {"colored-pattern", "uncolored-pattern"}:
        colored = paint == "colored-pattern"
        extra = internal_stream(
            (b"/C cs " + operands + b" scn " if colored else b"") + b"0 0 10 10 re f",
            b"/Type /Pattern /PatternType 1 "
            + (b"/PaintType 1 " if colored else b"/PaintType 2 ")
            + b"/TilingType 1 /BBox [0 0 10 10] /XStep 10 /YStep 10 "
            b"/Resources << /ColorSpace << /C 5 0 R >> >>",
        )
        resource = b"/Pattern << /P 6 0 R >>"
        operation = (
            b"/Pattern cs /P scn " if colored else b"/U cs " + operands + b" /P scn "
        ) + b"0 0 10 10 re f"
    elif paint == "shading":
        extra = (
            b"<< /ShadingType 2 /ColorSpace 5 0 R /Coords [0 0 10 0] /Extend [true true] "
            b"/Function << /FunctionType 2 /Domain [0 1] /C0 ["
            + operands
            + b"] /C1 ["
            + operands
            + b"] /N 1 >> >>"
        )
        resource = b"/Shading << /S 6 0 R >>"
        operation = b"0 0 10 10 re W n /S sh"
    if repeated:
        content = b" ".join(
            f"q /{setting} gs 1 0 0 1 {index * 10} 0 cm ".encode() + operation + b" Q"
            for index, setting in enumerate(("Off", "On", "Off", "Abs"))
        )
    else:
        content = (b"1 0 0 rg 0 0 40 10 re f " if initial else b"") + operation
    profile = default_cmyk_transform()
    assert profile is not None
    bodies = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 40 10] /Resources << "
        b"/ColorSpace << /C 5 0 R /U [/Pattern 5 0 R] >> "
        b"/ExtGState << /Off << /UseBlackPtComp /OFF >> /On << /UseBlackPtComp /ON >> "
        b"/Abs << /RI /AbsoluteColorimetric /UseBlackPtComp /ON >> >> "
        + resource
        + b" >> /Contents 4 0 R >>",
        internal_stream(content),
        space,
        extra,
        internal_stream(profile.profile, b"/N 4 /Alternate /DeviceCMYK"),
        internal_stream(
            b"\x00\x00\xff",
            b"/FunctionType 0 /Domain ["
            + b"0 1 " * original_count
            + b"] /Size ["
            + b"1 " * original_count
            + b"] /BitsPerSample 8 /Encode ["
            + b"0 0 " * original_count
            + b"] /Range [0 1 0 1 0 1]",
        ),
        b"<< /FunctionType 2 /Domain [0 1] /C0 [1 1 0] /C1 [1 1 0] /N 1 >>",
    ]
    result = b"%PDF-2.0\n"
    offsets = [0]
    for number, body in enumerate(bodies, 1):
        offsets.append(len(result))
        result += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(result)
    result += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    result += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    return (
        result
        + f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )


def internal_pixels(data: bytes) -> tuple[tuple[int, int, int], ...]:
    with PdfDocument(data) as document:
        pixels = document.pages[0].render().rasterize().array()
        return tuple(
            (int(pixels[5, x, 0]), int(pixels[5, x, 1]), int(pixels[5, x, 2]))
            for x in (5, 15, 25, 35)
        )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("paint", PAINTS)
def test_process_only_nchannel_matches_direct_process_paint(
    case: internal_ProcessCase, paint: str
) -> None:
    actual = internal_pixels(internal_document(internal_nchannel(case), case.values, paint))[0]
    expected = internal_pixels(internal_document(case.space, case.mapped, paint))[0]
    assert expected != (0, 0, 255)
    assert actual == expected


def test_initial_nchannel_rgb_components_are_one_and_not_process_defaults() -> None:
    case = CASES[0]
    nchannel = internal_pixels(
        internal_document(internal_nchannel(case), case.values, initial=True)
    )
    direct = internal_pixels(internal_document(case.space, case.mapped, initial=True))
    assert nchannel[0] == (255, 255, 255)
    assert direct[0] == (0, 0, 0)
    assert nchannel[1] == direct[1] == (255, 0, 0)


@pytest.mark.parametrize("paint", ["path", "image8", "image16", "indexed-path", "shading"])
@pytest.mark.parametrize("mixed_spot", [False, True], ids=["ordinary-devicen", "mixed-nchannel"])
def test_global_tint_fallback_remains_for_ordinary_devicen_and_mixed_spots(
    paint: str, mixed_spot: bool
) -> None:
    case = CASES[0]
    space = internal_nchannel(case, "NChannel" if mixed_spot else "DeviceN", spot=mixed_spot)
    values = (*case.values, 0.5) if mixed_spot else case.values
    expected = (0, 0, 255)
    if paint == "indexed-path":
        # Legacy scalar Indexed/DeviceN does not evaluate the global tint.
        # Process-only support must preserve this existing fallback behavior.
        names = (*case.colorants, "Spot") if mixed_spot else case.colorants
        baseline = b"[/DeviceN " + internal_names(names) + b" /DeviceRGB 8 0 R]"
        expected = internal_pixels(internal_document(baseline, values, paint))[0]
    assert internal_pixels(internal_document(space, values, paint))[0] == expected


@pytest.mark.parametrize(
    "paint", ["path", "image8", "image16", "colored-pattern", "uncolored-pattern", "shading"]
)
def test_nchannel_icc_process_inherits_intent_bpc_and_graphics_state(
    paint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = CASES[-1]
    expected = internal_pixels(internal_document(case.space, case.mapped, paint, repeated=True))
    calls: list[tuple[int, int]] = []
    original = imagecodecs.cms_transform

    def record(*args: Any, **kwargs: Any) -> Any:
        calls.append((kwargs["intent"], kwargs["flags"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(imagecodecs, "cms_transform", record)
    actual = internal_pixels(
        internal_document(internal_nchannel(case), case.values, paint, repeated=True)
    )
    assert actual == expected
    assert actual[0] != actual[1]
    assert {(1, 0x0100), (1, 0x2100), (3, 0x0100)} <= set(calls)


@pytest.mark.parametrize("depth", [8, 16])
def test_nchannel_image_intent_overrides_inherited_intent(
    depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = CASES[-1]
    calls: list[tuple[int, int]] = []
    original = imagecodecs.cms_transform

    def record(*args: Any, **kwargs: Any) -> Any:
        calls.append((kwargs["intent"], kwargs["flags"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(imagecodecs, "cms_transform", record)
    actual = internal_pixels(
        internal_document(
            internal_nchannel(case), case.values, f"image{depth}", repeated=True, image_intent=True
        )
    )
    assert actual[0] == actual[2]
    assert actual[1] == actual[3]
    assert {(0, 0x0100), (0, 0x2100)} <= set(calls)
    assert all(intent == 0 for intent, _flags in calls)


def test_nchannel_16bit_image_keeps_process_precision_until_icc_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CASES[-1]
    value = 16385 / 65535  # Deliberately not representable by an 8-bit sample.
    calls: list[tuple[str, tuple[int, ...]]] = []
    original = imagecodecs.cms_transform

    def record(samples: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("colorspace") == "cmyk":
            for row in samples.reshape(-1, 4):
                calls.append((str(samples.dtype), tuple(int(component) for component in row)))
        return original(samples, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(imagecodecs, "cms_transform", record)
        actual = internal_pixels(internal_document(internal_nchannel(case), (value,), "image16"))[0]
    assert ("uint16", (0, 0, 0, 16385)) in calls
    expected = internal_pixels(internal_document(case.space, (0, 0, 0, value), "image16"))[0]
    assert actual == expected

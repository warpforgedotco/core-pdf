# SPDX-License-Identifier: AGPL-3.0-only

"""Check how device colour and nested groups find a transparency group's colour space.

ISO 32000-2 11.7.2: "When converting colours, if the colour space of any graphics object
is a device colour space, and the current group or an ancestor of the current group is
defined with a CIE-based colour space with the same number of colourants, then, for
compositing purposes only, the colour space of the graphics object shall be the CIE-based
space of the nearest such ancestor." So DeviceRGB painted into a group whose CS is an
ICCBased RGB or CalRGB space is a colour of that space, and DeviceGray in a CalGray or
ICCBased gray group one of that; DeviceCMYK in an ICCBased CMYK group is the group's own
CMYK. A default space the painting stream's resources name still comes first (8.6.5.6),
and a colour set before the group and inherited into it counts as well. The same clause:
"If the colour space of the transparency group is a device colour space, and some ancestor
of the group has a CIE-based colour space with the same number of colourants, then the
colour space of this group shall be the CIE-based space of the nearest such ancestor", the
search Annex P draws, which starts from a soft mask's group afresh: that group "has no
parent group or page from which to inherit" (11.6.6, Table 145). Lab cannot be a blending
space (11.3.4), and a non-isolated group's CS is not its own (11.6.6), so neither changes
anything.

The pages use matrix-shaper ICC RGB and gray profiles with linear curves, CalRGB and CalGray
spaces with gammas far from sRGB's, and "linear naive" ICC CMYK profiles of two ink
strengths, all built here; each page's expected sRGB is worked out from the spaces'
definitions, in plain Python. A Luminosity mask's value is read through a black fill it
masks over white, 255 (1 - mask). Each sampled pixel must be within its case's tolerance.

With --mupdf (and PyMuPDF installed: uv run --with pymupdf), each sample is also rendered by
MuPDF and whether it agrees is printed; that is reported, not checked.

uv run --all-packages python scripts/check_group_spaces.py [--mupdf]
"""

from __future__ import annotations

import itertools
import struct
import sys
from collections.abc import Sequence
from typing import NamedTuple

import numpy

from core_pdf import PdfDocument

SIZE = 200
# The Bradford-adapted sRGB primaries under D50: the XYZ of linear red, green and blue.
SRGB_D50 = (
    (0.4360747, 0.2225045, 0.0139322),
    (0.3850649, 0.7168786, 0.0971045),
    (0.1430804, 0.0606169, 0.7141733),
)
D50 = (0.9642, 1.0, 0.8249)
CALRGB_MATRIX = b" ".join(b"%.7f" % value for primary in SRGB_D50 for value in primary)

# Where each page paints, as (x0, y0, x1, y1).
FIRST = (20, 20, 90, 170)
SECOND = (110, 20, 180, 100)


def encode(linear: float) -> float:
    """The sRGB encoding of a linear component, 0 to 1."""
    linear = min(1.0, max(0.0, linear))
    return 12.92 * linear if linear <= 0.0031308 else 1.055 * linear ** (1 / 2.4) - 0.055


def luminance(red: float, green: float, blue: float) -> float:
    """The Y under D50 of a linear colour of sRGB's primaries."""
    return SRGB_D50[0][1] * red + SRGB_D50[1][1] * green + SRGB_D50[2][1] * blue


def s15_fixed16(value: float) -> bytes:
    return struct.pack(">i", round(value * 65536))


def xyz_tag(xyz: Sequence[float]) -> bytes:
    return b"XYZ \0\0\0\0" + b"".join(s15_fixed16(value) for value in xyz)


def curve(gamma: float) -> bytes:
    return b"curv\0\0\0\0" + struct.pack(">IH", 1, round(gamma * 256))


def lut16(inputs: int, outputs: int, grid: int, clut: Sequence[int]) -> bytes:
    """An ICC lut16Type with identity matrix and curves and the given grid."""
    identity = struct.pack(">HH", 0, 65535)
    return (
        b"mft2\0\0\0\0"
        + bytes((inputs, outputs, grid, 0))
        + b"".join(s15_fixed16(value) for value in (1, 0, 0, 0, 1, 0, 0, 0, 1))
        + struct.pack(">HH", 2, 2)
        + identity * inputs
        + struct.pack(">%dH" % len(clut), *clut)
        + identity * outputs
    )


def profile(device_class: bytes, space: bytes, tags: list[tuple[bytes, bytes]]) -> bytes:
    """An ICC version 2 profile with an XYZ PCS, a D50 white and the given tags."""
    text = b"check_group_spaces\0"
    tags = [
        (b"desc", b"desc\0\0\0\0" + struct.pack(">I", len(text)) + text + b"\0" * 79),
        (b"wtpt", xyz_tag(D50)),
        (b"cprt", b"text\0\0\0\0none\0"),
        *tags,
    ]
    start = 128 + 4 + 12 * len(tags)
    table = struct.pack(">I", len(tags))
    body = b""
    for signature, data in tags:
        body += b"\0" * (-(start + len(body)) % 4)
        table += signature + struct.pack(">II", start + len(body), len(data))
        body += data
    header = (
        struct.pack(">I", start + len(body))
        + b"lcms"
        + bytes((2, 0x10, 0, 0))
        + device_class
        + space
        + b"XYZ "
        + b"\0" * 12
        + b"acsp"
        + b"\0" * 24
        + struct.pack(">I", 0)
        + b"".join(s15_fixed16(value) for value in D50)
        + b"\0" * 48
    )
    return header + table + body


def linear_rgb_profile() -> bytes:
    """A matrix-shaper RGB profile: sRGB's primaries under D50, with linear curves."""
    return profile(
        b"mntr",
        b"RGB ",
        [
            (b"rXYZ", xyz_tag(SRGB_D50[0])),
            (b"gXYZ", xyz_tag(SRGB_D50[1])),
            (b"bXYZ", xyz_tag(SRGB_D50[2])),
            (b"rTRC", curve(1.0)),
            (b"gTRC", curve(1.0)),
            (b"bTRC", curve(1.0)),
        ],
    )


def linear_gray_profile() -> bytes:
    """A gray profile whose Y is its component."""
    return profile(b"mntr", b"GRAY", [(b"kTRC", curve(1.0))])


def naive_linear(cmyk: Sequence[float], strength: float) -> tuple[float, float, float]:
    """The linear sRGB of a colour of the linear naive CMYK profile of an ink strength."""
    c, m, y, k = cmyk
    return (
        (1 - strength * c) * (1 - k),
        (1 - strength * m) * (1 - k),
        (1 - strength * y) * (1 - k),
    )


def pcs_word(value: float) -> int:
    """An XYZ PCS value in a version 2 lut16: 1.0 is 0x8000."""
    return max(0, min(65535, round(value * 32768)))


def naive_cmyk_profile(strength: float) -> bytes:
    """A CMYK output profile: its A2B0 the linear naive conversion of an ink strength, and
    its B2A0 an inverse, K = 1 - max(R, G, B) and C = (1 - R / (1 - K)) / strength and so
    on."""
    forward: list[int] = []
    for cmyk in itertools.product([step / 16 for step in range(17)], repeat=4):
        linear = naive_linear(cmyk, strength)
        forward.extend(
            pcs_word(sum(SRGB_D50[i][axis] * linear[i] for i in range(3))) for axis in range(3)
        )
    inverse = numpy.linalg.inv(numpy.array(SRGB_D50).T)
    grid = 33
    steps = numpy.arange(grid) / (grid - 1) * (65535 / 32768)
    backward: list[int] = []
    for xyz in itertools.product(steps, repeat=3):
        linear = numpy.clip(inverse @ numpy.array(xyz), 0.0, 1.0)
        brightest = float(linear.max())
        if brightest <= 0.0:
            backward.extend((0, 0, 0, 65535))
            continue
        inks = numpy.clip((1.0 - linear / brightest) / strength, 0.0, 1.0)
        backward.extend(round(65535 * value) for value in (*inks, 1.0 - brightest))
    return profile(
        b"prtr",
        b"CMYK",
        [(b"A2B0", lut16(4, 3, 17, forward)), (b"B2A0", lut16(3, 4, grid, backward))],
    )


def stream(dictionary: bytes, data: bytes) -> bytes:
    return b"<< " + dictionary + b" /Length %d >>\nstream\n" % len(data) + data + b"\nendstream"


def document(page: bytes, content: bytes, extra: Sequence[bytes]) -> bytes:
    """A one-page PDF: the page dictionary's entries, its content, and objects 5 onwards."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] " % (SIZE, SIZE) + page + b" >>",
        stream(b"", content),
        *extra,
    ]
    out = bytearray(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def numbers(values: Sequence[float]) -> bytes:
    return b" ".join(b"%.4f" % value for value in values)


def rectangle(box: tuple[int, int, int, int]) -> bytes:
    x0, y0, x1, y1 = box
    return b"%d %d %d %d re f" % (x0, y0, x1 - x0, y1 - y0)


def calrgb(gamma: float) -> bytes:
    return b"[/CalRGB << /WhitePoint [0.9642 1 0.8249] /Gamma [%g %g %g] /Matrix [%s] >>]" % (
        gamma,
        gamma,
        gamma,
        CALRGB_MATRIX,
    )


def calgray(gamma: float) -> bytes:
    return b"[/CalGray << /WhitePoint [0.9642 1 0.8249] /Gamma %g >>]" % gamma


class Sample(NamedTuple):
    box: tuple[int, int, int, int]
    rgb: tuple[float, float, float]
    what: str


class Case(NamedTuple):
    name: str
    pdf: bytes
    samples: tuple[Sample, ...]
    tolerance: float


def srgb(linear: Sequence[float]) -> tuple[float, float, float]:
    """The sRGB bytes of a linear colour of sRGB's primaries."""
    return 255 * encode(linear[0]), 255 * encode(linear[1]), 255 * encode(linear[2])


def neutral(linear: float) -> tuple[float, float, float]:
    return srgb((linear, linear, linear))


def device(rgb: Sequence[float]) -> tuple[float, float, float]:
    return 255 * rgb[0], 255 * rgb[1], 255 * rgb[2]


def masked(mask: float) -> tuple[float, float, float]:
    """The colour of a black fill painted over white through a mask value."""
    level = 255 * (1 - mask)
    return (level, level, level)


def device_luminosity(rgb: Sequence[float]) -> float:
    return 0.3 * rgb[0] + 0.59 * rgb[1] + 0.11 * rgb[2]


# Objects 5 to 8 of every page: the linear RGB and gray profiles, and naive CMYK profiles of
# ink strengths 1 (A) and 0.8 (B).
PROFILES = (
    stream(b"/N 3", linear_rgb_profile()),
    stream(b"/N 1", linear_gray_profile()),
    stream(b"/N 4", naive_cmyk_profile(1.0)),
    stream(b"/N 4", naive_cmyk_profile(0.8)),
)
LINEAR_RGB = b"[/ICCBased 5 0 R]"
LINEAR_GRAY = b"[/ICCBased 6 0 R]"
CMYK_A = b"[/ICCBased 7 0 R]"
CMYK_B = b"[/ICCBased 8 0 R]"
FIRST_RGB, SECOND_RGB = (0.5, 0.2, 0.8), (0.9, 0.6, 0.1)
# Light colours, which the built-in DeviceCMYK profile's gamut holds.
FIRST_CMYK, SECOND_CMYK = (0.3, 0.2, 0.4, 0.1), (0.1, 0.35, 0.2, 0.05)


def form(group: bytes, resources: bytes, body: bytes) -> bytes:
    return stream(
        b"/Type /XObject /Subtype /Form /BBox [0 0 200 200] /Group << /S /Transparency %s >> "
        b"/Resources << %s >>" % (group, resources),
        body,
    )


def two_fills(paint: tuple[bytes, bytes]) -> bytes:
    return paint[0] + b" " + rectangle(FIRST) + b" " + paint[1] + b" " + rectangle(SECOND)


def group_case(
    name: str,
    group: bytes,
    paint: tuple[bytes, bytes],
    expected: tuple[tuple[float, float, float], tuple[float, float, float]],
    *,
    resources: bytes = b"",
    body: bytes | None = None,
    page_content: bytes = b"/G Do",
    page_group: bytes = b"",
    extra: Sequence[bytes] = (),
    tolerance: float = 3,
) -> Case:
    """A page invoking a group XObject /G, whose group attributes are group, painting FIRST
    and SECOND with paint's operators (or else body), whose sRGB bytes are expected. Further
    objects, extra, are 10 onwards; /G is 9."""
    content = two_fills(paint) if body is None else body
    pdf = document(
        b"%s/Resources << /XObject << /G 9 0 R >> >> /Contents 4 0 R" % page_group,
        page_content,
        [*PROFILES, form(group, resources, content), *extra],
    )
    return Case(
        name,
        pdf,
        (Sample(FIRST, expected[0], "first colour"), Sample(SECOND, expected[1], "second colour")),
        tolerance,
    )


def mask_case(
    name: str,
    space: bytes,
    paint: tuple[bytes, bytes],
    expected: tuple[float, float],
    *,
    resources: bytes = b"",
    page_group: bytes = b"",
    extra: Sequence[bytes] = (),
    tolerance: float = 3,
) -> Case:
    """A black page masked by a Luminosity mask whose group's CS is space, painting FIRST
    and SECOND with paint's operators, whose luminosities are expected. The mask's group is
    9, and further objects, extra, 10 onwards."""
    page = (
        b"%s/Resources << /ExtGState << /M << /SMask << /S /Luminosity /G 9 0 R >> >> >> >> "
        b"/Contents 4 0 R" % page_group
    )
    pdf = document(
        page,
        b"/M gs 0 g 0 0 200 200 re f",
        [*PROFILES, form(b"/CS %s" % space, resources, two_fills(paint)), *extra],
    )
    return Case(
        name,
        pdf,
        (
            Sample(FIRST, masked(expected[0]), "first colour"),
            Sample(SECOND, masked(expected[1]), "second colour"),
        ),
        tolerance,
    )


def rgb_paint(first: Sequence[float], second: Sequence[float]) -> tuple[bytes, bytes]:
    return b"%s rg" % numbers(first), b"%s rg" % numbers(second)


def device_color_cases() -> list[Case]:
    """11.7.2: device colour painted into a group whose space is CIE-based, with as many
    components, is a colour of that space."""
    linear_isolated = b"/I true /CS %s" % LINEAR_RGB
    lookup = bytes(round(255 * value) for value in (*FIRST_RGB, *SECOND_RGB))
    image = stream(
        b"/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceRGB "
        b"/BitsPerComponent 8",
        lookup,
    )
    image_case = group_case(
        "a DeviceRGB image in an ICCBased RGB group",
        linear_isolated,
        (b"", b""),
        (srgb(FIRST_RGB), srgb(FIRST_RGB)),
        resources=b"/XObject << /I 10 0 R >>",
        body=b"q 160 0 0 150 20 20 cm /I Do Q",
        extra=(image,),
    )
    image_case = image_case._replace(
        samples=(
            Sample((20, 20, 96, 170), srgb(FIRST_RGB), "left"),
            Sample((104, 20, 180, 170), srgb(SECOND_RGB), "right"),
        )
    )
    shading = (
        b"<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 200 0] /Function "
        b"<< /FunctionType 2 /Domain [0 1] /C0 [%s] /C1 [%s] /N 1 >> >>"
        % (numbers(FIRST_RGB), numbers(FIRST_RGB))
    )
    return [
        group_case(
            "DeviceRGB in an ICCBased RGB group is the group's RGB",
            linear_isolated,
            rgb_paint(FIRST_RGB, SECOND_RGB),
            (srgb(FIRST_RGB), srgb(SECOND_RGB)),
        ),
        group_case(
            "DeviceRGB in a CalRGB group is the group's RGB",
            b"/I true /CS %s" % calrgb(1.8),
            rgb_paint(FIRST_RGB, SECOND_RGB),
            (srgb([value**1.8 for value in FIRST_RGB]), srgb([value**1.8 for value in SECOND_RGB])),
        ),
        group_case(
            "DeviceGray in an ICCBased gray group is the group's gray",
            b"/I true /CS %s" % LINEAR_GRAY,
            (b"0.5 g", b"0.25 g"),
            (neutral(0.5), neutral(0.25)),
        ),
        group_case(
            "DeviceGray in a CalGray group is the group's gray",
            b"/I true /CS %s" % calgray(1.8),
            (b"0.5 g", b"0.25 g"),
            (neutral(0.5**1.8), neutral(0.25**1.8)),
        ),
        image_case,
        group_case(
            "a DeviceRGB shading in an ICCBased RGB group",
            linear_isolated,
            (b"", b""),
            (srgb(FIRST_RGB), srgb(FIRST_RGB)),
            resources=b"/Shading << /S %s >>" % shading,
            body=b"/S sh",
        ),
        group_case(
            "an Indexed space over DeviceRGB in an ICCBased RGB group",
            linear_isolated,
            (b"/X cs 0 sc", b"/X cs 1 sc"),
            (srgb(FIRST_RGB), srgb(SECOND_RGB)),
            resources=b"/ColorSpace << /X [/Indexed /DeviceRGB 1 <%s>] >>" % lookup.hex().encode(),
        ),
        group_case(
            "a DeviceRGB colour set before the group and inherited into it",
            linear_isolated,
            (b"", b""),
            (srgb(FIRST_RGB), srgb(SECOND_RGB)),
            body=rectangle(FIRST) + b" %s rg " % numbers(SECOND_RGB) + rectangle(SECOND),
            page_content=b"%s rg /G Do" % numbers(FIRST_RGB),
        ),
        group_case(
            "a DefaultRGB of the group's resources comes before the group's space",
            linear_isolated,
            rgb_paint(FIRST_RGB, SECOND_RGB),
            (srgb([value**1.8 for value in FIRST_RGB]), srgb([value**1.8 for value in SECOND_RGB])),
            resources=b"/ColorSpace << /DefaultRGB %s >>" % calrgb(1.8),
        ),
        group_case(
            "DeviceCMYK in an ICCBased CMYK group is the group's CMYK",
            b"/I true /CS %s" % CMYK_B,
            (b"%s k" % numbers(FIRST_CMYK), b"%s k" % numbers(SECOND_CMYK)),
            (srgb(naive_linear(FIRST_CMYK, 0.8)), srgb(naive_linear(SECOND_CMYK, 0.8))),
        ),
        group_case(
            "a Lab group cannot blend: DeviceRGB stays as it is",
            b"/I true /CS [/Lab << /WhitePoint [0.9642 1 0.8249] >>]",
            rgb_paint(FIRST_RGB, SECOND_RGB),
            (device(FIRST_RGB), device(SECOND_RGB)),
        ),
        group_case(
            "a non-isolated group's CS is not its own: DeviceRGB stays as it is",
            b"/CS %s" % LINEAR_RGB,
            rgb_paint(FIRST_RGB, SECOND_RGB),
            (device(FIRST_RGB), device(SECOND_RGB)),
        ),
        mask_case(
            "DeviceRGB in an ICCBased RGB Luminosity mask is the group's RGB",
            LINEAR_RGB,
            rgb_paint(FIRST_RGB, SECOND_RGB),
            (luminance(*FIRST_RGB), luminance(*SECOND_RGB)),
        ),
        mask_case(
            "DeviceRGB in a CalRGB Luminosity mask is the group's RGB",
            calrgb(2.2),
            rgb_paint(FIRST_RGB, SECOND_RGB),
            (
                luminance(*(value**2.2 for value in FIRST_RGB)),
                luminance(*(value**2.2 for value in SECOND_RGB)),
            ),
        ),
        mask_case(
            "DeviceGray in an ICCBased gray Luminosity mask is the group's gray",
            LINEAR_GRAY,
            (b"0.5 g", b"0.25 g"),
            (0.5, 0.25),
        ),
        mask_case(
            "DeviceCMYK in an ICCBased CMYK Luminosity mask is the group's CMYK",
            CMYK_A,
            (b"%s k" % numbers(FIRST_CMYK), b"%s k" % numbers(SECOND_CMYK)),
            (luminance(*naive_linear(FIRST_CMYK, 1.0)), luminance(*naive_linear(SECOND_CMYK, 1.0))),
        ),
    ]


def ancestor_cases() -> list[Case]:
    """11.7.2 and Annex P: a group whose CS is a device space, with no default space for
    it, blends in the CIE-based space of the nearest ancestor group with as many
    components; a soft mask's group has no ancestors."""
    inner_cmyk = form(
        b"/I true /CS /DeviceCMYK",
        b"/ColorSpace << /A %s >>" % CMYK_A,
        b"%s k " % numbers(FIRST_CMYK)
        + rectangle(FIRST)
        + b" /A cs %s sc " % numbers(SECOND_CMYK)
        + rectangle(SECOND),
    )
    cmyk_expected = (srgb(naive_linear(FIRST_CMYK, 0.8)), srgb(naive_linear(SECOND_CMYK, 1.0)))
    inner_gray = form(
        b"/I true /CS /DeviceGray",
        b"",
        b"1 0 0 rg " + rectangle(FIRST) + b" 0 0 1 rg " + rectangle(SECOND),
    )
    return [
        group_case(
            "a DeviceCMYK group in an ICCBased CMYK group blends in the ancestor's CMYK",
            b"/I true /CS %s" % CMYK_B,
            (b"", b""),
            cmyk_expected,
            resources=b"/XObject << /F 10 0 R >>",
            body=b"/F Do",
            extra=(inner_cmyk,),
            tolerance=6,
        ),
        group_case(
            "a DeviceCMYK group in a non-isolated group in an ICCBased CMYK page group",
            b"",
            (b"", b""),
            cmyk_expected,
            resources=b"/XObject << /F 10 0 R >>",
            body=b"/F Do",
            page_group=b"/Group << /S /Transparency /CS %s >> " % CMYK_B,
            extra=(inner_cmyk,),
            tolerance=6,
        ),
        group_case(
            "a non-isolated group in an ICCBased RGB group paints in its RGB",
            b"/I true /CS %s" % LINEAR_RGB,
            (b"", b""),
            (srgb(FIRST_RGB), srgb(SECOND_RGB)),
            resources=b"/XObject << /F 10 0 R >>",
            body=b"/F Do",
            extra=(form(b"", b"", two_fills(rgb_paint(FIRST_RGB, SECOND_RGB))),),
        ),
        mask_case(
            "a DeviceGray group in a CalGray Luminosity mask blends in CalGray",
            calgray(1.0),
            (b"/F Do q 0 0 0 0 re W n", b"Q 0 0 1 rg 0 0 0 0 re W n"),
            (luminance(1, 0, 0), luminance(0, 0, 1)),
            resources=b"/XObject << /F 10 0 R >>",
            extra=(inner_gray,),
        ),
        mask_case(
            "DeviceGray in a CalGray group in a CalRGB Luminosity mask is CalGray's gray",
            calrgb(1.0),
            (b"/F Do q 0 0 0 0 re W n", b"Q 0 0 0 0 re W n"),
            (0.5**1.8, 0.25**1.8),
            resources=b"/XObject << /F 10 0 R >>",
            extra=(
                form(
                    b"/I true /CS %s" % calgray(1.8),
                    b"",
                    b"0.5 g " + rectangle(FIRST) + b" 0.25 g " + rectangle(SECOND),
                ),
            ),
        ),
        mask_case(
            "a DeviceRGB Luminosity mask does not inherit the page group's CalRGB",
            b"/DeviceRGB",
            rgb_paint(FIRST_RGB, SECOND_RGB),
            (device_luminosity(FIRST_RGB), device_luminosity(SECOND_RGB)),
            page_group=b"/Group << /S /Transparency /CS %s >> " % calrgb(1.0),
        ),
    ]


def render(data: bytes) -> numpy.ndarray:
    """The page's pixels over white, as RGB."""
    with PdfDocument(data) as pdf:
        image = pdf.pages[0].render().rasterize()
    pixels = numpy.frombuffer(bytes(image.pixels), numpy.uint8).reshape(
        image.height, image.width, 4
    )
    rgb = pixels[..., :3].astype(numpy.float64)
    alpha = pixels[..., 3:].astype(numpy.float64) / 255.0
    return rgb * alpha + 255.0 * (1.0 - alpha)


def render_mupdf(data: bytes) -> numpy.ndarray:
    import pymupdf  # noqa: PLC0415  # ty: ignore[unresolved-import]

    document = pymupdf.open("pdf", data)
    pixmap = document[0].get_pixmap(alpha=False)
    return (
        numpy.frombuffer(pixmap.samples, numpy.uint8)
        .reshape(pixmap.height, pixmap.width, pixmap.n)[..., :3]
        .astype(numpy.float64)
    )


def sample_failures(case: Case, pixels: numpy.ndarray) -> list[str]:
    failures = []
    for sample in case.samples:
        x0, y0, x1, y1 = sample.box
        # The box's pixels, two pixels in from each edge; rows run down from the top.
        region = pixels[SIZE - y1 + 2 : SIZE - y0 - 2, x0 + 2 : x1 - 2].reshape(-1, 3)
        error = numpy.abs(region - numpy.array(sample.rgb)).max(axis=1)
        wrong = int((error > case.tolerance).sum())
        if wrong:
            found = tuple(round(float(value)) for value in region[int(error.argmax())])
            shown = tuple(round(value) for value in sample.rgb)
            failures.append(
                f"{case.name}, {sample.what}: {wrong} of {len(region)} pixels off, "
                f"as {found} != {shown}"
            )
    return failures


def main(argv: list[str]) -> int:
    cases = [*device_color_cases(), *ancestor_cases()]
    failures = [failure for case in cases for failure in sample_failures(case, render(case.pdf))]
    for failure in failures:
        print(failure)
    if "--mupdf" in argv:
        for case in cases:
            disagreements = sample_failures(case, render_mupdf(case.pdf))
            print(f"  MuPDF {'disagrees' if disagreements else 'agrees'}: {case.name}")
            for disagreement in disagreements:
                print(f"    {disagreement}")
    print(f"{len(cases)} group colour space pages: {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

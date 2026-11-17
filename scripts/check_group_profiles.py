# SPDX-License-Identifier: AGPL-3.0-only

"""Check colour profiles in Luminosity soft masks and in CMYK transparency groups.

A Luminosity soft mask takes the luminosity of its group's colour, composited over the BC
colour; in a CIE-based space -- CalGray, CalRGB, ICCBased, or a device space a default space
remaps to one -- that is the CIE 1931 Y of the colour (ISO 32000-2 11.5.3), and the transfer
function TR applies to it (11.6.5.1). BC defaults to black (Table 142), 0 0 0 1 in a CMYK
space. A colour painted into a transparency group whose colour space is not its own converts
to the group's (11.7.2): an ICCBased CMYK colour into a DeviceCMYK group goes through the
built-in DeviceCMYK profile's CMYK, into an ICCBased CMYK group through that group's profile,
while DeviceCMYK in an ICCBased CMYK group is the group's own CMYK.

The pages here use CalGray and CalRGB spaces, ICC matrix-shaper RGB and gray profiles with
linear or gamma curves, and ICC CMYK profiles built here, "linear naive" ones whose linear
sRGB is (1 - s C)(1 - K) and so on for an ink strength s, all far from sRGB and from the
built-in DeviceCMYK profile. Each page's expected colours are worked out here from the
spaces' definitions, in plain Python: a mask's value is read through a black fill it masks
over white, 255 (1 - mask), inside each region the mask group paints, inside its bounding
box where it paints nothing, and outside the box; a group's colour is read where it paints.
Each sampled pixel must be within its case's tolerance.

uv run --all-packages python scripts/check_group_profiles.py
"""

from __future__ import annotations

import itertools
import struct
import sys
from collections.abc import Callable, Sequence
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
# A CalRGB matrix whose columns are SRGB_D50's primaries and whose white is D50.
CALRGB_MATRIX = b" ".join(b"%.7f" % value for primary in SRGB_D50 for value in primary)


def encode(linear: float) -> float:
    """The sRGB encoding of a linear component, 0 to 1."""
    linear = min(1.0, max(0.0, linear))
    return 12.92 * linear if linear <= 0.0031308 else 1.055 * linear ** (1 / 2.4) - 0.055


def luminance(red: float, green: float, blue: float) -> float:
    """The Y under D50 of a linear sRGB colour."""
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
    text = b"check_group_profiles\0"
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


def rgb_profile(gamma: float) -> bytes:
    """A matrix-shaper RGB profile: sRGB's primaries under D50, each curve x ** gamma."""
    return profile(
        b"mntr",
        b"RGB ",
        [
            (b"rXYZ", xyz_tag(SRGB_D50[0])),
            (b"gXYZ", xyz_tag(SRGB_D50[1])),
            (b"bXYZ", xyz_tag(SRGB_D50[2])),
            (b"rTRC", curve(gamma)),
            (b"gTRC", curve(gamma)),
            (b"bTRC", curve(gamma)),
        ],
    )


def gray_profile(gamma: float) -> bytes:
    """A gray profile whose Y is x ** gamma."""
    return profile(b"mntr", b"GRAY", [(b"kTRC", curve(gamma))])


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
    """A CMYK output profile: its A2B0 the linear naive conversion of an ink strength on a
    17-point grid, within 1 / 4096 of it between grid points; and its B2A0 an inverse on a
    33-point grid, K = 1 - max(R, G, B) and C = (1 - R / (1 - K)) / strength and so on, so
    that its black is black, as black point compensation finds it."""
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


# Where each page paints, as (x0, y0, x1, y1): two regions, the rest of a group's bounding
# box (which leaves a strip above it, y > 180, outside the box), and that strip.
FIRST = (20, 20, 90, 170)
SECOND = (110, 20, 180, 100)
EMPTY = (110, 120, 180, 170)
OUTSIDE = (20, 186, 180, 196)
GROUP_BBOX = b"[0 0 200 180]"


def rectangle(box: tuple[int, int, int, int]) -> bytes:
    x0, y0, x1, y1 = box
    return b"%d %d %d %d re f" % (x0, y0, x1 - x0, y1 - y0)


class Sample(NamedTuple):
    box: tuple[int, int, int, int]
    rgb: tuple[float, float, float]
    what: str


class Case(NamedTuple):
    name: str
    pdf: bytes
    samples: tuple[Sample, ...]
    tolerance: float


def masked(mask: float) -> tuple[float, float, float]:
    """The colour of a black fill painted over white through a mask value."""
    level = 255 * (1 - mask)
    return (level, level, level)


def mask_case(
    name: str,
    space: bytes,
    paint: tuple[bytes, bytes],
    expected: tuple[float, float],
    *,
    backdrop: tuple[bytes, float] | None = None,
    transfer: tuple[bytes, Callable[[float], float]] | None = None,
    profile_data: tuple[bytes, int] = (b"", 3),
    resources: bytes = b"",
    tolerance: float = 3,
) -> Case:
    """A page masked by a Luminosity mask whose group's CS is space, painting FIRST and
    SECOND with paint's operators, whose luminosities are expected; the rest of the group's
    box and the strip outside it take BC (black, luminosity 0, by default)."""
    function = transfer[1] if transfer is not None else (lambda value: value)
    bc_luminosity = 0.0 if backdrop is None else backdrop[1]
    entries = b""
    if backdrop is not None:
        entries += b" /BC [%s]" % backdrop[0]
    extra: list[bytes] = []
    if transfer is not None:
        entries += b" /TR 6 0 R"
    group = stream(
        b"/Type /XObject /Subtype /Form /BBox %s "
        b"/Group << /S /Transparency /CS %s >> /Resources << %s >>"
        % (GROUP_BBOX, space, resources),
        paint[0] + b" " + rectangle(FIRST) + b" " + paint[1] + b" " + rectangle(SECOND),
    )
    extra.append(group)
    extra.append(
        transfer[0] if transfer is not None else b"<< /FunctionType 2 /Domain [0 1] /N 1 >>"
    )
    extra.append(stream(b"/N %d" % profile_data[1], profile_data[0]))
    page = (
        b"/Resources << /ExtGState << /M << /SMask << /S /Luminosity /G 5 0 R%s >> >> >> >> "
        b"/Contents 4 0 R" % entries
    )
    content = b"/M gs 0 g 0 0 200 200 re f"
    return Case(
        name,
        document(page, content, extra),
        (
            Sample(FIRST, masked(function(expected[0])), "first colour"),
            Sample(SECOND, masked(function(expected[1])), "second colour"),
            Sample(EMPTY, masked(function(bc_luminosity)), "BC inside the box"),
            Sample(OUTSIDE, masked(function(bc_luminosity)), "BC outside the box"),
        ),
        tolerance,
    )


# The ICC profile object is 7 0 R on mask pages (object 5 is the group, 6 the transfer).
ICC = b"[/ICCBased 7 0 R]"


def calrgb(gamma: float) -> bytes:
    return b"[/CalRGB << /WhitePoint [0.9642 1 0.8249] /Gamma [%g %g %g] /Matrix [%s] >>]" % (
        gamma,
        gamma,
        gamma,
        CALRGB_MATRIX,
    )


def calrgb_y(components: Sequence[float], gamma: float) -> float:
    """11.5.3's formula: Y = YA A ** GR + YB B ** GG + YC C ** GB."""
    return sum(SRGB_D50[i][1] * components[i] ** gamma for i in range(3))


def mask_cases() -> list[Case]:
    first, second = (0.5, 0.5, 0.5), (0.2, 0.6, 0.9)
    naive = naive_cmyk_profile(1.0)
    first_cmyk, second_cmyk = (0.2, 0.4, 0.1, 0.1), (0.6, 0.1, 0.3, 0.0)
    gamma_half = b"<< /FunctionType 2 /Domain [0 1] /Range [0 1] /N 0.5 >>"
    linear = rgb_profile(1.0)

    def icc_paint(values: Sequence[float]) -> bytes:
        return b"/P cs %s sc" % numbers(values)

    icc_resources = b"/ColorSpace << /P %s >>" % ICC
    return [
        mask_case(
            "CalRGB, gamma 1",
            calrgb(1.0),
            (b"/P cs %s sc" % numbers(first), b"/P cs %s sc" % numbers(second)),
            (calrgb_y(first, 1.0), calrgb_y(second, 1.0)),
            resources=b"/ColorSpace << /P %s >>" % calrgb(1.0),
        ),
        mask_case(
            "CalRGB, gamma 2.2, a BC colour and a transfer function",
            calrgb(2.2),
            (b"/P cs %s sc" % numbers(first), b"/P cs %s sc" % numbers(second)),
            (calrgb_y(first, 2.2), calrgb_y(second, 2.2)),
            backdrop=(b"0.7 0.4 0.3", calrgb_y((0.7, 0.4, 0.3), 2.2)),
            transfer=(gamma_half, lambda value: value**0.5),
            resources=b"/ColorSpace << /P %s >>" % calrgb(2.2),
        ),
        mask_case(
            "CalGray, gamma 1.8",
            b"[/CalGray << /WhitePoint [0.9642 1 0.8249] /Gamma 1.8 >>]",
            (b"/P cs 0.5 sc", b"/P cs 0.8 sc"),
            (0.5**1.8, 0.8**1.8),
            backdrop=(b"0.3", 0.3**1.8),
            resources=b"/ColorSpace << /P [/CalGray << /WhitePoint [0.9642 1 0.8249] "
            b"/Gamma 1.8 >>] >>",
        ),
        mask_case(
            "ICCBased RGB, linear",
            ICC,
            (icc_paint(first), icc_paint(second)),
            (luminance(*first), luminance(*second)),
            profile_data=(linear, 3),
            resources=icc_resources,
        ),
        mask_case(
            "ICCBased gray, linear",
            b"[/ICCBased 7 0 R]",
            (icc_paint((0.5,)), icc_paint((0.25,))),
            (0.5, 0.25),
            profile_data=(gray_profile(1.0), 1),
            resources=icc_resources,
        ),
        mask_case(
            "ICCBased CMYK, BC its default black",
            ICC,
            (icc_paint(first_cmyk), icc_paint(second_cmyk)),
            (luminance(*naive_linear(first_cmyk, 1.0)), luminance(*naive_linear(second_cmyk, 1.0))),
            profile_data=(naive, 4),
            resources=icc_resources,
        ),
        mask_case(
            "ICCBased CMYK, a BC colour",
            ICC,
            (icc_paint(first_cmyk), icc_paint(second_cmyk)),
            (luminance(*naive_linear(first_cmyk, 1.0)), luminance(*naive_linear(second_cmyk, 1.0))),
            backdrop=(b"0.1 0.2 0.3 0.4", luminance(*naive_linear((0.1, 0.2, 0.3, 0.4), 1.0))),
            profile_data=(naive, 4),
            resources=icc_resources,
        ),
        mask_case(
            "DeviceRGB remapped by DefaultRGB to a linear ICCBased space, painted with rg",
            b"/DeviceRGB",
            (b"%s rg" % numbers(first), b"%s rg" % numbers(second)),
            (luminance(*first), luminance(*second)),
            profile_data=(linear, 3),
            resources=b"/ColorSpace << /DefaultRGB %s >>" % ICC,
        ),
        mask_case(
            "DeviceGray, as it was",
            b"/DeviceGray",
            (b"0.5 g", b"0.25 g"),
            (0.5, 0.25),
            backdrop=(b"0.2", 0.2),
        ),
        mask_case(
            "DeviceCMYK, as it was",
            b"/DeviceCMYK",
            (b"0.2 0.4 0.1 0.1 k", b"0.6 0.1 0.3 0 k"),
            (1 - min(1.0, 0.3 * 0.2 + 0.59 * 0.4 + 0.11 * 0.1 + 0.1), 1 - (0.18 + 0.059 + 0.033)),
        ),
    ]


def group_case(
    name: str,
    group_space: bytes,
    resources: bytes,
    paint: tuple[bytes, bytes],
    expected: tuple[tuple[float, float, float], tuple[float, float, float]],
    extra: Sequence[bytes],
    *,
    page_group: bytes | None = None,
    body: bytes | None = None,
    tolerance: float = 3,
) -> Case:
    """An isolated group whose CS is group_space (or, with page_group, the page group)
    painting FIRST and SECOND with paint's operators (or else body), whose sRGB (0 to 1) is
    expected."""
    if body is None:
        body = paint[0] + b" " + rectangle(FIRST) + b" " + paint[1] + b" " + rectangle(SECOND)
    if page_group is not None:
        pdf = document(
            b"/Group << /S /Transparency /CS %s >> /Resources << %s >> /Contents 4 0 R"
            % (page_group, resources),
            body,
            extra,
        )
    else:
        pdf = document(
            b"/Resources << /XObject << /G %d 0 R >> >> /Contents 4 0 R" % (5 + len(extra)),
            b"/G Do",
            [
                *extra,
                stream(
                    b"/Type /XObject /Subtype /Form /BBox [0 0 200 200] "
                    b"/Group << /S /Transparency /I true /CS %s >> /Resources << %s >>"
                    % (group_space, resources),
                    body,
                ),
            ],
        )
    return Case(
        name,
        pdf,
        (
            Sample(FIRST, scaled(expected[0]), "first colour"),
            Sample(SECOND, scaled(expected[1]), "second colour"),
        ),
        tolerance,
    )


def scaled(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    return 255 * rgb[0], 255 * rgb[1], 255 * rgb[2]


def naive_srgb(cmyk: Sequence[float], strength: float) -> tuple[float, float, float]:
    red, green, blue = naive_linear(cmyk, strength)
    return encode(red), encode(green), encode(blue)


def group_cases() -> list[Case]:
    naive = naive_cmyk_profile(1.0)
    weak = naive_cmyk_profile(0.8)
    # Light colours, which the built-in DeviceCMYK profile's gamut holds.
    first, second = (0.3, 0.2, 0.4, 0.1), (0.1, 0.35, 0.2, 0.05)
    # Objects 5 and 6: the profiles of spaces A (ink strength 1) and B (0.8).
    profiles = [stream(b"/N 4", naive), stream(b"/N 4", weak)]
    space_a, space_b = b"[/ICCBased 5 0 R]", b"[/ICCBased 6 0 R]"
    in_a = b"/ColorSpace << /A %s >>" % space_a

    def paint_a(values: Sequence[float]) -> bytes:
        return b"/A cs %s sc" % numbers(values)

    image = stream(
        b"/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace %s "
        b"/BitsPerComponent 8" % space_a,
        bytes(round(255 * value) for value in (*first, *second)),
    )
    separation = (
        b"[/Separation /Spot %s << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] "
        b"/C1 [0.6 0.4 0.8 0.2] /N 1 >>]" % space_a
    )
    image_case = group_case(
        "an ICCBased CMYK image in a DeviceCMYK group",
        b"/DeviceCMYK",
        b"/XObject << /I 7 0 R >>",
        (b"", b""),
        (naive_srgb(first, 1.0), naive_srgb(first, 1.0)),
        [*profiles, image],
        body=b"q 70 0 0 150 20 20 cm /I Do Q",
        tolerance=6,
    )
    image_case = image_case._replace(
        samples=(
            Sample((20, 20, 52, 170), scaled(naive_srgb(first, 1.0)), "left"),
            Sample((58, 20, 90, 170), scaled(naive_srgb(second, 1.0)), "right"),
        )
    )
    return [
        group_case(
            "ICCBased CMYK in a DeviceCMYK group",
            b"/DeviceCMYK",
            in_a,
            (paint_a(first), paint_a(second)),
            (naive_srgb(first, 1.0), naive_srgb(second, 1.0)),
            profiles,
            tolerance=6,
        ),
        group_case(
            "ICCBased CMYK in an ICCBased CMYK group of another profile",
            space_b,
            in_a,
            (paint_a(first), paint_a(second)),
            (naive_srgb(first, 1.0), naive_srgb(second, 1.0)),
            profiles,
        ),
        group_case(
            "ICCBased CMYK in a DeviceCMYK group DefaultCMYK remaps to another profile",
            b"/DeviceCMYK",
            in_a + b" /ColorSpace << /A %s /DefaultCMYK %s >>" % (space_a, space_b),
            (paint_a(first), b"%s k" % numbers(second)),
            (naive_srgb(first, 1.0), naive_srgb(second, 0.8)),
            profiles,
        ),
        image_case,
        group_case(
            "a Separation whose alternate is ICCBased CMYK, in an ICCBased CMYK group",
            space_b,
            b"/ColorSpace << /S %s >>" % separation,
            (b"/S cs 0.5 sc", b"/S cs 1 sc"),
            (naive_srgb((0.3, 0.2, 0.4, 0.1), 1.0), naive_srgb((0.6, 0.4, 0.8, 0.2), 1.0)),
            profiles,
        ),
        group_case(
            "ICCBased CMYK in a DeviceCMYK page group",
            b"",
            in_a,
            (paint_a(first), paint_a(second)),
            (naive_srgb(first, 1.0), naive_srgb(second, 1.0)),
            profiles,
            page_group=b"/DeviceCMYK",
            tolerance=6,
        ),
        group_case(
            "DeviceCMYK in an ICCBased CMYK group is the group's CMYK",
            space_b,
            b"",
            (b"%s k" % numbers(first), b"%s k" % numbers(second)),
            (naive_srgb(first, 0.8), naive_srgb(second, 0.8)),
            profiles,
        ),
        group_case(
            "DeviceRGB in an ICCBased CMYK group",
            space_b,
            b"",
            (b"0.8 0.6 0.7 rg", b"0.5 0.7 0.9 rg"),
            ((0.8, 0.6, 0.7), (0.5, 0.7, 0.9)),
            profiles,
        ),
        group_case(
            "ICCBased CMYK in a CalRGB group",
            calrgb(1.0),
            in_a,
            (paint_a(first), paint_a(second)),
            (naive_srgb(first, 1.0), naive_srgb(second, 1.0)),
            profiles,
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


def check(case: Case) -> list[str]:
    pixels = render(case.pdf)
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


def main() -> int:
    cases = [*mask_cases(), *group_cases()]
    failures = [failure for case in cases for failure in check(case)]
    for failure in failures:
        print(failure)
    print(f"{len(cases)} soft mask and group colour space pages: {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: AGPL-3.0-only

"""Check overprinting against the colorant arithmetic of ISO 32000-2 8.6.7 and 11.7.4.

An overprinting paint marks only the process colorants its colour space names: DeviceCMYK
under overprint mode 1, given directly as the current colour of a fill, stroke, text or
image mask, marks its nonzero components; a Separation or DeviceN space marks the process
colorants it names, whose tints it paints directly -- an image's samples and a shading's
function values included -- in a DeviceCMYK group or on a simulated page without a group
colour space, the device's native process space, while in an ICCBased CMYK group it
converts to its alternate and marks every component (11.7.3); a spot colorant marks none,
its alternate colour's tints adding to the backdrop's; any other space marks every
component, so overprinting changes nothing (Table 146). Core composites overprinting
where paint lands in a DeviceCMYK blending space: a transparency group or page group whose
colour space is CMYK; with RenderOptions(overprint="simulate") a page without a group
colour space too, its tiling cells and form XObjects included; and with "auto", such a
page of a PDF/X document.

Each page here paints a backdrop and an overprinting paint over it, mostly in a DeviceCMYK
page group, and names the CMYK each sampled pixel must show, worked out here from the
colour values: a marked component takes the paint's tint, an unmarked one the backdrop's
plus the paint's, clamped at 1, and a non-opaque paint composites as Normal does. The
expected CMYK becomes sRGB through the built-in DeviceCMYK profile, as a CMYK group's colour
does, and each pixel must be within TOLERANCE of it. Other pages must render exactly as a
companion page does: overprinting RGB colour, an RGB page group, overprint off.

A spot colorant keeps a plate of its own while its CMYK group paints, added to the process
inks when the group ends: a spot over the same spot replaces it ("Spot colourant named in
source space" is Cs, Table 146), whatever paints it, an overprinting paint keeps it and one
that does not overprint erases it. The plates must reach the page from tiling cells and
knockout groups, and cost memory for the area painted on them, not the page.

With --mupdf (and PyMuPDF installed: uv run --with pymupdf), each page is also rendered by
MuPDF with the overprint simulation mutool draw applies by default, the page's spot
colorants composited from separations of their own, and the mean difference from core's
rendering is printed for each page; it is reported, not checked, as MuPDF converts colour
its own way.

uv run --all-packages python scripts/check_overprint.py [--mupdf]
"""

from __future__ import annotations

import sys
import tracemalloc
from collections.abc import Sequence
from importlib import resources
from typing import NamedTuple

import numpy

from core_pdf import PdfDocument
from core_pdf.impl.graphics.device_profiles import INTERNAL_DEFAULT_CMYK_PROFILE, cmyk_bytes_to_srgb
from core_pdf.impl.render.model import RenderOptions

SIZE = 200
TOLERANCE = 3
CMYK_GROUP = b"/Group << /S /Transparency /CS /DeviceCMYK >> "
RGB_GROUP = b"/Group << /S /Transparency /CS /DeviceRGB >> "
# An ICCBased CMYK page group of the built-in profile (object 8), as long as CMYK_GROUP, so
# one can stand for the other in a built document.
ICC_GROUP = b"/Group<</S/Transparency/CS[/ICCBased 8 0 R]>>".ljust(len(CMYK_GROUP))
EXTGSTATES = (
    b"/ExtGState << "
    b"/Off << /OP false /op false >> "
    b"/On0 << /OP true /op true /OPM 0 >> "
    b"/On1 << /OP true /op true /OPM 1 >> "
    b"/Stroke << /OP true /op false /OPM 1 >> "
    b"/Fill << /OP false /op true /OPM 1 >> "
    b"/Half << /OP true /op true /OPM 1 /ca 0.5 >> "
    b"/Mul << /OP true /op true /OPM 1 /BM /Multiply >> "
    b"/Masked << /OP true /op true /OPM 1 /CA 0.5 /ca 0.5 "
    b"/SMask << /S /Luminosity /G 7 0 R >> >> "
    b"/MaskOn << /OP true /op true /OPM 1 /SMask << /S /Luminosity /G 7 0 R >> >> "
    b">> "
)
SPACES = (
    b"/ColorSpace << "
    b"/Spot [/Separation /Spot /DeviceCMYK "
    b"<< /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 0.5 1 0] /N 1 >>] "
    b"/Mag [/Separation /Magenta /DeviceCMYK "
    b"<< /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 1 0 0] /N 1 >>] "
    b"/All [/Separation /All /DeviceCMYK "
    b"<< /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [1 1 1 1] /N 1 >>] "
    b"/CS [/DeviceN [/Cyan /Spot] /DeviceCMYK 5 0 R] "
    b"/BlackGray [/Separation /Black /DeviceGray "
    b"<< /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >>] "
    b"/CyanRGB [/Separation /Cyan /DeviceRGB "
    b"<< /FunctionType 2 /Domain [0 1] /C0 [1 1 1] /C1 [0 1 1] /N 1 >>] "
    b"/CK [/DeviceN [/Cyan /Black] /DeviceRGB 6 0 R] "
    b"/ICC [/ICCBased 8 0 R] "
    b"/Spot2 [/Separation /Spot2 /DeviceCMYK "
    b"<< /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0.3 0 0 0] /N 1 >>] "
    b"/IdxBlack [/Indexed [/Separation /Black /DeviceGray "
    b"<< /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >>] 1 <0080>] "
    b">> "
)
# The DeviceN space's tint transform: c s to c, s / 2, s, 0.
DEVICEN_FUNCTION = (
    b"/FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1 0 1]",
    b"{ dup 0.5 mul exch 0 }",
)
# The DeviceN Cyan and Black space's tint transform to RGB: c k to (1 - c)(1 - k), 1 - k,
# 1 - k.
CYAN_BLACK_FUNCTION = (
    b"/FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1]",
    b"{ 1 exch sub exch 1 exch sub 1 index mul exch dup }",
)
# The soft mask's group: a gray ramp from 0 at the page's left edge to 1 at its right, so
# a pixel at x masks to (x + 0.5) / SIZE.
SOFT_MASK_GROUP = (
    b"/Type /XObject /Subtype /Form /BBox [0 0 200 200] "
    b"/Group << /S /Transparency /CS /DeviceGray >> "
    b"/Resources << /Shading << /Ramp << /ShadingType 2 /ColorSpace /DeviceGray "
    b"/Coords [0 0 200 0] /Extend [true true] "
    b"/Function << /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [1] /N 1 >> >> >> >>",
    b"/Ramp sh",
)


def built_in_cmyk_profile() -> bytes:
    """The built-in DeviceCMYK profile, for an ICCBased CMYK space."""
    return (
        resources.files("core_pdf.impl.data.icc")
        .joinpath(INTERNAL_DEFAULT_CMYK_PROFILE)
        .read_bytes()
    )


# The regions every page paints: the backdrop at x 20 to 100, the paint at x 60 to 180,
# overlapping from 60 to 100, and the pixel sampled in each, at y 100.
BACKDROP_ONLY = (40, 100)
OVERLAP = (80, 100)
PAINT_ONLY = (140, 100)
BACKDROP_RECT = b"20 20 80 160 re f "
PAINT_RECT = b"60 60 120 80 re"

type Cmyk = tuple[float, float, float, float]


def stream(dictionary: bytes, data: bytes) -> bytes:
    return b"<< " + dictionary + b" /Length %d >>\nstream\n" % len(data) + data + b"\nendstream"


def document(
    page: bytes,
    content: bytes,
    extra: Sequence[bytes] = (),
    *,
    resources_extra: bytes = b"",
    catalog: bytes = b"",
    info: bytes | None = None,
) -> bytes:
    """A one-page PDF: the page dictionary's entries, its content, and objects 9 onwards;
    objects 5 and 6 are the DeviceN spaces' tint transforms, 7 the soft mask's group, 8
    the ICC profile and 9 the same profile with a Range beyond 0 to 1, and the page's
    resources name the graphics states, colour spaces, a font F and the XObjects of extra as
    X10, X11..., with resources_extra's entries after them. catalog holds more entries of
    the catalog, and info, the entries of a document information dictionary."""
    xobjects = b"".join(b"/X%d %d 0 R " % (number, number) for number in range(10, 10 + len(extra)))
    resources = (
        b"/Resources << "
        + EXTGSTATES
        + SPACES
        + b"/Font << /F << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> "
        + b"/XObject << "
        + xobjects
        + b">> "
        + resources_extra
        + b">> "
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R " + catalog + b">>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] " % (SIZE, SIZE)
        + resources
        + page
        + b"/Contents 4 0 R >>",
        stream(b"", content),
        stream(*DEVICEN_FUNCTION),
        stream(*CYAN_BLACK_FUNCTION),
        stream(*SOFT_MASK_GROUP),
        stream(b"/N 4 /Alternate /DeviceCMYK", built_in_cmyk_profile()),
        stream(b"/N 4 /Alternate /DeviceCMYK /Range [0 1 0 1 0 1 0 2]", built_in_cmyk_profile()),
        *extra,
    ]
    trailer = b""
    if info is not None:
        objects.append(b"<< " + info + b">>")
        trailer = b"/Info %d 0 R " % len(objects)
    out = bytearray(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R %s>>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        trailer,
        xref,
    )
    return bytes(out)


def render(data: bytes, overprint: str = "groups") -> numpy.ndarray:
    """The page's pixels over white, as RGB."""
    with PdfDocument(data) as pdf:
        image = pdf.pages[0].render(RenderOptions(overprint=overprint)).rasterize()
    pixels = numpy.frombuffer(bytes(image.pixels), numpy.uint8).reshape(
        image.height, image.width, 4
    )
    rgb = pixels[..., :3].astype(numpy.float64)
    alpha = pixels[..., 3:].astype(numpy.float64) / 255.0
    return numpy.rint(rgb * alpha + 255.0 * (1.0 - alpha)).astype(numpy.int16)


def srgb(cmyk: Cmyk) -> tuple[int, ...]:
    """The sRGB bytes a CMYK colour renders as in a DeviceCMYK group."""
    samples = numpy.array([[round(min(1.0, max(0.0, c)) * 255) for c in cmyk]], numpy.uint8)
    return tuple(int(value) for value in cmyk_bytes_to_srgb(samples, None)[0])


def add(backdrop: Cmyk, paint: Cmyk, marks: str) -> Cmyk:
    """The special blend mode's result, fully opaque: a marked component (of "cmyk") takes
    the paint's tint, an unmarked one the backdrop's plus the paint's, clamped at 1."""
    cyan, magenta, yellow, black = (
        p if name in marks else min(1.0, b + p)
        for name, b, p in zip("cmyk", backdrop, paint, strict=True)
    )
    return cyan, magenta, yellow, black


class Case(NamedTuple):
    name: str
    data: bytes
    # Each sampled pixel and the CMYK it must show.
    samples: tuple[tuple[tuple[int, int], Cmyk], ...]
    overprint: str = "groups"
    # A known difference: core does not render the samples yet, which is reported, not
    # failed; passing becomes a failure, so the case is promoted.
    known: bool = False


class Same(NamedTuple):
    """Two pages that must render the same pixels."""

    name: str
    data: bytes
    reference: bytes
    overprint: str = "groups"
    reference_overprint: str = "groups"


CYAN: Cmyk = (1.0, 0.0, 0.0, 0.0)
MAGENTA: Cmyk = (0.0, 1.0, 0.0, 0.0)


def two_fills(paint: bytes, *, group: bytes = CMYK_GROUP, backdrop: bytes = b"1 0 0 0 k") -> bytes:
    """A cyan backdrop and paint, which sets its state and colour, filling the paint's
    rectangle."""
    return document(group, backdrop + b" " + BACKDROP_RECT + paint + b" " + PAINT_RECT + b" f")


def regions(backdrop: Cmyk, paint: Cmyk, overlap: Cmyk) -> tuple[tuple[tuple[int, int], Cmyk], ...]:
    return ((BACKDROP_ONLY, backdrop), (OVERLAP, overlap), (PAINT_ONLY, paint))


def simulated(case: Case) -> Case:
    """case on a page without a group, simulating overprint: where the page stands for the
    output device's CMYK, a Separation or DeviceN space may name process colorants as such
    (11.7.3). The page group's entry becomes spaces, keeping every offset."""
    return case._replace(
        name=f"{case.name} (simulated page)",
        data=case.data.replace(CMYK_GROUP, b" " * len(CMYK_GROUP), 1),
        overprint="simulate",
    )


def turned_off(data: bytes) -> bytes:
    """data with its overprinting state On0 as Off, keeping every offset."""
    return data.replace(b"/On0 gs", b"/Off gs")


# The fill cases naming process colorants in a Separation or DeviceN space, which only a
# page standing for the device's CMYK paints as such.
DEVICE_FILLS = frozenset(
    {
        "Separation Magenta marks magenta alone",
        "DeviceN Cyan and a spot: cyan replaces, the spot adds",
        "Separation Black over DeviceGray paints its tint as black, and keeps CMY",
        "Separation Cyan over DeviceRGB paints its tint as cyan, and keeps MYK",
        "DeviceN Cyan and Black over DeviceRGB paints both tints, and keeps MY",
        "Indexed over Separation Black paints its entry's tint as black",
    }
)
# Separation Magenta text, a stroke along y 30 and an image mask, over a cyan backdrop.
MAGENTA_TEXT = b"1 0 0 0 k 0 0 200 200 re f /On0 gs /Mag cs 0.6 sc BT /F 150 Tf 30 50 Td (H) Tj ET"
MAGENTA_STROKE = b"1 0 0 0 k " + BACKDROP_RECT + b"/On0 gs /Mag CS 0.6 SC 10 w 40 30 m 180 30 l S"
MAGENTA_MASK = (
    b"1 0 0 0 k " + BACKDROP_RECT + b"/On0 gs /Mag cs 0.6 sc q 120 0 0 80 60 60 cm /X10 Do Q"
)


def device_paint_cases() -> list[Case]:
    """A stroke, text and an image mask in Separation Magenta on a simulated page mark
    magenta alone."""
    magenta = (0.0, 0.6, 0.0, 0.0)
    over_cyan = (1.0, 0.6, 0.0, 0.0)
    return [
        Case(
            "a Separation Magenta stroke marks magenta alone",
            document(b"", MAGENTA_STROKE),
            (((80, 170), over_cyan), ((140, 170), magenta)),
            "simulate",
        ),
        Case(
            "Separation Magenta text marks magenta alone",
            document(b"", MAGENTA_TEXT),
            (((48, 100), over_cyan),),
            "simulate",
        ),
        Case(
            "a Separation Magenta image mask marks magenta alone",
            document(b"", MAGENTA_MASK, [IMAGE_MASK]),
            regions(CYAN, magenta, over_cyan),
            "simulate",
        ),
    ]


def in_icc_group(data: bytes) -> bytes:
    """data, a page in a DeviceCMYK page group, in an ICCBased CMYK one instead."""
    return data.replace(CMYK_GROUP, ICC_GROUP, 1)


# A form XObject (object 10) whose isolated DeviceCMYK group paints the cyan backdrop and
# fills the paint's rectangle with Separation Magenta at 0.6, overprinting.
MAGENTA_FORM = stream(
    b"/Type /XObject /Subtype /Form /BBox [0 0 200 200] "
    b"/Group << /S /Transparency /CS /DeviceCMYK /I true >> /Resources << "
    + EXTGSTATES
    + SPACES
    + b">>",
    b"1 0 0 0 k " + BACKDROP_RECT + b"/On0 gs /Mag cs 0.6 sc " + PAINT_RECT + b" f",
)


def device_group_cases() -> list[Case]:
    """A group whose colour space is DeviceCMYK itself composites in the native process
    space of the CMYK device: a Separation Magenta paint in it marks magenta alone, on any
    page."""
    overprinted = regions(CYAN, (0.0, 0.6, 0.0, 0.0), (1.0, 0.6, 0.0, 0.0))
    content = b"/X10 Do"
    return [
        Case(
            "a DeviceCMYK form group on a DeviceCMYK page marks magenta alone",
            document(CMYK_GROUP, content, [MAGENTA_FORM]),
            overprinted,
        ),
        Case(
            "a DeviceCMYK form group on a page without a group marks magenta alone",
            document(b"", content, [MAGENTA_FORM]),
            overprinted,
        ),
    ]


def icc_group_sames() -> list[Same]:
    """In a group whose colour space is ICCBased CMYK -- "any other colour space" than the
    device's -- a Separation or DeviceN space naming process colorants converts to its
    alternate (11.7.3): a process colour space, every component of which takes the source's
    value (Table 146), so it paints as with overprint off; so does a DeviceCMYK group inside
    one, whose space is the ancestor's (11.7.2)."""
    pages = [(case.name, case.data) for case in fill_cases() if case.name in DEVICE_FILLS]
    pages += [
        ("a Separation Magenta fill", two_fills(b"/On0 gs /Mag cs 0.6 sc")),
        (
            "a DeviceN Cyan and spot fill",
            two_fills(b"/On0 gs /CS cs 0.4 1 sc", backdrop=b"0 0.3 0 0.2 k"),
        ),
        ("a Separation Magenta stroke", document(CMYK_GROUP, MAGENTA_STROKE)),
        ("Separation Magenta text", document(CMYK_GROUP, MAGENTA_TEXT)),
        ("a Separation Magenta image mask", document(CMYK_GROUP, MAGENTA_MASK, [IMAGE_MASK])),
        (
            "a DeviceN Cyan and Magenta image",
            document(CMYK_GROUP, b"0 0 0 0.5 k 0 0 200 200 re f /On0 gs " + PLACE, CM_IMAGE),
        ),
        ("a Separation Cyan shading", cyan_shading(b"/On0")),
        (
            "a Separation Cyan shading pattern",
            shading_pattern_page(CYAN_RGB_SEPARATION, 0.6, b"0 1 0 0 k", CMYK_GROUP),
        ),
    ]
    sames = [
        Same(
            f"in an ICCBased CMYK group, {name}: it paints as with overprint off",
            in_icc_group(data),
            turned_off(in_icc_group(data)),
        )
        for name, data in pages
    ]
    nested = document(ICC_GROUP, b"/X10 Do", [MAGENTA_FORM])
    sames.append(
        Same(
            "a DeviceCMYK form group in an ICCBased CMYK page group paints Separation Magenta "
            "as with overprint off",
            nested,
            turned_off(nested),
        )
    )
    # On a simulated page (kept simulating by an OPM 1 fill of 0 0 0 0 k, which marks
    # nothing), a DeviceN Cyan and Magenta image whose soft mask has a Matte: its tints
    # cannot be painted directly, so it paints its alternate's colour, not overprinting.
    matted = (
        stream(
            b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 8 "
            b"/ColorSpace [/DeviceN [/Cyan /Magenta] /DeviceRGB 11 0 R] /SMask 12 0 R",
            bytes([153, 102]),
        ),
        CM_IMAGE[1],
        stream(
            b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 8 "
            b"/ColorSpace /DeviceGray /Matte [0 0]",
            bytes([255]),
        ),
    )
    data = document(
        b"", b"0 0 0 0.5 k 0 0 200 200 re f /On1 gs 0 0 0 0 k 0 0 1 1 re f /On0 gs " + PLACE, matted
    )
    # An Indexed image over Separation Magenta whose lookup is one entry short of hival + 1,
    # at the missing index: likewise.
    short = stream(
        b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 8 "
        b"/ColorSpace [/Indexed [/Separation /Magenta /DeviceCMYK "
        b"<< /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 1 0 0] /N 1 >>] 1 <80>]",
        bytes([1]),
    )
    short_page = document(CMYK_GROUP, b"1 0 0 0 k 0 0 200 200 re f /On0 gs " + PLACE, [short])
    sames.append(
        Same(
            "an Indexed image whose lookup is short does not overprint",
            short_page,
            turned_off(short_page),
        )
    )
    sames.append(
        Same(
            "an image whose tints cannot be painted directly does not overprint",
            data,
            turned_off(data),
            "simulate",
            "simulate",
        )
    )
    return sames


def fill_cases() -> list[Case]:
    spot = (0.0, 0.5, 1.0, 0.0)
    soft_backdrop: Cmyk = (0.0, 0.3, 0.0, 0.2)
    devicen = (0.4, 0.5, 1.0, 0.0)
    return [
        Case(
            "op false: magenta replaces cyan",
            two_fills(b"/Off gs 0 1 0 0 k"),
            regions(CYAN, MAGENTA, MAGENTA),
        ),
        Case(
            "op true, OPM 1: magenta's zero cyan keeps the backdrop's",
            two_fills(b"/On1 gs 0 1 0 0 k"),
            regions(CYAN, MAGENTA, add(CYAN, MAGENTA, "m")),
        ),
        Case(
            "op true, OPM 0: DeviceCMYK marks all four components",
            two_fills(b"/On0 gs 0 1 0 0 k"),
            regions(CYAN, MAGENTA, MAGENTA),
        ),
        Case(
            "OPM 1 with two nonzero components",
            two_fills(b"/On1 gs 0.5 0 1 0 k", backdrop=b"0.2 0.7 0.1 0.3 k"),
            regions((0.2, 0.7, 0.1, 0.3), (0.5, 0, 1, 0), (0.5, 0.7, 1.0, 0.3)),
        ),
        Case(
            "OPM 1 at 50% opacity: the marked component composites, the others stay",
            two_fills(b"/Half gs 0 1 0 0 k"),
            # Alone, the paint composites half opaque over the page's white, in sRGB.
            ((BACKDROP_ONLY, CYAN), (OVERLAP, (1.0, 0.5, 0.0, 0.0))),
        ),
        Case(
            "Separation spot: its alternate's tints add to the process inks",
            two_fills(b"/On0 gs /Spot cs 1 sc"),
            regions(CYAN, spot, add(CYAN, spot, "")),
        ),
        Case(
            "Separation spot, op false: it erases the process inks",
            two_fills(b"/Off gs /Spot cs 1 sc"),
            regions(CYAN, spot, spot),
        ),
        Case(
            "Separation Magenta marks magenta alone",
            two_fills(b"/On0 gs /Mag cs 0.6 sc"),
            regions(CYAN, (0, 0.6, 0, 0), (1.0, 0.6, 0.0, 0.0)),
        ),
        Case(
            "Separation All marks every component",
            two_fills(b"/On1 gs /All cs 0.5 sc"),
            regions(CYAN, (0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
        ),
        Case(
            "DeviceN Cyan and a spot: cyan replaces, the spot adds",
            two_fills(b"/On0 gs /CS cs 0.4 1 sc", backdrop=b"0 0.3 0 0.2 k"),
            regions(soft_backdrop, devicen, (0.4, 0.8, 1.0, 0.2)),
        ),
        Case(
            "Separation Black over DeviceGray paints its tint as black, and keeps CMY",
            two_fills(b"/On0 gs /BlackGray cs 0.5 sc", backdrop=b"0.3 0 0 0 k"),
            regions((0.3, 0, 0, 0), (0, 0, 0, 0.5), (0.3, 0.0, 0.0, 0.5)),
        ),
        Case(
            "Separation Cyan over DeviceRGB paints its tint as cyan, and keeps MYK",
            two_fills(b"/On0 gs /CyanRGB cs 0.6 sc", backdrop=b"0 1 0 0 k"),
            regions(MAGENTA, (0.6, 0, 0, 0), (0.6, 1.0, 0.0, 0.0)),
        ),
        Case(
            "DeviceN Cyan and Black over DeviceRGB paints both tints, and keeps MY",
            two_fills(b"/On0 gs /CK cs 0.4 0.5 sc", backdrop=b"0 1 0 0 k"),
            regions(MAGENTA, (0.4, 0, 0, 0.5), (0.4, 1.0, 0.0, 0.5)),
        ),
        Case(
            "Indexed over Separation Black paints its entry's tint as black",
            two_fills(b"/On0 gs /IdxBlack cs 1 sc", backdrop=b"0.3 0 0 0 k"),
            regions((0.3, 0, 0, 0), (0, 0, 0, 128 / 255), (0.3, 0.0, 0.0, 128 / 255)),
        ),
        Case(
            "Multiply: the implicit group's result multiplies with the backdrop",
            two_fills(b"/Mul gs 0 1 0 0 k", backdrop=b"0.5 0 0 0 k"),
            # Cyan, unmarked, blends to 0.5 and multiplies with 0.5: additive 0.25. Magenta
            # multiplies with the backdrop's 0.
            regions((0.5, 0, 0, 0), MAGENTA, (0.75, 1.0, 0.0, 0.0)),
        ),
        Case(
            "a page without a group overprints when simulating",
            two_fills(b"/On1 gs 0 1 0 0 k", group=b""),
            regions(CYAN, MAGENTA, add(CYAN, MAGENTA, "m")),
            "simulate",
        ),
        Case(
            "a CMYK page group ignores overprinting when told to",
            two_fills(b"/On1 gs 0 1 0 0 k"),
            regions(CYAN, MAGENTA, MAGENTA),
            "ignore",
        ),
    ]


def stroke_cases() -> list[Case]:
    # A magenta fill and, along y 30, a magenta stroke 10 wide, over the cyan backdrop.
    content = b"1 0 0 0 k " + BACKDROP_RECT + b"0 1 0 0 k 0 1 0 0 K 10 w %s gs " + PAINT_RECT
    content += b" f 40 30 m 180 30 l S"
    stroke_overlap = (80, 170)
    cases = []
    for state, fill_overprints, stroke_overprints in (
        (b"/Stroke", False, True),
        (b"/Fill", True, False),
    ):
        cases.append(
            Case(
                f"{state.decode()[1:].lower()} overprint alone",
                document(CMYK_GROUP, content.replace(b"%s", state)),
                (
                    (OVERLAP, add(CYAN, MAGENTA, "m") if fill_overprints else MAGENTA),
                    (stroke_overlap, add(CYAN, MAGENTA, "m") if stroke_overprints else MAGENTA),
                ),
            )
        )
    cases.append(
        Case(
            "B with overprint: fill and stroke each overprint",
            document(
                CMYK_GROUP,
                b"1 0 0 0 k " + BACKDROP_RECT + b"/On1 gs 0 1 0 0 k 0 0 1 0 K 10 w "
                b"60 60 120 80 re B",
            ),
            # The stroke runs along the rectangle's left edge, x 55 to 65, over the fill.
            ((OVERLAP, add(CYAN, MAGENTA, "m")), ((60, 100), (1.0, 1.0, 1.0, 0.0))),
        )
    )
    return cases


def masked_fill_stroke_case() -> Case:
    """B overprinting magenta over cyan through a soft mask that ramps across the page:
    fill and stroke overprint within one group, which composites through the mask."""
    content = b"1 0 0 0 k 0 0 200 200 re f /MaskOn gs 0 1 0 0 k 0 1 0 0 K 10 w 60 60 120 80 re B"
    samples = tuple(((x, 100), (1.0, (x + 0.5) / SIZE, 0.0, 0.0)) for x in (90, 120, 150))
    return Case("B through a ramping soft mask", document(CMYK_GROUP, content), samples)


def group_cases() -> list[Case]:
    inner = b"1 0 0 0 k " + BACKDROP_RECT + b"/On1 gs 0 1 0 0 k " + PAINT_RECT + b" f"
    cases = []
    for knockout, overlap in ((False, add(CYAN, MAGENTA, "m")), (True, MAGENTA)):
        form = stream(
            b"/Type /XObject /Subtype /Form /BBox [0 0 %d %d] " % (SIZE, SIZE)
            + b"/Group << /S /Transparency /CS /DeviceCMYK /I true /K %s >> "
            % (b"true" if knockout else b"false")
            + b"/Resources << "
            + EXTGSTATES
            + b">>",
            inner,
        )
        cases.append(
            Case(
                f"an isolated {'knockout' if knockout else 'non-knockout'} CMYK group on an "
                "RGB page",
                document(b"", b"/X10 Do", [form]),
                # In a knockout group magenta composites with the group's empty backdrop.
                regions(CYAN, MAGENTA, overlap),
            )
        )
    return cases


IMAGE_MASK = stream(
    b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /ImageMask true /BitsPerComponent 1",
    b"\x00",
)


def image_cases() -> list[Case]:
    place = b"q 120 0 0 80 60 60 cm /X10 Do Q"
    mask = IMAGE_MASK
    separation = stream(
        b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 8 "
        b"/ColorSpace [/Separation /Spot /DeviceCMYK "
        b"<< /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 0.5 1 0] /N 1 >>]",
        bytes([153]),
    )
    cmyk = stream(
        b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 8 "
        b"/ColorSpace /DeviceCMYK",
        bytes([0, 255, 0, 0]),
    )
    backdrop = b"1 0 0 0 k " + BACKDROP_RECT
    return [
        Case(
            "an image mask paints the current colour, which OPM 1 applies to",
            document(CMYK_GROUP, backdrop + b"/On1 gs 0 1 0 0 k " + place, [mask]),
            regions(CYAN, MAGENTA, add(CYAN, MAGENTA, "m")),
        ),
        Case(
            "a Separation spot image adds its alternate's tints",
            document(CMYK_GROUP, backdrop + b"/On0 gs " + place, [separation]),
            regions(CYAN, (0, 0.3, 0.6, 0), (1.0, 0.3, 0.6, 0.0)),
        ),
        Case(
            "a DeviceCMYK image marks all four components, whatever OPM",
            document(CMYK_GROUP, backdrop + b"/On1 gs " + place, [cmyk]),
            regions(CYAN, MAGENTA, MAGENTA),
        ),
    ]


def shading_case() -> Case:
    shading = (
        b"<< /ShadingType 2 /ColorSpace [/Separation /Spot /DeviceCMYK "
        b"<< /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 0.5 1 0] /N 1 >>] "
        b"/Coords [60 0 180 0] /Function << /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [1] "
        b"/N 1 >> >>"
    )
    content = b"1 0 0 0 k " + BACKDROP_RECT + b"/On0 gs q " + PAINT_RECT + b" W n /Sh sh Q"
    data = document(CMYK_GROUP, content).replace(
        b"/XObject << >>", b"/XObject << >> /Shading << /Sh " + shading + b" >>"
    )
    # The pixel's centre, x + 0.5, sets the tint along the axis from x 60 to 180.
    samples = []
    for x in (70, 90, 150):
        tint = (x + 0.5 - 60) / 120
        samples.append(((x, 100), (1.0 if x < 100 else 0.0, tint / 2, tint, 0.0)))
    return Case("a Separation spot shading adds its alternate's tints", data, tuple(samples))


CYAN_CMYK_SEPARATION = (
    b"[/Separation /Cyan /DeviceCMYK "
    b"<< /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [1 0 0 0] /N 1 >>]"
)
CYAN_RGB_SEPARATION = (
    b"[/Separation /Cyan /DeviceRGB "
    b"<< /FunctionType 2 /Domain [0 1] /C0 [1 1 1] /C1 [0 1 1] /N 1 >>]"
)
SPOT_SEPARATION = (
    b"[/Separation /Spot /DeviceCMYK "
    b"<< /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 0.5 1 0] /N 1 >>]"
)


def with_shadings(data: bytes, shadings: bytes) -> bytes:
    """data, a page from document, with a Shading resource dictionary of shadings."""
    return data.replace(b"/XObject << ", b"/Shading << " + shadings + b">> /XObject << ", 1)


def process_shading_cases() -> list[Case]:
    """Shadings in spaces naming process colorants alone paint their tints directly while
    overprinting, and leave the components they do not name as the backdrop has them."""
    cases = []
    for alternate, space in (("CMYK", CYAN_CMYK_SEPARATION), ("RGB", CYAN_RGB_SEPARATION)):
        shading = (
            b"/Sh << /ShadingType 2 /ColorSpace " + space + b" /Coords [60 0 180 0] "
            b"/Function << /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [1] /N 1 >> >> "
        )
        content = b"0 1 0 0 k " + BACKDROP_RECT + b"/On0 gs q " + PAINT_RECT + b" W n /Sh sh Q"
        samples = []
        for x in (70, 90, 150):
            tint = (x + 0.5 - 60) / 120
            samples.append(((x, 100), (tint, 1.0 if x < 100 else 0.0, 0.0, 0.0)))
        cases.append(
            Case(
                f"a Separation Cyan shading over a {alternate} alternate paints its tint as "
                "cyan, and keeps magenta",
                with_shadings(document(CMYK_GROUP, content), shading),
                tuple(samples),
            )
        )
    # A function-based shading in DeviceN Magenta and Yellow through an RGB alternate (object
    # 10), at tints 0.3 and 0.7 everywhere (object 11).
    function_shading = (
        b"/Sh << /ShadingType 1 /ColorSpace [/DeviceN [/Magenta /Yellow] /DeviceRGB 10 0 R] "
        b"/Domain [0 200 0 200] /Function 11 0 R >> "
    )
    extra = (
        stream(
            b"/FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1]",
            b"{ 1 exch sub exch 1 exch sub exch 1 3 1 roll }",
        ),
        stream(b"/FunctionType 4 /Domain [0 200 0 200] /Range [0 1 0 1]", b"{ pop pop 0.3 0.7 }"),
    )
    content = b"1 0 0 0 k " + BACKDROP_RECT + b"/On0 gs q " + PAINT_RECT + b" W n /Sh sh Q"
    cases.append(
        Case(
            "a function-based DeviceN Magenta and Yellow shading over an RGB alternate paints "
            "both tints, and keeps cyan",
            with_shadings(document(CMYK_GROUP, content, extra), function_shading),
            regions(CYAN, (0, 0.3, 0.7, 0), (1.0, 0.3, 0.7, 0.0)),
        )
    )
    # A free-form mesh (object 10) of two triangles over the paint's rectangle, every vertex
    # at the Separation Cyan tint 128 / 255: flag, x, y and tint, a byte each.
    vertices = bytes(
        [0, 0, 0, 128, 0, 255, 0, 128, 0, 0, 255, 128, 0, 255, 0, 128, 0, 255, 255, 128]
        + [0, 0, 255, 128]
    )
    mesh = stream(
        b"/ShadingType 4 /ColorSpace " + CYAN_RGB_SEPARATION + b" /BitsPerCoordinate 8 "
        b"/BitsPerComponent 8 /BitsPerFlag 8 /Decode [60 180 60 140 0 1]",
        vertices,
    )
    content = b"0 1 0 0 k " + BACKDROP_RECT + b"/On0 gs /Sh sh"
    cases.append(
        Case(
            "a mesh in Separation Cyan over an RGB alternate paints its tint as cyan, and keeps "
            "magenta",
            with_shadings(document(CMYK_GROUP, content, [mesh]), b"/Sh 10 0 R "),
            regions(MAGENTA, (128 / 255, 0, 0, 0), (128 / 255, 1.0, 0.0, 0.0)),
        )
    )
    return cases


def process_image_cases() -> list[Case]:
    """Images in spaces naming process colorants alone paint their samples' tints directly
    while overprinting, before the tint transform, and keep the other components."""
    black_image = stream(
        b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 4 "
        b"/Decode [1 0] /ColorSpace [/Separation /Black /DeviceGray "
        b"<< /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >>]",
        bytes([0x50]),
    )
    # The 4-bit sample 5 decodes through [1 0] to the tint 1 - 5 / 15.
    black = 1 - 5 / 15
    return [
        Case(
            "a DeviceN Cyan and Magenta image over an RGB alternate paints its tints, and "
            "keeps black",
            document(CMYK_GROUP, b"0 0 0 0.5 k 0 0 200 200 re f /On0 gs " + PLACE, CM_IMAGE),
            ((OVERLAP, (0.6, 0.4, 0.0, 0.5)), (PAINT_ONLY, (0.6, 0.4, 0.0, 0.5))),
        ),
        Case(
            "a 4-bit Separation Black image with a Decode array paints its decoded tint",
            document(
                CMYK_GROUP, b"1 0 0 0 k " + BACKDROP_RECT + b"/On0 gs " + PLACE, [black_image]
            ),
            regions(CYAN, (0, 0, 0, black), (1.0, 0.0, 0.0, black)),
        ),
    ]


def shading_pattern_page(space: bytes, tint: float, backdrop: bytes, group: bytes) -> bytes:
    """A backdrop, then the paint's rectangle filled, overprinting, with a shading pattern
    (object 10) in space at tint everywhere."""
    pattern = (
        b"<< /PatternType 2 /Shading << /ShadingType 2 /ColorSpace " + space + b" "
        b"/Coords [0 0 200 0] /Function << /FunctionType 2 /Domain [0 1] "
        b"/C0 [%g] /C1 [%g] /N 1 >> >> >>" % (tint, tint)
    )
    content = backdrop + b" " + BACKDROP_RECT + b"/On0 gs /Pattern cs /P scn " + PAINT_RECT + b" f"
    return document(group, content, [pattern], resources_extra=b"/Pattern << /P 10 0 R >> ")


def shading_pattern_cases() -> list[Case]:
    """A fill with a shading pattern marks what the shading's colour space names."""
    spot = (0.0, 0.5, 1.0, 0.0)
    return [
        Case(
            "a shading pattern in Separation Cyan over an RGB alternate paints cyan, and "
            "keeps magenta",
            shading_pattern_page(CYAN_RGB_SEPARATION, 0.6, b"0 1 0 0 k", CMYK_GROUP),
            regions(MAGENTA, (0.6, 0, 0, 0), (0.6, 1.0, 0.0, 0.0)),
        ),
        Case(
            "a shading pattern in a spot colour adds its alternate's tints",
            shading_pattern_page(SPOT_SEPARATION, 1.0, b"1 0 0 0 k", CMYK_GROUP),
            regions(CYAN, spot, add(CYAN, spot, "")),
        ),
    ]


def tiling_page(group: bytes) -> bytes:
    """A cyan backdrop, then the paint's rectangle filled with a tiling pattern whose cell
    (object 10) fills itself with magenta, overprinting under OPM 1."""
    cell = stream(
        b"/PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 40 40] /XStep 40 /YStep 40 "
        b"/Resources << " + EXTGSTATES + b">>",
        b"/On1 gs 0 1 0 0 k 0 0 40 40 re f",
    )
    content = b"1 0 0 0 k " + BACKDROP_RECT + b"/Pattern cs /T scn " + PAINT_RECT + b" f"
    return document(group, content, [cell], resources_extra=b"/Pattern << /T 10 0 R >> ")


def census_cases() -> list[Case]:
    """Overprinting inside a tiling pattern's cell or a form XObject: the cell composites
    with the backdrop, as the non-isolated group 11.6.7 encloses a pattern in, and
    simulation sees the overprinting wherever it is."""
    form = stream(
        b"/Type /XObject /Subtype /Form /BBox [0 0 200 200] /Resources << " + EXTGSTATES + b">>",
        b"/On1 gs 0 1 0 0 k " + PAINT_RECT + b" f",
    )
    overprinted = regions(CYAN, MAGENTA, add(CYAN, MAGENTA, "m"))
    return [
        Case("a tiling cell overprints its backdrop", tiling_page(CMYK_GROUP), overprinted),
        Case(
            "simulation sees overprinting in a tiling cell",
            tiling_page(b""),
            overprinted,
            "simulate",
        ),
        Case(
            "simulation sees overprinting in a form XObject",
            document(b"", b"1 0 0 0 k " + BACKDROP_RECT + b"/X10 Do", [form]),
            overprinted,
            "simulate",
        ),
    ]


# A cyan backdrop and magenta overprinting it under OPM 1, on a page without a group.
PDFX_CONTENT = b"1 0 0 0 k " + BACKDROP_RECT + b"/On1 gs 0 1 0 0 k " + PAINT_RECT + b" f"


def output_intent(subtype: bytes) -> bytes:
    """Catalog entries naming an output intent of subtype."""
    return (
        b"/OutputIntents [<< /Type /OutputIntent /S /"
        + subtype
        + b" /OutputConditionIdentifier (CGATS TR 001) >>] "
    )


def auto_cases() -> list[Case]:
    """auto simulates overprinting in a PDF/X document."""
    overprinted = regions(CYAN, MAGENTA, add(CYAN, MAGENTA, "m"))
    return [
        Case(
            "auto simulates a document with a PDF/X output intent",
            document(b"", PDFX_CONTENT, catalog=output_intent(b"GTS_PDFX")),
            overprinted,
            "auto",
        ),
        Case(
            "auto simulates a document whose Info names a PDF/X version",
            document(b"", PDFX_CONTENT, info=b"/GTS_PDFXVersion (PDF/X-1:2001) "),
            overprinted,
            "auto",
        ),
    ]


# The spot colorant Spot's alternate at tint 1 and at 0.4: c m y k of 0, t / 2, t, 0.
SPOT_FULL: Cmyk = (0.0, 0.5, 1.0, 0.0)
SPOT_LIGHT: Cmyk = (0.0, 0.2, 0.4, 0.0)
# A cyan backdrop, then Spot at tint 1, overprinting, over x 20 to 180: what a paint of
# Spot at 0.4 over the paint's rectangle replaces.
SPOT_BACKDROP = b"1 0 0 0 k " + BACKDROP_RECT + b"/On0 gs /Spot cs 1 sc 20 20 160 160 re f "
# The same over the whole page, for text, whose glyph covers the pixel sampled at 48, 100.
SPOT_PAGE = b"1 0 0 0 k 0 0 200 200 re f /On0 gs /Spot cs 1 sc 0 0 200 200 re f "
SPOT_TEXT_SAMPLES = (((48, 100), add(CYAN, SPOT_LIGHT, "")), ((190, 190), add(CYAN, SPOT_FULL, "")))
# A replaced spot: cyan and the full spot where only the backdrop is, cyan and the light one
# where the paint is over it, and the light one alone.
SPOT_REPLACED = regions(add(CYAN, SPOT_FULL, ""), SPOT_LIGHT, add(CYAN, SPOT_LIGHT, ""))
SPOT_TEXT = b"BT /F 150 Tf 30 50 Td (H) Tj ET"
SPOT_IMAGE = stream(
    b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 8 "
    b"/ColorSpace " + SPOT_SEPARATION,
    bytes([102]),
)
MAGENTA_IMAGE = stream(
    b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 8 "
    b"/ColorSpace /DeviceCMYK",
    bytes([0, 255, 0, 0]),
)


def constant_shading(space: bytes, values: bytes) -> bytes:
    """An axial shading in space whose function gives values everywhere."""
    return (
        b"<< /ShadingType 2 /ColorSpace "
        + space
        + b" /Coords [0 0 200 0] /Function << /FunctionType 2 /Domain [0 1] /C0 ["
        + values
        + b"] /C1 ["
        + values
        + b"] /N 1 >> >>"
    )


def spot_tiling(content: bytes, cell: bytes, *, group: bytes = CMYK_GROUP) -> bytes:
    """A page whose content fills with pattern T, a 40 by 40 tiling whose cell paints cell
    (object 10), with the page's graphics states and colour spaces."""
    pattern = stream(
        b"/PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 40 40] /XStep 40 /YStep 40 "
        b"/Resources << " + EXTGSTATES + SPACES + b">>",
        cell,
    )
    return document(group, content, [pattern], resources_extra=b"/Pattern << /T 10 0 R >> ")


def knockout_form(content: bytes, *, isolated: bool) -> bytes:
    """A form XObject whose DeviceCMYK knockout group paints content."""
    return stream(
        b"/Type /XObject /Subtype /Form /BBox [0 0 200 200] "
        b"/Group << /S /Transparency /CS /DeviceCMYK /K true /I "
        + (b"true" if isolated else b"false")
        + b" >> /Resources << "
        + EXTGSTATES
        + SPACES
        + b"/Font << /F << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>",
        content,
    )


def spot_replace_cases() -> list[Case]:
    """A spot colorant over the same spot replaces its tint, whatever paints it: "Spot
    colourant named in source space" takes Cs (Table 146)."""
    paint = b"/On0 gs "
    shading = constant_shading(SPOT_SEPARATION, b"0.4")
    return [
        Case(
            "a spot fill over the same spot replaces its tint",
            document(CMYK_GROUP, SPOT_BACKDROP + paint + b"0.4 sc " + PAINT_RECT + b" f"),
            SPOT_REPLACED,
        ),
        Case(
            "a spot stroke over the same spot replaces its tint",
            document(
                CMYK_GROUP, SPOT_BACKDROP + paint + b"/Spot CS 0.4 SC 80 w 60 100 m 180 100 l S"
            ),
            SPOT_REPLACED,
        ),
        Case(
            "spot text over the same spot replaces its tint",
            document(CMYK_GROUP, SPOT_PAGE + b"0.4 sc " + SPOT_TEXT),
            SPOT_TEXT_SAMPLES,
        ),
        Case(
            "spot text in a knockout group over the same spot replaces its tint",
            document(
                CMYK_GROUP,
                SPOT_PAGE + b"/X10 Do",
                [knockout_form(b"/On0 gs /Spot cs 0.4 sc " + SPOT_TEXT, isolated=False)],
            ),
            SPOT_TEXT_SAMPLES,
        ),
        Case(
            "a spot image over the same spot replaces its tint",
            document(CMYK_GROUP, SPOT_BACKDROP + paint + PLACE, [SPOT_IMAGE]),
            SPOT_REPLACED,
        ),
        Case(
            "a spot image mask over the same spot replaces its tint",
            document(CMYK_GROUP, SPOT_BACKDROP + paint + b"0.4 sc " + PLACE, [IMAGE_MASK]),
            SPOT_REPLACED,
        ),
        Case(
            "a spot shading over the same spot replaces its tint",
            with_shadings(
                document(CMYK_GROUP, SPOT_BACKDROP + paint + b"q " + PAINT_RECT + b" W n /Sh sh Q"),
                b"/Sh " + shading,
            ),
            SPOT_REPLACED,
        ),
        Case(
            "a spot shading pattern over the same spot replaces its tint",
            document(
                CMYK_GROUP,
                SPOT_BACKDROP + paint + b"/Pattern cs /P scn " + PAINT_RECT + b" f",
                [b"<< /PatternType 2 /Shading " + shading + b" >>"],
                resources_extra=b"/Pattern << /P 10 0 R >> ",
            ),
            SPOT_REPLACED,
        ),
        Case(
            "a spot tiling cell over the same spot replaces its tint",
            spot_tiling(
                SPOT_BACKDROP + b"/Pattern cs /T scn " + PAINT_RECT + b" f",
                b"/On0 gs /Spot cs 0.4 sc 0 0 40 40 re f",
            ),
            SPOT_REPLACED,
        ),
        Case(
            "a spot tiling cell's plate reaches the page, for the same spot to replace",
            spot_tiling(
                b"1 0 0 0 k " + BACKDROP_RECT + b"/Pattern cs /T scn 20 20 160 160 re f "
                b"/On0 gs /Spot cs 0.4 sc " + PAINT_RECT + b" f",
                b"/On0 gs /Spot cs 1 sc 0 0 40 40 re f",
            ),
            SPOT_REPLACED,
        ),
        Case(
            "B in a spot over the same spot replaces its tint",
            document(
                CMYK_GROUP, SPOT_BACKDROP + paint + b"0.4 sc /Spot CS 0.4 SC 10 w 60 60 120 80 re B"
            ),
            SPOT_REPLACED,
        ),
        Case(
            "a spot over the same spot painted without overprinting replaces its tint",
            document(
                CMYK_GROUP,
                b"1 0 0 0 k " + BACKDROP_RECT + b"/Off gs /Spot cs 1 sc 20 20 160 160 re f "
                b"/On0 gs 0.4 sc " + PAINT_RECT + b" f",
            ),
            # Without overprinting, the full spot erases the cyan under it.
            regions(SPOT_FULL, SPOT_LIGHT, SPOT_LIGHT),
        ),
        Case(
            "a spot over the same spot replaces its tint on a simulated page",
            document(b"", SPOT_BACKDROP + paint + b"0.4 sc " + PAINT_RECT + b" f"),
            SPOT_REPLACED,
            "simulate",
        ),
    ]


# A spot over the backdrop's rectangle, which an overprinting process paint keeps.
SPOT_UNDER = b"/On0 gs /Spot cs 1 sc " + BACKDROP_RECT


def spot_erase_cases() -> list[Case]:
    """A process paint that does not overprint erases a spot under it, whatever paints it
    (OP false: 0 for a spot colourant the source does not name, Table 146), and one that
    overprints keeps it (Cb), even when it marks every process component: under overprint
    mode 0, a gray level, an ICCBased CMYK colour."""
    erased = regions(SPOT_FULL, MAGENTA, MAGENTA)
    off = b"/Off gs "
    magenta_shading = constant_shading(b"/DeviceCMYK", b"0 1 0 0")
    kept = (0.0, 1.0, 1.0, 0.0)
    return [
        Case(
            "a process fill without overprinting erases a spot",
            document(CMYK_GROUP, SPOT_UNDER + off + b"0 1 0 0 k " + PAINT_RECT + b" f"),
            erased,
        ),
        Case(
            "a process stroke without overprinting erases a spot",
            document(CMYK_GROUP, SPOT_UNDER + off + b"0 1 0 0 K 80 w 60 100 m 180 100 l S"),
            erased,
        ),
        Case(
            "process text without overprinting erases a spot",
            document(
                CMYK_GROUP,
                b"/On0 gs /Spot cs 1 sc 0 0 200 200 re f " + off + b"0 1 0 0 k " + SPOT_TEXT,
            ),
            (((48, 100), MAGENTA), ((190, 190), SPOT_FULL)),
        ),
        Case(
            "a process image without overprinting erases a spot",
            document(CMYK_GROUP, SPOT_UNDER + off + PLACE, [MAGENTA_IMAGE]),
            erased,
        ),
        Case(
            "a process image mask without overprinting erases a spot",
            document(CMYK_GROUP, SPOT_UNDER + off + b"0 1 0 0 k " + PLACE, [IMAGE_MASK]),
            erased,
        ),
        Case(
            "a process shading without overprinting erases a spot",
            with_shadings(
                document(CMYK_GROUP, SPOT_UNDER + off + b"q " + PAINT_RECT + b" W n /Sh sh Q"),
                b"/Sh " + magenta_shading,
            ),
            erased,
        ),
        Case(
            "B without overprinting erases a spot",
            document(CMYK_GROUP, SPOT_UNDER + off + b"0 1 0 0 k 0 1 0 0 K 10 w 60 60 120 80 re B"),
            erased,
        ),
        Case(
            "B without overprinting erases a spot where shape is tracked",
            document(
                CMYK_GROUP,
                SPOT_UNDER + b"/X10 Do",
                [
                    knockout_form(
                        off + b"0 1 0 0 k 0 1 0 0 K 10 w 60 60 120 80 re B", isolated=False
                    )
                ],
            ),
            erased,
        ),
        Case(
            "overprint mode 0 marks every process component and keeps a spot",
            document(CMYK_GROUP, SPOT_UNDER + b"/On0 gs 0 1 0 0 k " + PAINT_RECT + b" f"),
            regions(SPOT_FULL, MAGENTA, kept),
        ),
        Case(
            "an overprinting gray level keeps a spot",
            document(CMYK_GROUP, SPOT_UNDER + b"/On0 gs 0.5 g " + PAINT_RECT + b" f"),
            regions(SPOT_FULL, (0.0, 0.0, 0.0, 0.5), (0.0, 0.5, 1.0, 0.5)),
        ),
        Case(
            "an overprinting ICCBased CMYK colour keeps a spot",
            document(CMYK_GROUP, SPOT_UNDER + b"/On0 gs /ICC cs 0 1 0 0 sc " + PAINT_RECT + b" f"),
            regions(SPOT_FULL, MAGENTA, kept),
        ),
    ]


def spot_cases() -> list[Case]:
    """A spot over another spot adds to it, and in a knockout group a paint knocks out a
    spot an earlier paint of the group left, though not the one under the group."""
    spot2 = (0.3, 0.0, 0.0, 0.0)
    half_magenta = (0.0, 0.5, 0.0, 0.0)
    return [
        *spot_replace_cases(),
        *spot_erase_cases(),
        Case(
            "a spot over another spot keeps both",
            document(
                CMYK_GROUP,
                b"/On0 gs /Spot cs 1 sc " + BACKDROP_RECT + b"/Spot2 cs 1 sc " + PAINT_RECT + b" f",
            ),
            regions(SPOT_FULL, spot2, add(SPOT_FULL, spot2, "")),
        ),
        Case(
            "a spot over a tiling cell's process paint adds to it",
            spot_tiling(
                b"/Pattern cs /T scn "
                + BACKDROP_RECT
                + b"/On0 gs /Spot cs 1 sc "
                + PAINT_RECT
                + b" f",
                b"/Off gs 1 0 0 0 k 0 0 40 40 re f",
            ),
            regions(CYAN, SPOT_FULL, add(CYAN, SPOT_FULL, "")),
        ),
        Case(
            "a half-opaque paint in an isolated knockout group knocks out a spot",
            document(
                CMYK_GROUP,
                # An opaque white backdrop, for the half-opaque paint to composite with in CMYK.
                b"0 0 0 0 k 0 0 200 200 re f /X10 Do",
                [
                    knockout_form(
                        b"/On0 gs /Spot cs 1 sc "
                        + BACKDROP_RECT
                        + b"/Half gs 0 1 0 0 k "
                        + PAINT_RECT
                        + b" f",
                        isolated=True,
                    )
                ],
            ),
            regions(SPOT_FULL, half_magenta, half_magenta),
        ),
        Case(
            "a half-opaque overprinting paint in a knockout group keeps the spot under it",
            document(
                CMYK_GROUP,
                b"/On0 gs /Spot cs 1 sc 20 20 160 160 re f /X10 Do",
                [knockout_form(b"/Half gs 0 1 0 0 k " + PAINT_RECT + b" f", isolated=False)],
            ),
            regions(SPOT_FULL, (0.0, 1.0, 1.0, 0.0), (0.0, 1.0, 1.0, 0.0)),
        ),
    ]


def spot_sames() -> list[Same]:
    """Pages that must render the same with a spot plate as with the process colour of the
    spot's alternate, in a strip at the bottom: a tiling cell overprinting the backdrop, in
    Normal and in Multiply, composites with it either way."""
    sames = []
    for state in (b"/On1", b"/Mul"):
        cell = state + b" gs 0 1 0 0 k 0 0 40 40 re f"
        content = b"0.5 0 0 0 k " + BACKDROP_RECT + b"%s 0 0 200 20 re f /Pattern cs /T scn "
        content += PAINT_RECT + b" f"
        sames.append(
            Same(
                f"a tiling cell in {state.decode()[1:]} overprints its backdrop beside a spot",
                spot_tiling(content.replace(b"%s", b"/On0 gs /Spot2 cs 1 sc"), cell),
                spot_tiling(content.replace(b"%s", b"/Off gs 0.3 0 0 0 k"), cell),
            )
        )
    return sames


def memory_failures() -> list[str]:
    """At 600 dpi, two small spot fills and spot text in a knockout group cost little more
    memory than the same paint in a process colorant: a spot plate is as large as the paint
    on it, not the page."""
    text = knockout_form(
        b"/On0 gs /Spot cs 0.4 sc BT /F 40 Tf 30 150 Td (Hi) Tj ET", isolated=False
    )
    content = (
        b"1 0 0 0 k " + BACKDROP_RECT + b"/On0 gs /Spot cs 1 sc 30 30 20 20 re f "
        b"/Spot2 cs 1 sc 120 120 20 20 re f /Off gs 0 1 0 0 k 35 35 5 5 re f /X10 Do"
    )
    process = content.replace(b"/Spot2 cs", b"/Mag cs").replace(b"/Spot cs", b"/Mag cs")
    peaks = []
    for page in (
        document(CMYK_GROUP, content, [text]),
        document(CMYK_GROUP, process, [text.replace(b"/Spot cs", b"/Mag cs")]),
    ):
        with PdfDocument(page) as pdf:
            rendered = pdf.pages[0].render(RenderOptions())
            tracemalloc.start()
            try:
                rendered.rasterize(scale=600 / 72)
                peaks.append(tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()
    side = round(SIZE * 600 / 72)
    # The plane of the page the fill erasing a plate records its alpha on, a float each,
    # and a quarter of the page's RGBA raster: less than a page-sized plate of either spot,
    # three bytes a pixel, would take.
    allowed = 4 * side * side + side * side
    if peaks[0] - peaks[1] > allowed:
        return [
            f"spot plates at 600 dpi: peak {peaks[0] / 1e6:.1f} MB against {peaks[1] / 1e6:.1f} "
            f"MB in process colorants, more than {allowed / 1e6:.1f} MB over"
        ]
    return []


def text_failures() -> list[str]:
    """Black text, 0 0 0 1 k under OPM 1, over cyan: where a glyph covers a pixel fully,
    its cyan stays under the black."""
    text = b"BT /F 150 Tf 30 50 Td (H) Tj ET"
    reference = render(document(CMYK_GROUP, b"0 0 0 1 k " + text))
    data = document(CMYK_GROUP, b"1 0 0 0 k 0 0 200 200 re f /On1 gs 0 0 0 1 k " + text)
    page = render(data)
    black = numpy.array(srgb((0, 0, 0, 1)))
    covered = numpy.all(reference == black, axis=2)
    if covered.sum() < 1000:
        return [f"text: only {int(covered.sum())} pixels fully covered"]
    expected = numpy.array(srgb((1, 0, 0, 1)))
    wrong = int((numpy.abs(page[covered] - expected).max(axis=1) > TOLERANCE).sum())
    if wrong:
        return [f"text: {wrong} of {int(covered.sum())} covered pixels are not cyan and black"]
    return []


# A cyan backdrop, then B in a state %s with a fill and stroke whose black is 0: under
# OPM 1 they overprint black alone, which is 0 in the backdrop too, so nothing changes, and
# in the CMY pass they mark every component.
FILL_STROKE = (
    b"1 0 0 0 k " + BACKDROP_RECT + b"%s gs 0.2 0.3 0.4 0 k 0.5 0.4 0.3 0 K 10 w 60 60 120 80 re B"
)


PLACE = b"q 120 0 0 80 60 60 cm /X10 Do Q"
# A 1 by 1 image in DeviceN Cyan and Magenta, through an RGB alternate whose tint transform,
# object 11, takes c m to 1 - c, 1 - m, 1.
CM_IMAGE = (
    stream(
        b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 8 "
        b"/ColorSpace [/DeviceN [/Cyan /Magenta] /DeviceRGB 11 0 R]",
        bytes([153, 102]),
    ),
    stream(
        b"/FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1]",
        b"{ 1 exch sub exch 1 exch sub exch 1 }",
    ),
)


def cyan_shading(state: bytes) -> bytes:
    """A Separation Cyan shading across the page, at tint 1, in a graphics state state."""
    return document(CMYK_GROUP, state + b" gs /Sh sh").replace(
        b"/XObject << >>",
        b"/XObject << >> /Shading << /Sh << /ShadingType 2 "
        b"/ColorSpace [/Separation /Cyan /DeviceCMYK "
        b"<< /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [1 0 0 0] /N 1 >>] "
        b"/Coords [0 0 200 0] "
        b"/Function << /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [1] /N 1 >> >> >>",
    )


def cyan_shading_failures() -> list[str]:
    """A Separation Cyan shading over white paints cyan, not black: in a DeviceCMYK group,
    through its alternate, and on a simulated page, as its tint directly."""
    failures = []
    data = cyan_shading(b"/On0")
    for where, page in (
        ("in a DeviceCMYK group", render(data)),
        (
            "on a simulated page",
            render(data.replace(CMYK_GROUP, b" " * len(CMYK_GROUP), 1), "simulate"),
        ),
    ):
        red, green, blue = (int(value) for value in page[100, 100])
        if red > 80 or blue < 200:
            failures.append(f"Separation Cyan shading {where}: {(red, green, blue)} is not cyan")
    return failures


def same_cases() -> list[Same]:
    return [
        Same(
            "RGB colour with overprint on changes nothing",
            two_fills(b"/On1 gs 1 0 0 rg"),
            two_fills(b"/Off gs 1 0 0 rg"),
        ),
        Same(
            "OPM 1 leaves no ICCBased CMYK component: it is no DeviceCMYK",
            two_fills(b"/On1 gs /ICC cs 0 1 0 0 sc"),
            two_fills(b"/Off gs /ICC cs 0 1 0 0 sc"),
        ),
        Same(
            "B with half alpha and a soft mask, overprinting nothing visible, paints as without",
            document(CMYK_GROUP, FILL_STROKE % b"/Masked"),
            document(CMYK_GROUP, FILL_STROKE % b"/Masked"),
            "groups",
            "ignore",
        ),
        Same(
            "an image mask in Separation Black paints as the same shape filled",
            document(
                CMYK_GROUP,
                b"1 0 0 0 k " + BACKDROP_RECT + b"/On0 gs /BlackGray cs 0.5 sc "
                b"q 120 0 0 80 60 60 cm /X10 Do Q",
                [IMAGE_MASK],
            ),
            two_fills(b"/On0 gs /BlackGray cs 0.5 sc"),
        ),
        Same(
            "OPM 1 leaves no component of DeviceCMYK remapped through DefaultCMYK",
            two_fills(b"/On1 gs 0 1 0 0 k").replace(
                b"/ColorSpace << ", b"/ColorSpace << /DefaultCMYK [/ICCBased 8 0 R] "
            ),
            two_fills(b"/Off gs 0 1 0 0 k").replace(
                b"/ColorSpace << ", b"/ColorSpace << /DefaultCMYK [/ICCBased 8 0 R] "
            ),
        ),
        Same(
            "DefaultCMYK whose Range leaves the native operators to k's handler, likewise",
            two_fills(b"/On1 gs 0 1 0 0 k").replace(
                b"/ColorSpace << ", b"/ColorSpace << /DefaultCMYK [/ICCBased 9 0 R] "
            ),
            two_fills(b"/Off gs 0 1 0 0 k").replace(
                b"/ColorSpace << ", b"/ColorSpace << /DefaultCMYK [/ICCBased 9 0 R] "
            ),
        ),
        Same(
            "an Indexed index past hival paints entry hival",
            two_fills(b"/On0 gs /IdxBlack cs 5 sc", backdrop=b"0.3 0 0 0 k"),
            two_fills(b"/On0 gs /IdxBlack cs 1 sc", backdrop=b"0.3 0 0 0 k"),
        ),
        Same(
            "auto renders a document that is not PDF/X as groups does",
            two_fills(b"/On1 gs 0 1 0 0 k", group=b""),
            two_fills(b"/On1 gs 0 1 0 0 k", group=b""),
            "auto",
            "groups",
        ),
        Same(
            "auto takes a PDF/A output intent for no PDF/X",
            document(b"", PDFX_CONTENT, catalog=output_intent(b"GTS_PDFA1")),
            document(b"", PDFX_CONTENT, catalog=output_intent(b"GTS_PDFA1")),
            "auto",
            "groups",
        ),
        Same(
            "gray with overprint on changes nothing",
            two_fills(b"/On1 gs 0.5 g"),
            two_fills(b"/Off gs 0.5 g"),
        ),
        Same(
            "an RGB page group ignores overprinting, even when simulating",
            two_fills(b"/On1 gs 0 1 0 0 k", group=RGB_GROUP),
            two_fills(b"/Off gs 0 1 0 0 k", group=RGB_GROUP),
            "simulate",
        ),
        Same(
            "a page without a group ignores overprinting unless simulating",
            two_fills(b"/On1 gs 0 1 0 0 k", group=b""),
            two_fills(b"/Off gs 0 1 0 0 k", group=b""),
        ),
        Same(
            "simulation leaves a page that does not overprint as it is",
            two_fills(b"/On1 gs 1 0 0 rg", group=b""),
            two_fills(b"/On1 gs 1 0 0 rg", group=b""),
            "simulate",
            "ignore",
        ),
    ]


def check_case(case: Case) -> list[str]:
    try:
        page = render(case.data, case.overprint)
    except Exception as error:  # noqa: BLE001 - a page that fails to render is a failure
        return [f"{case.name}: {type(error).__name__}: {error}"]
    failures = []
    for (x, y), cmyk in case.samples:
        expected = srgb(cmyk)
        actual = tuple(int(value) for value in page[y, x])
        if max(abs(a - e) for a, e in zip(actual, expected, strict=True)) > TOLERANCE:
            failures.append(f"{case.name}: at {x},{y} {actual} != {expected} (CMYK {cmyk})")
    if case.known:
        if not failures:
            return [f"{case.name}: a known difference now passes; make it an ordinary case"]
        print(f"known difference, {failures[0]}")
        return []
    return failures


def check_same(case: Same) -> list[str]:
    try:
        page = render(case.data, case.overprint)
        reference = render(case.reference, case.reference_overprint)
    except Exception as error:  # noqa: BLE001 - a page that fails to render is a failure
        return [f"{case.name}: {type(error).__name__}: {error}"]
    wrong = int(numpy.any(page != reference, axis=2).sum())
    return [f"{case.name}: {wrong} pixels differ"] if wrong else []


def mupdf_distances(cases: Sequence[Case]) -> None:
    """Print each page's mean difference from MuPDF's rendering with mutool draw's overprint
    simulation: a page that overprints is drawn on a CMYK pixmap and converted."""
    import pymupdf  # noqa: PLC0415  # ty: ignore[unresolved-import]
    from pymupdf import mupdf  # noqa: PLC0415  # ty: ignore[unresolved-import]

    for case in cases:
        document = pymupdf.open("pdf", case.data)
        page = mupdf.fz_load_page(document.this, 0)
        # As mutool draw does: the page's spot colorants as separations composited into
        # the process ones, or, on a page without them, none.
        separations = mupdf.fz_page_separations(page)
        if separations.m_internal:
            for index in range(mupdf.fz_count_separations(separations)):
                mupdf.fz_set_separation_behavior(separations, index, mupdf.FZ_SEPARATION_COMPOSITE)
        else:
            separations = mupdf.fz_new_separations(0)
        bbox = mupdf.fz_round_rect(mupdf.fz_bound_page(page))
        if case.overprint == "ignore":
            separations = mupdf.FzSeparations()
        pixmap = mupdf.fz_new_pixmap_with_bbox(mupdf.fz_device_rgb(), bbox, separations, 0)
        mupdf.fz_clear_pixmap_with_value(pixmap, 255)
        device = mupdf.fz_new_draw_device(mupdf.FzMatrix(), pixmap)
        mupdf.fz_run_page(page, device, mupdf.FzMatrix(), mupdf.FzCookie())
        mupdf.fz_close_device(device)
        stride = pixmap.stride()
        raw = bytes(mupdf.fz_samples_get(pixmap, i) for i in range(stride * pixmap.h()))
        theirs = numpy.frombuffer(raw, numpy.uint8).reshape(pixmap.h(), pixmap.w(), -1)[..., :3]
        ours = render(case.data, case.overprint)
        distance = float(numpy.abs(ours - theirs.astype(numpy.int16)).mean())
        print(f"  MuPDF {distance:6.2f}  {case.name}")


def all_cases() -> list[Case]:
    return [
        *fill_cases(),
        *(simulated(case) for case in fill_cases() if case.name in DEVICE_FILLS),
        *device_paint_cases(),
        *device_group_cases(),
        *stroke_cases(),
        masked_fill_stroke_case(),
        *group_cases(),
        *image_cases(),
        shading_case(),
        *process_shading_cases(),
        *map(simulated, process_shading_cases()),
        *process_image_cases(),
        *shading_pattern_cases(),
        *census_cases(),
        *auto_cases(),
        *spot_cases(),
    ]


def main(argv: list[str]) -> int:
    cases = all_cases()
    failures = []
    for case in cases:
        failures.extend(check_case(case))
    sames = [*same_cases(), *icc_group_sames(), *spot_sames()]
    for same in sames:
        failures.extend(check_same(same))
    failures.extend(text_failures())
    failures.extend(cyan_shading_failures())
    failures.extend(memory_failures())
    for failure in failures:
        print(failure)
    if "--mupdf" in argv:
        mupdf_distances(cases)
    print(f"{len(cases) + len(sames) + 2} overprint pages: {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

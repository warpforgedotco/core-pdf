# SPDX-License-Identifier: AGPL-3.0-only

"""Check overprinting against the colorant arithmetic of ISO 32000-2 8.6.7 and 11.7.4.

An overprinting paint marks only the process colorants its colour space names: DeviceCMYK
under overprint mode 1, given directly as the current colour of a fill, stroke, text or
image mask, marks its nonzero components; a Separation or DeviceN space marks the process
colorants it names, and a spot colorant none, its alternate colour's tints adding to the
backdrop's; any other space marks every component, so overprinting changes nothing
(Table 146). Core composites overprinting where paint lands in a DeviceCMYK blending space:
a transparency group or page group whose colour space is CMYK, and with
RenderOptions(overprint="simulate") a page without a group colour space too.

Each page here paints a backdrop and an overprinting paint over it, mostly in a DeviceCMYK
page group, and names the CMYK each sampled pixel must show, worked out here from the
colour values: a marked component takes the paint's tint, an unmarked one the backdrop's
plus the paint's, clamped at 1, and a non-opaque paint composites as Normal does. The
expected CMYK becomes sRGB through the built-in DeviceCMYK profile, as a CMYK group's colour
does, and each pixel must be within TOLERANCE of it. Other pages must render exactly as a
companion page does: overprinting RGB colour, an RGB page group, overprint off.

With --mupdf (and PyMuPDF installed: uv run --with pymupdf), each page is also rendered by
MuPDF with the overprint simulation mutool draw applies by default, and the mean difference
from core's rendering is printed for each page; it is reported, not checked, as MuPDF
composites spots and converts colour its own way.

uv run --all-packages python scripts/check_overprint.py [--mupdf]
"""

from __future__ import annotations

import sys
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


def document(page: bytes, content: bytes, extra: Sequence[bytes] = ()) -> bytes:
    """A one-page PDF: the page dictionary's entries, its content, and objects 9 onwards;
    objects 5 and 6 are the DeviceN spaces' tint transforms, 7 the soft mask's group, 8
    the ICC profile and 9 the same profile with a Range beyond 0 to 1, and the page's
    resources name the graphics states, colour spaces, a font F and the XObjects of extra as
    X10, X11..."""
    xobjects = b"".join(b"/X%d %d 0 R " % (number, number) for number in range(10, 10 + len(extra)))
    resources = (
        b"/Resources << "
        + EXTGSTATES
        + SPACES
        + b"/Font << /F << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> "
        + b"/XObject << "
        + xobjects
        + b">> >> "
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
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
    """A Separation Cyan shading over white paints cyan, not black."""
    page = render(cyan_shading(b"/On0"))
    red, green, blue = (int(value) for value in page[100, 100])
    if red > 80 or blue < 200:
        return [f"Separation Cyan shading: {(red, green, blue)} is not cyan"]
    return []


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
            "a Separation Cyan shading paints as with overprint off",
            cyan_shading(b"/On0"),
            cyan_shading(b"/Off"),
        ),
        Same(
            "a DeviceN Cyan and Magenta image paints as with overprint off",
            document(CMYK_GROUP, b"0 0 0 0.5 k 0 0 200 200 re f /On0 gs " + PLACE, CM_IMAGE),
            document(CMYK_GROUP, b"0 0 0 0.5 k 0 0 200 200 re f /Off gs " + PLACE, CM_IMAGE),
        ),
        Same(
            "an Indexed index past hival paints entry hival",
            two_fills(b"/On0 gs /IdxBlack cs 5 sc", backdrop=b"0.3 0 0 0 k"),
            two_fills(b"/On0 gs /IdxBlack cs 1 sc", backdrop=b"0.3 0 0 0 k"),
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
    page = render(case.data, case.overprint)
    failures = []
    for (x, y), cmyk in case.samples:
        expected = srgb(cmyk)
        actual = tuple(int(value) for value in page[y, x])
        if max(abs(a - e) for a, e in zip(actual, expected, strict=True)) > TOLERANCE:
            failures.append(f"{case.name}: at {x},{y} {actual} != {expected} (CMYK {cmyk})")
    return failures


def check_same(case: Same) -> list[str]:
    page = render(case.data, case.overprint)
    reference = render(case.reference, case.reference_overprint)
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


def main(argv: list[str]) -> int:
    cases = [
        *fill_cases(),
        *stroke_cases(),
        masked_fill_stroke_case(),
        *group_cases(),
        *image_cases(),
        shading_case(),
    ]
    failures = []
    for case in cases:
        failures.extend(check_case(case))
    sames = same_cases()
    for same in sames:
        failures.extend(check_same(same))
    failures.extend(text_failures())
    failures.extend(cyan_shading_failures())
    for failure in failures:
        print(failure)
    if "--mupdf" in argv:
        mupdf_distances(cases)
    print(f"{len(cases) + len(sames) + 1} overprint pages: {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

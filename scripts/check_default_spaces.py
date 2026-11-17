# SPDX-License-Identifier: AGPL-3.0-only

"""Check that device colour is remapped through DefaultGray, DefaultRGB and DefaultCMYK.

When the ColorSpace subdictionary of the current resource dictionary has a DefaultGray,
DefaultRGB or DefaultCMYK entry, colour in the corresponding device space is painted in
that space instead, with its components passed unchanged (ISO 32000-2 8.6.5.6): whether the
device space is selected by g, rg, k and their stroking forms, by cs naming it, as the base
of an Indexed space, the alternate of a Separation or the underlying space of a Pattern
space, or by the ColorSpace of an image XObject, an inline image or a shading. The current
resource dictionary is that of the content stream doing the painting, so a form XObject with
its own resources uses its own default spaces, not the page's. A transparency group's CS
naming a device space is remapped too (11.6.6, Table 145), as is the page group's.

Each case here is a page that paints device colour under default spaces, and the same page
written with the CIE-based space named explicitly in place of the device space. The two must
render the same pixels, within 2 of 255; and, so that each case shows the remapping, the
page painted without the default spaces must differ from the explicit one. The default
spaces are CalGray and CalRGB spaces with gamma 1, whose colours are far lighter than the
device colours, and an ICCBased CMYK space built here whose profile is a naive conversion,
far from the built-in DeviceCMYK profile.

uv run --all-packages python scripts/check_default_spaces.py
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
TOLERANCE = 2
# The share of pixels the device rendering must change for a case to show the remapping.
DISCRIMINATING_SHARE = 0.05

CALRGB = (
    b"[/CalRGB << /WhitePoint [0.9505 1 1.089] /Gamma [1 1 1] "
    b"/Matrix [0.4124 0.2126 0.0193 0.3576 0.7152 0.1192 0.1805 0.0722 0.9505] >>]"
)
# A second RGB space, its primaries rotated, for a form's own resources.
CALRGB_ROTATED = (
    b"[/CalRGB << /WhitePoint [0.9505 1 1.089] /Gamma [1.8 1.8 1.8] "
    b"/Matrix [0.3576 0.7152 0.1192 0.1805 0.0722 0.9505 0.4124 0.2126 0.0193] >>]"
)
CALGRAY = b"[/CalGray << /WhitePoint [0.9505 1 1.089] /Gamma 1 >>]"
# The ICCBased CMYK space's profile is object 5 on every page that uses it.
ICC_CMYK = b"[/ICCBased 5 0 R]"


def s15_fixed16(value: float) -> bytes:
    return struct.pack(">i", round(value * 65536))


def srgb_to_lab(red: float, green: float, blue: float) -> tuple[float, float, float]:
    """CIELAB under D50 of an sRGB colour, through the Bradford-adapted sRGB matrix."""

    def linear(value: float) -> float:
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = linear(red), linear(green), linear(blue)
    x = (0.4360747 * r + 0.3850649 * g + 0.1430804 * b) / 0.9642
    y = 0.2225045 * r + 0.7168786 * g + 0.0606169 * b
    z = (0.0139322 * r + 0.0971045 * g + 0.7141733 * b) / 0.8249

    def f(value: float) -> float:
        return value ** (1 / 3) if value > 216 / 24389 else (24389 / 27 * value + 16) / 116

    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


def naive_cmyk_profile() -> bytes:
    """An ICC version 2 CMYK output profile whose A2B0 is a lut8 of the naive conversion,
    R = (1 - C)(1 - K) and so on, on a grid of the 16 corners of the CMYK cube."""
    clut = bytearray()
    for c, m, y, k in itertools.product((0, 1), repeat=4):
        lightness, a, b = srgb_to_lab((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
        clut += bytes((round(lightness / 100 * 255), round(a + 128), round(b + 128)))
    identity = bytes(range(256))
    lut = (
        b"mft1\0\0\0\0"
        + bytes((4, 3, 2, 0))
        + b"".join(s15_fixed16(value) for value in (1, 0, 0, 0, 1, 0, 0, 0, 1))
        + identity * 4
        + bytes(clut)
        + identity * 3
    )
    white = s15_fixed16(0.9642) + s15_fixed16(1.0) + s15_fixed16(0.8249)
    text = b"naive CMYK\0"
    tags = [
        (b"desc", b"desc\0\0\0\0" + struct.pack(">I", len(text)) + text + b"\0" * 79),
        (b"wtpt", b"XYZ \0\0\0\0" + white),
        (b"cprt", b"text\0\0\0\0none\0"),
        (b"A2B0", lut),
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
        + b"prtrCMYKLab "
        + b"\0" * 12
        + b"acsp"
        + b"\0" * 24
        + struct.pack(">I", 0)
        + white
        + b"\0" * 48
    )
    return header + table + body


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


def icc_object() -> bytes:
    return stream(b"/N 4", naive_cmyk_profile())


class Case(NamedTuple):
    """A page painted three ways: under default spaces, with the CIE-based spaces named
    explicitly, and with neither (device colour as it is)."""

    name: str
    remapped: bytes
    explicit: bytes
    device: bytes


def defaults(**spaces: bytes) -> bytes:
    return b"".join(b"/Default%s %s " % (name.encode(), value) for name, value in spaces.items())


def fill_case() -> Case:
    content = b"0.2 0.5 0.8 rg 20 20 160 70 re f 0.9 0.3 0.1 RG 16 w 30 140 m 170 140 l S"
    explicit = (
        b"/C cs 0.2 0.5 0.8 sc 20 20 160 70 re f /C CS 0.9 0.3 0.1 SC 16 w 30 140 m 170 140 l S"
    )
    extra = [icc_object()]
    return Case(
        "rg and RG under DefaultRGB",
        document(
            b"/Resources << /ColorSpace << %s>> >> /Contents 4 0 R" % defaults(RGB=CALRGB),
            content,
            extra,
        ),
        document(
            b"/Resources << /ColorSpace << /C %s >> >> /Contents 4 0 R" % CALRGB, explicit, extra
        ),
        document(b"/Contents 4 0 R", content, extra),
    )


def gray_cmyk_case() -> Case:
    content = (
        b"0.4 g 10 10 50 180 re f 0.6 G 8 w 35 20 m 35 180 l S "
        b"0.1 0.7 0.3 0 k 70 10 120 85 re f 0.6 0.1 0.2 0.1 K 8 w 80 150 m 180 150 l S"
    )
    explicit = (
        b"/G cs 0.4 sc 10 10 50 180 re f /G CS 0.6 SC 8 w 35 20 m 35 180 l S "
        b"/K cs 0.1 0.7 0.3 0 sc 70 10 120 85 re f "
        b"/K CS 0.6 0.1 0.2 0.1 SC 8 w 80 150 m 180 150 l S"
    )
    extra = [icc_object()]
    return Case(
        "g, G, k and K under DefaultGray and DefaultCMYK",
        document(
            b"/Resources << /ColorSpace << %s>> >> /Contents 4 0 R"
            % defaults(Gray=CALGRAY, CMYK=ICC_CMYK),
            content,
            extra,
        ),
        document(
            b"/Resources << /ColorSpace << /G %s /K %s >> >> /Contents 4 0 R" % (CALGRAY, ICC_CMYK),
            explicit,
            extra,
        ),
        document(b"/Contents 4 0 R", content, extra),
    )


def named_case() -> Case:
    content = (
        b"/DeviceRGB cs 0.2 0.5 0.8 sc 10 10 85 180 re f "
        b"/DeviceCMYK cs 105 10 85 85 re f /DeviceGray CS 0.3 SC 10 w 110 150 m 190 150 l S"
    )
    explicit = (
        b"/C cs 0.2 0.5 0.8 sc 10 10 85 180 re f "
        b"/K cs 0 0 0 1 sc 105 10 85 85 re f /G CS 0.3 SC 10 w 110 150 m 190 150 l S"
    )
    extra = [icc_object()]
    return Case(
        "cs naming a device space, and DeviceCMYK's initial black",
        document(
            b"/Resources << /ColorSpace << %s>> >> /Contents 4 0 R"
            % defaults(RGB=CALRGB, CMYK=ICC_CMYK, Gray=CALGRAY),
            content,
            extra,
        ),
        document(
            b"/Resources << /ColorSpace << /C %s /K %s /G %s >> >> /Contents 4 0 R"
            % (CALRGB, ICC_CMYK, CALGRAY),
            explicit,
            extra,
        ),
        document(b"/Contents 4 0 R", content, extra),
    )


def special_spaces(base: bytes) -> bytes:
    return (
        b"/I [/Indexed %s 1 <3366CCE64D1A>] " % base
        + b"/S [/Separation /Spot %s << /FunctionType 2 /Domain [0 1] /C0 [1 1 1] " % base
        + b"/C1 [0.1 0.6 0.3] /N 1 >>] "
        + b"/P [/Pattern %s] " % base
    )


def special_case() -> Case:
    content = (
        b"/I cs 1 sc 10 10 55 180 re f /S cs 0.8 sc 72 10 55 180 re f "
        b"/P cs 0.2 0.5 0.8 /T scn 135 10 55 180 re f"
    )
    tiling = stream(
        b"/Type /Pattern /PatternType 1 /PaintType 2 /TilingType 1 /BBox [0 0 10 10] "
        b"/XStep 10 /YStep 10 /Resources << >>",
        b"0 0 10 10 re f",
    )
    extra = [icc_object(), tiling]
    resources = b"/Resources << /Pattern << /T 6 0 R >> /ColorSpace << %s%s>> >> /Contents 4 0 R"
    return Case(
        "the bases of Indexed and Pattern spaces and a Separation's alternate",
        document(resources % (defaults(RGB=CALRGB), special_spaces(b"/DeviceRGB")), content, extra),
        document(resources % (b"", special_spaces(CALRGB)), content, extra),
        document(resources % (b"", special_spaces(b"/DeviceRGB")), content, extra),
    )


def form_case() -> Case:
    """The page's DefaultRGB, a form whose own resources have another, a form with its own
    resources and none, and a form with no resources, which takes the page's."""
    strip = b"0.2 0.5 0.8 rg %d 10 40 180 re f"
    explicit_strip = b"/C cs 0.2 0.5 0.8 sc %d 10 40 180 re f"

    def objects(own: bytes, empty: bytes, inherit: bytes) -> list[bytes]:
        form = b"/Type /XObject /Subtype /Form /BBox [0 0 200 200] "
        return [
            icc_object(),
            stream(form + b"/Resources << /ColorSpace << %s >> >>" % own, strip_of(1)),
            stream(form + b"/Resources << /ProcSet [/PDF] >>", empty),
            stream(form, inherit),
        ]

    def strip_of(index: int, *, explicit: bool = False) -> bytes:
        return (explicit_strip if explicit else strip) % (10 + 47 * index)

    page = (
        b"/Resources << /ColorSpace << %s >> /XObject << /F1 6 0 R /F2 7 0 R /F3 8 0 R >> >> "
        b"/Contents 4 0 R"
    )
    content = strip_of(0) + b" /F1 Do /F2 Do /F3 Do"
    remapped = objects(b"/DefaultRGB " + CALRGB_ROTATED, strip_of(2), strip_of(3))
    explicit = objects(b"/C " + CALRGB_ROTATED, strip_of(2), strip_of(3, explicit=True))
    explicit[1] = explicit[1].replace(strip_of(1), strip_of(1, explicit=True))
    return Case(
        "a form's own resources scope its default spaces",
        document(page % (b"/DefaultRGB " + CALRGB), content, remapped),
        document(
            page % (b"/C " + CALRGB),
            strip_of(0, explicit=True) + b" /F1 Do /F2 Do /F3 Do",
            explicit,
        ),
        document(page % b"", content, objects(b"", strip_of(2), strip_of(3))),
    )


def image_samples(components: int, seed: int) -> bytes:
    random = numpy.random.default_rng(seed)
    return bytes(random.integers(0, 256, (4, 4, components), dtype=numpy.uint8).tobytes())


def image_case() -> Case:
    rgb = image_samples(3, 1)
    gray = image_samples(1, 2)
    indices = bytes(numpy.random.default_rng(3).integers(0, 2, 16, dtype=numpy.uint8))
    cmyk = image_samples(4, 4)

    def images(rgb_space: bytes, gray_space: bytes, cmyk_space: bytes) -> list[bytes]:
        head = b"/Type /XObject /Subtype /Image /Width 4 /Height 4 /BitsPerComponent 8 "
        return [
            icc_object(),
            stream(head + b"/ColorSpace " + rgb_space, rgb),
            stream(head + b"/ColorSpace " + gray_space, gray),
            stream(head + b"/ColorSpace [/Indexed %s 1 <3366CCE64D1A>]" % rgb_space, indices),
            stream(head + b"/ColorSpace " + cmyk_space, cmyk),
        ]

    content = (
        b"q 90 0 0 90 5 105 cm /A Do Q q 90 0 0 90 105 105 cm /B Do Q "
        b"q 90 0 0 90 5 5 cm /C Do Q q 90 0 0 90 105 5 cm /D Do Q"
    )
    page = b"/Resources << %s/XObject << /A 6 0 R /B 7 0 R /C 8 0 R /D 9 0 R >> >> /Contents 4 0 R"
    color_spaces = b"/ColorSpace << %s>> " % defaults(RGB=CALRGB, Gray=CALGRAY, CMYK=ICC_CMYK)
    return Case(
        "image XObjects in device spaces and an Indexed space on one",
        document(
            page % color_spaces, content, images(b"/DeviceRGB", b"/DeviceGray", b"/DeviceCMYK")
        ),
        document(page % b"", content, images(CALRGB, CALGRAY, ICC_CMYK)),
        document(page % b"", content, images(b"/DeviceRGB", b"/DeviceGray", b"/DeviceCMYK")),
    )


def inline_case() -> Case:
    data = image_samples(3, 5).hex().upper().encode() + b">"
    gray = image_samples(1, 6).hex().upper().encode() + b">"

    def content(rgb: bytes, gray_space: bytes) -> bytes:
        return b"q 180 0 0 85 10 105 cm BI /W 4 /H 4 /CS %s /BPC 8 /F /AHx ID %s EI Q " % (
            rgb,
            data,
        ) + b"q 180 0 0 85 10 10 cm BI /W 4 /H 4 /CS %s /BPC 8 /F /AHx ID %s EI Q" % (
            gray_space,
            gray,
        )

    extra = [icc_object()]
    return Case(
        "inline images in abbreviated device spaces",
        document(
            b"/Resources << /ColorSpace << %s>> >> /Contents 4 0 R"
            % defaults(RGB=CALRGB, Gray=CALGRAY),
            content(b"/RGB", b"/G"),
            extra,
        ),
        document(
            b"/Resources << /ColorSpace << /C %s /K %s >> >> /Contents 4 0 R" % (CALRGB, CALGRAY),
            content(b"/C", b"/K"),
            extra,
        ),
        document(b"/Contents 4 0 R", content(b"/RGB", b"/G"), extra),
    )


def shading_case() -> Case:
    def shading(space: bytes) -> bytes:
        return (
            b"<< /ShadingType 2 /ColorSpace %s /Coords [0 0 200 0] /Extend [true true] " % space
            + b"/Function << /FunctionType 2 /Domain [0 1] /C0 [0.9 0.2 0.1] /C1 [0.1 0.4 0.9] "
            + b"/N 1 >> >>"
        )

    def objects(space: bytes) -> list[bytes]:
        return [icc_object(), shading(space), b"<< /Type /Pattern /PatternType 2 /Shading 6 0 R >>"]

    page = b"/Resources << %s/Shading << /Sh 6 0 R >> /Pattern << /P 7 0 R >> >> /Contents 4 0 R"
    content = b"q 0 0 200 95 re W n /Sh sh Q /Pattern cs /P scn 0 105 200 95 re f"
    return Case(
        "an axial shading by sh and as a shading pattern",
        document(
            page % (b"/ColorSpace << %s>> " % defaults(RGB=CALRGB)), content, objects(b"/DeviceRGB")
        ),
        document(page % b"", content, objects(CALRGB)),
        document(page % b"", content, objects(b"/DeviceRGB")),
    )


def group_case() -> Case:
    """An isolated group whose CS is DeviceCMYK, under its own resources' DefaultCMYK: the
    group composites in, and converts from, the ICCBased space."""
    inside = b"0.1 0.7 0.3 0 k 20 20 160 160 re f"
    explicit_inside = b"/K cs 0.1 0.7 0.3 0 sc 20 20 160 160 re f"

    def objects(group_space: bytes, resources: bytes, body: bytes) -> list[bytes]:
        return [
            icc_object(),
            stream(
                b"/Type /XObject /Subtype /Form /BBox [0 0 200 200] "
                b"/Group << /S /Transparency /I true /CS %s >> /Resources << %s >>"
                % (group_space, resources),
                body,
            ),
        ]

    page = b"/Resources << /XObject << /G 6 0 R >> >> /Contents 4 0 R"
    content = b"/G Do"
    return Case(
        "a transparency group whose CS is a remapped device space",
        document(
            page,
            content,
            objects(b"/DeviceCMYK", b"/ColorSpace << /DefaultCMYK %s >>" % ICC_CMYK, inside),
        ),
        document(
            page, content, objects(ICC_CMYK, b"/ColorSpace << /K %s >>" % ICC_CMYK, explicit_inside)
        ),
        document(page, content, objects(b"/DeviceCMYK", b"", inside)),
    )


def page_group_case() -> Case:
    content = b"0.1 0.7 0.3 0 k 20 20 160 160 re f"
    explicit = b"/K cs 0.1 0.7 0.3 0 sc 20 20 160 160 re f"
    group = b"/Group << /S /Transparency /CS %s >> "
    extra = [icc_object()]
    return Case(
        "the page group's CS, a remapped device space",
        document(
            group % b"/DeviceCMYK"
            + b"/Resources << /ColorSpace << /DefaultCMYK %s >> >> /Contents 4 0 R" % ICC_CMYK,
            content,
            extra,
        ),
        document(
            group % ICC_CMYK
            + b"/Resources << /ColorSpace << /K %s >> >> /Contents 4 0 R" % ICC_CMYK,
            explicit,
            extra,
        ),
        document(group % b"/DeviceCMYK" + b"/Contents 4 0 R", content, extra),
    )


def annotation_case() -> Case:
    def objects(resources: bytes, body: bytes) -> list[bytes]:
        return [
            icc_object(),
            b"<< /Type /Annot /Subtype /Square /Rect [20 20 180 180] /AP << /N 7 0 R >> >>",
            stream(
                b"/Type /XObject /Subtype /Form /BBox [0 0 160 160] /Resources << %s >>"
                % resources,
                body,
            ),
        ]

    body = b"0.2 0.5 0.8 rg 0 0 160 160 re f"
    explicit = b"/C cs 0.2 0.5 0.8 sc 0 0 160 160 re f"
    page = b"/Annots [6 0 R] /Contents 4 0 R"
    return Case(
        "an annotation appearance's own resources",
        document(page, b"", objects(b"/ColorSpace << /DefaultRGB %s >>" % CALRGB, body)),
        document(page, b"", objects(b"/ColorSpace << /C %s >>" % CALRGB, explicit)),
        document(page, b"", objects(b"", body)),
    )


CASES: tuple[Callable[[], Case], ...] = (
    fill_case,
    gray_cmyk_case,
    named_case,
    special_case,
    form_case,
    image_case,
    inline_case,
    shading_case,
    group_case,
    page_group_case,
    annotation_case,
)


def render(data: bytes) -> numpy.ndarray:
    """The page's pixels over white, as RGB."""
    with PdfDocument(data) as pdf:
        image = pdf.pages[0].render().rasterize()
    pixels = numpy.frombuffer(bytes(image.pixels), numpy.uint8).reshape(
        image.height, image.width, 4
    )
    rgb = pixels[..., :3].astype(numpy.float64)
    alpha = pixels[..., 3:].astype(numpy.float64) / 255.0
    return numpy.rint(rgb * alpha + 255.0 * (1.0 - alpha)).astype(numpy.int16)


def check(case: Case) -> list[str]:
    remapped = render(case.remapped)
    explicit = render(case.explicit)
    device = render(case.device)
    failures = []
    difference = numpy.abs(remapped - explicit).max(axis=2)
    wrong = int((difference > TOLERANCE).sum())
    if wrong:
        row, column = numpy.argwhere(difference > TOLERANCE)[0]
        failures.append(
            f"{case.name}: {wrong} pixels differ from the explicit space's, as at "
            f"{column},{row}: {tuple(int(v) for v in remapped[row, column])} != "
            f"{tuple(int(v) for v in explicit[row, column])}"
        )
    changed = float((numpy.abs(device - explicit).max(axis=2) > 8).mean())
    if changed < DISCRIMINATING_SHARE:
        failures.append(
            f"{case.name}: the device rendering changes only {changed:.1%} of the pixels, "
            "so the case does not show the remapping"
        )
    return failures


def main() -> int:
    failures = []
    for build in CASES:
        failures.extend(check(build()))
    for failure in failures:
        print(failure)
    print(f"{len(CASES)} default colour space pages: {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

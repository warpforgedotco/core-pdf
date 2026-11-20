# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 11.7.4.3: which process colorants an overprinting paint marks.

Painting with the overprint parameter true marks only the colorants its colour space names
and leaves the others as they were (8.6.7); in a DeviceCMYK blending space, the special
blend mode of 11.7.4.3 takes the source's value for the components the source colour space
specifies and the backdrop's for the rest, which Table 146 sums up:

- DeviceCMYK, specified directly by the current colour of a fill, stroke, text or image
  mask, under overprint mode 1: the nonzero components replace the backdrop's and the zero
  ones leave it. Under overprint mode 0, in a sampled image or a shading, or remapped
  through a default space, it marks all four.
- Separation and DeviceN: the colorants they name. A process colorant name (Cyan, Magenta,
  Yellow, Black) marks that process component alone; any other name is a spot colorant,
  which marks no process component, and the separation All marks every component.
- Any other colour space -- DeviceGray, DeviceRGB, CIE-based, ICCBased, Indexed over one
  of those -- marks every process component, so overprinting changes nothing.

Nonzero overprint mode applies to DeviceCMYK alone, not to an ICCBased CMYK space: both
editions limit it to "the current colour space is DeviceCMYK (or is implicitly converted to
DeviceCMYK; see 8.6.5.7)", an implicit conversion of a CIE-based space whose conditions
"cannot be specified in PDF" and which core never makes, and Table 146 gives "any process
colour space (including other cases of DeviceCMYK)" the source's value in every component.

A paint's marks are a mask of the process components, CYAN | MAGENTA | YELLOW | BLACK, with
SPOT when its space also names a spot colorant, or None when it marks all four. A process
colorant named in a Separation or DeviceN space is painted directly: its tint is that
colorant's value (8.6.6.4), whatever the alternate space, and a component the space does not
name keeps the backdrop's exactly. A sampled image's or a shading's colours reach the
raster through the space's alternate, not as the colorants' tints, so an image or a shading
in a space naming process colorants alone does not overprint: it marks every component and
paints as it would with overprint off (raw_overprint_marks). A
DeviceCMYK colour a DefaultCMYK space remaps is no DeviceCMYK colour: "Provisions in this
standard that apply specifically to device colour spaces are then not applicable" (8.6.5.6),
so its marks are those of the default space. A spot colorant, which a CMYK raster has no plate for,
adds its alternate colour's tints to the backdrop's, clamped at full, as a press's separate
plate of it would print over the process inks (the composite MuPDF's overprint simulation
makes of spot plates).
"""

from __future__ import annotations

from collections.abc import Sequence

from core_pdf.impl.base.values import recover_pdf_name
from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_08_graphics.color_spec import ColorSpace

CYAN = 1
MAGENTA = 2
YELLOW = 4
BLACK = 8
ALL_PROCESS = CYAN | MAGENTA | YELLOW | BLACK
# Set in a paint's marks when its space names a spot colorant: the components it does not
# mark take its alternate colour's tints added to the backdrop's, rather than keeping it.
SPOT = 16
PROCESS_COLORANTS: dict[str, int] = {
    "Cyan": CYAN,
    "Magenta": MAGENTA,
    "Yellow": YELLOW,
    "Black": BLACK,
}
# The marks of each CMYK component in order.
CMYK_MARKS = (CYAN, MAGENTA, YELLOW, BLACK)


def colorant_marks(kind: str, names: Sequence[str]) -> int | None:
    """The process components a Separation or DeviceN space naming names marks: each
    process colorant it names, and none for a spot colorant or None (8.6.6.4, 8.6.6.5), with
    SPOT for a spot colorant; the separation All marks every component."""
    if kind == "Separation" and tuple(names) == ("All",):
        return None
    marks = 0
    for name in names:
        if name in PROCESS_COLORANTS:
            marks |= PROCESS_COLORANTS[name]
        elif name != "None":
            marks |= SPOT
    return None if marks & ALL_PROCESS == ALL_PROCESS else marks


@cites(
    "ISO 32000-2",
    "11.7.4.5",
    introduced="1.3",
    rule="under overprint mode 1 a zero DeviceCMYK component of the current colour keeps the "
    "backdrop's",
)
@cites(
    "ISO 32000-2",
    "8.6.7",
    introduced="1.2",
    rule="an overprinting paint leaves the colorants its colour space does not name",
)
def overprint_marks(
    space: ColorSpace | None, color: Sequence[float] | None, mode: int, *, current_color: bool
) -> int | None:
    """The process components a paint in space marks while overprinting, or None for all
    four. current_color: the paint uses the graphics state's current colour, color (a fill,
    stroke, text or image mask), whose zero DeviceCMYK components overprint mode 1 leaves;
    an image or a shading does not (8.6.7), nor an ICCBased CMYK space."""
    while space is not None and space.kind == "Indexed":
        space = space.base
    if space is None:
        return None
    kind = space.kind
    if kind == "DeviceCMYK":
        if mode != 1 or not current_color or color is None or len(color) != 4:
            return None
        marks = 0
        for bit, component in zip(CMYK_MARKS, color, strict=True):
            if component != 0:
                marks |= bit
        return None if marks == ALL_PROCESS else marks
    if kind in {"Separation", "DeviceN"}:
        return colorant_marks(kind, space.colorants)
    return None


def process_color(space: ColorSpace, color: Sequence[float]) -> tuple[float, ...] | None:
    """The CMYK a Separation or DeviceN colour naming process colorants alone paints
    directly: each named colorant's tint, and 0 for the others (8.6.6.4); an Indexed colour
    over such a space, its entry's. None for any other space or colour."""
    if space.kind == "Indexed":
        base, lookup = space.base, space.lookup
        if base is None or lookup is None or len(color) != 1:
            return None
        count = len(base.component_ranges)
        # An index is rounded and clamped to 0 to hival (8.6.6.3).
        start = min(max(round(color[0]), 0), space.hival) * count
        entry = lookup[start : start + count]
        if count == 0 or len(entry) != count:
            return None
        color = [
            low + value / 255.0 * (high - low)
            for value, (low, high) in zip(entry, base.component_ranges, strict=True)
        ]
        space = base
    if space.kind not in {"Separation", "DeviceN"} or len(color) != len(space.colorants):
        return None
    cmyk = [0.0, 0.0, 0.0, 0.0]
    for name, tint in zip(space.colorants, color, strict=True):
        if name in PROCESS_COLORANTS:
            cmyk[CMYK_MARKS.index(PROCESS_COLORANTS[name])] = float(tint)
        elif name != "None":
            return None
    return tuple(cmyk)


def spot_marks(marks: int | None) -> int | None:
    """marks when they name a spot colorant, otherwise None: the marks of an image or a
    shading (raw_overprint_marks)."""
    return marks if marks is not None and marks & SPOT else None


def raw_overprint_marks(value: object) -> int | None:
    """overprint_marks of an image's or a shading's ColorSpace, a resolved PDF object: no
    overprint mode applies to them. Their colours reach a CMYK raster through the space's
    alternate, not as the colorants' tints, so only a space that names a spot colorant
    overprints, its alternate's tints standing for the spot; one naming process colorants
    alone marks every component (None), as its alternate's conversion would put ink in the
    components it does not name."""
    seen: set[int] = set()
    while isinstance(value, (list, tuple)) and value and id(value) not in seen:
        seen.add(id(value))
        kind = recover_pdf_name(value[0])
        if kind == "Indexed" and len(value) == 4:
            value = value[1]
            continue
        if kind == "Separation" and len(value) == 4:
            name = recover_pdf_name(value[1])
            return None if name is None else spot_marks(colorant_marks(kind, (name,)))
        if kind == "DeviceN" and len(value) in {4, 5} and isinstance(value[1], (list, tuple)):
            names = [recover_pdf_name(entry) for entry in value[1]]
            if not names or any(name is None for name in names):
                return None
            return spot_marks(colorant_marks(kind, [name for name in names if name is not None]))
        return None
    return None

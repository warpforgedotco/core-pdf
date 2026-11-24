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
  which marks no process component, and the separation All marks every component. A space
  may name process colorants as such only where paint lands in the output device's own
  colour space: "within a transparency group, this should be done only if the group
  inherits the native colour space of the output device ... If any other colour space has
  been specified for the group, the Separation or DeviceN colour space shall be converted
  to its alternate colour space" (11.7.3). Core renders for a CMYK device whose native
  process space is DeviceCMYK, so (our reading) a group whose resolved colour space is
  DeviceCMYK itself, or where no group names one, takes the colorants as named; inside a
  group whose space is any other -- ICCBased CMYK, CalRGB, Lab -- such a space is its
  alternate -- "likewise for Separation and DeviceN spaces that revert to their alternate
  colour space" (11.7.4.3, NOTE 2) -- a process colour space reached through a tint
  transform, not "DeviceCMYK, specified directly", which marks every component (Table
  146); nonzero overprint mode does not apply either, as the current colour space is not
  DeviceCMYK (8.6.7). A DeviceN space naming a spot as well converts as a whole.
- Any other colour space -- DeviceGray, DeviceRGB, CIE-based, ICCBased, Indexed over one
  of those -- marks every process component, so overprinting changes nothing.

Nonzero overprint mode applies to DeviceCMYK alone, not to an ICCBased CMYK space: both
editions limit it to "the current colour space is DeviceCMYK (or is implicitly converted to
DeviceCMYK; see 8.6.5.7)", an implicit conversion of a CIE-based space whose conditions
"cannot be specified in PDF" and which core never makes, and Table 146 gives "any process
colour space (including other cases of DeviceCMYK)" the source's value in every component.

A paint's marks are a mask of the process components, CYAN | MAGENTA | YELLOW | BLACK, with
SPOT and the number of the plate its spot colorants stand on (spot_plate) when its space also
names a spot colorant. ALL_PROCESS alone is a paint overprinting with every process component
marked, which keeps the spot plates under it ("Spot colourant" with OP true is Cb, Table
146), and None a paint that does not overprint, which erases them (0 with OP false); a paint
naming a spot colorant that does not overprint has OFF too, and replaces its own spots' plate
while it erases every other colorant.

A process colorant named in a Separation or DeviceN space is painted directly: its tint is
that colorant's value (8.6.6.4), whatever the alternate space, and a component the space does
not name keeps the backdrop's exactly. That holds for a sampled image and a shading too: an
image's samples are tints before the tint transform, and a shading's function gives tints,
so in a space naming process colorants alone the renderer paints those tints rather than the
alternate's conversion (process_color), which would put ink in the components the space does
not name. A DeviceCMYK colour a DefaultCMYK space remaps is no DeviceCMYK colour: "Provisions
in this standard that apply specifically to device colour spaces are then not applicable"
(8.6.5.6), so its marks are those of the default space.

A spot colorant, for which a CMYK group has no component, is kept on a plate of its own while
the group paints (render.raster_spots), its alternate colour's tints standing for it: a spot
overprinting the same spot replaces it, as Table 146 gives the source's value for "Spot
colourant named in source space", and the plates add to the process inks, clamped at full,
when the group ends, as a press's separate plate of the spot would print over them (the
composite MuPDF's overprint simulation makes of its spot plates).
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

from core_pdf.impl.base.values import recover_pdf_name
from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_08_graphics.color_spec import ColorSpace

CYAN = 1
MAGENTA = 2
YELLOW = 4
BLACK = 8
ALL_PROCESS = CYAN | MAGENTA | YELLOW | BLACK
# Set in a paint's marks when its space names a spot colorant: its alternate colour's tints
# in the components it does not mark stand for its spots, on a plate of their own
# (render.raster_spots), numbered by the bits from SPOT_SHIFT up (spot_plate).
SPOT = 16
# Set with SPOT for a paint naming a spot colorant that does not overprint: it replaces its
# spots' plate and erases every other colorant (Table 146, OP false).
OFF = 32
SPOT_SHIFT = 6
PROCESS_COLORANTS: dict[str, int] = {
    "Cyan": CYAN,
    "Magenta": MAGENTA,
    "Yellow": YELLOW,
    "Black": BLACK,
}
# The marks of each CMYK component in order.
CMYK_MARKS = (CYAN, MAGENTA, YELLOW, BLACK)


@cites(
    "ISO 32000-2",
    "11.7.3",
    introduced="1.4",
    rule="a Separation or DeviceN space naming process colourants converts to its alternate in "
    "a group whose colour space is not the device's native one",
)
def colorant_marks(kind: str, names: Sequence[str], *, device: bool = True) -> int | None:
    """The process components a Separation or DeviceN space naming names marks: each
    process colorant it names, and none for a spot colorant or None (8.6.6.4, 8.6.6.5), with
    SPOT and the spot colorants' plate (spot_plate) for a spot colorant; the separation All
    marks every component. device: the paint lands in the device's native DeviceCMYK
    (capture.recording.TextState.device_cmyk_group); elsewhere a space naming a process
    colorant "shall be converted to its alternate colour space" (11.7.3), a process colour space
    whose every component takes the source's value (Table 146), so it marks all four."""
    if kind == "Separation" and tuple(names) == ("All",):
        return None
    if not device and any(name in PROCESS_COLORANTS for name in names):
        return None
    marks = 0
    spots = set()
    for name in names:
        if name in PROCESS_COLORANTS:
            marks |= PROCESS_COLORANTS[name]
        elif name != "None":
            spots.add(name)
    if marks & ALL_PROCESS == ALL_PROCESS:
        return None
    if spots:
        marks |= SPOT | spot_plate(spots) << SPOT_SHIFT
    return marks


def spot_plate(names: set[str]) -> int:
    """The number of the plate a paint naming the spot colorants names paints on: the same
    for the same set in any order, from a CRC-32 of the names, whose collisions between the
    few spots of a page are negligible; never 0."""
    return zlib.crc32("\0".join(sorted(names)).encode("utf-8", "surrogatepass")) | 1 << 32


def spot_key(marks: int | None) -> int:
    """The plate a paint's spot colorants stand on (spot_plate), or 0 for none."""
    return marks >> SPOT_SHIFT if marks is not None and marks & SPOT else 0


def knockout_marks(marks: int | None) -> int | None:
    """The marks of a paint that does not overprint, given those its space would have if it
    did: OFF with them when they name a spot colorant, whose plate it replaces, and None,
    erasing every plate, for any other."""
    return marks | OFF if marks is not None and marks & SPOT else None


def paints_tints(marks: int | None) -> bool:
    """Whether a paint with marks overprints in a space naming process colorants alone, whose
    tints it paints directly (process_color)."""
    return marks is not None and not marks & SPOT and marks != ALL_PROCESS


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
    space: ColorSpace | None,
    color: Sequence[float] | None,
    mode: int,
    *,
    current_color: bool,
    device: bool = True,
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
        return colorant_marks(kind, space.colorants, device=device)
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


def process_indices(space: ColorSpace | None) -> tuple[int | None, ...] | None:
    """For a Separation or DeviceN space naming process colorants alone, the CMYK component
    each of its components paints directly (None for the colorant None); None for any other
    space, the separation All included."""
    if space is None or space.kind not in {"Separation", "DeviceN"}:
        return None
    names = tuple(space.colorants)
    if not names or names == ("All",):
        return None
    indices: list[int | None] = []
    for name in names:
        if name in PROCESS_COLORANTS:
            indices.append(CMYK_MARKS.index(PROCESS_COLORANTS[name]))
        elif name == "None":
            indices.append(None)
        else:
            return None
    if all(index is None for index in indices):
        return None
    return tuple(indices)


def shading_space(shading: object) -> object | None:
    """The ColorSpace entry of a shading dictionary, or None."""
    if not isinstance(shading, dict):
        return None
    for key, value in shading.items():
        if str(key) == "ColorSpace":
            return value
    return None


def raw_overprint_marks(value: object, *, device: bool = True) -> int | None:
    """overprint_marks of an image's or a shading's ColorSpace, a resolved PDF object: no
    overprint mode applies to them (Table 146 lists DeviceCMYK "not in a sampled image"
    alone). A Separation or DeviceN space marks the colorants it names, as for any paint;
    its process colorants' tints are painted directly (process_color), and its spot
    colorants stand on a plate of their own (render.raster_spots)."""
    seen: set[int] = set()
    while isinstance(value, (list, tuple)) and value and id(value) not in seen:
        seen.add(id(value))
        kind = recover_pdf_name(value[0])
        if kind == "Indexed" and len(value) == 4:
            value = value[1]
            continue
        if kind == "Separation" and len(value) == 4:
            name = recover_pdf_name(value[1])
            return None if name is None else colorant_marks(kind, (name,), device=device)
        if kind == "DeviceN" and len(value) in {4, 5} and isinstance(value[1], (list, tuple)):
            names = [recover_pdf_name(entry) for entry in value[1]]
            if not names or any(name is None for name in names):
                return None
            return colorant_marks(kind, [name for name in names if name is not None], device=device)
        return None
    return None

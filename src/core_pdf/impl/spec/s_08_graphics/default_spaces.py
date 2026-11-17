# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 8.6.5.6: default colour spaces.

A resource dictionary's ColorSpace subdictionary may name a DefaultGray, DefaultRGB or
DefaultCMYK colour space. Colour that the content stream using those resources paints in
DeviceGray, DeviceRGB or DeviceCMYK is then painted in the default space instead, its
components passed unchanged and clipped to the default space's ranges. That holds however
the device space is selected: by the current colour space (g, rg, k and their stroking
forms, or cs naming the space), by the ColorSpace of an image XObject, an inline image or a
shading, and as the base of an Indexed space, the underlying space of a Pattern space or the
alternate of a Separation or DeviceN space. A transparency group's CS naming a device space
is remapped the same way (11.6.6, Table 145).

What is not remapped:

- The definition of a default space itself, nor any space inside a CIE-based one, such as an
  ICCBased space's Alternate: remapping replaces a device space where it is selected, once.
- A default space that is Lab, Indexed or Pattern, or has a different number of components
  from the device space, is not a valid default; the device space stays as it is.
- The current resource dictionary is the one of the content stream that paints, so a form
  XObject, pattern cell, Type 3 glyph or annotation appearance with resources of its own
  uses their default spaces, not those of the stream that invoked it.

Where the clauses leave a choice open, this resolves it so:

- A transparency group's CS is remapped through the default spaces of the group XObject's
  own resources: the resource dictionary that is current while its objects are painted into
  the group, and the one its CS entry sits beside. The page group's CS uses the page's.
- The informative Annex P suggests that a group whose device CS has no valid default may
  inherit an ancestor's CIE-based blending space; that is not done here, and such a group
  blends in its device space.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import decoded_name
from core_pdf.impl.spec.s_08_graphics.color_spec import ColorSpace

# The resource each device space is remapped through.
DEFAULT_SPACE_KEYS: dict[str, str] = {
    "DeviceGray": "DefaultGray",
    "DeviceRGB": "DefaultRGB",
    "DeviceCMYK": "DefaultCMYK",
}
DEVICE_SPACE_COMPONENTS: dict[str, int] = {"DeviceGray": 1, "DeviceRGB": 3, "DeviceCMYK": 4}
INVALID_DEFAULT_KINDS = frozenset({"Lab", "Indexed", "Pattern"})
# The initial colour cs sets in each device space (8.6.8, Table 73), which a default space
# takes unchanged.
DEVICE_INITIAL_COLORS: dict[str, tuple[float, ...]] = {
    "DeviceGray": (0.0,),
    "DeviceRGB": (0.0, 0.0, 0.0),
    "DeviceCMYK": (0.0, 0.0, 0.0, 1.0),
}


class DefaultSpaces:
    """The valid default colour spaces of one resource dictionary: values holds each as
    resolved from the PDF and spaces as parsed, both by the device space it remaps."""

    __slots__ = ("spaces", "unit_ranges", "values")

    def __init__(self, values: dict[str, object], spaces: dict[str, ColorSpace]) -> None:
        self.values = values
        self.spaces = spaces
        # Whether each space's components range over 0 to 1, as device components do, so
        # that a device colour needs no clipping to be one of the default space's.
        self.unit_ranges = all(
            low == 0.0 and high == 1.0
            for space in spaces.values()
            for low, high in space.component_ranges
        )

    def remap(
        self, space: ColorSpace, color: tuple[float, ...]
    ) -> tuple[ColorSpace, tuple[float, ...]]:
        """A device space and colour, as the default space (if any) and the colour clipped
        to its ranges; any other space as it is."""
        remapped = self.spaces.get(space.kind)
        if remapped is None or remapped is space:
            return space, color
        return remapped, clip_to_ranges(remapped, color)

    def initial_color(self, space: ColorSpace) -> tuple[float, ...] | None:
        """The initial colour cs sets when it selects space, if space is one of these
        defaults: the device space's initial colour, which passes to it unchanged."""
        for device, remapped in self.spaces.items():
            if remapped is space:
                return clip_to_ranges(space, DEVICE_INITIAL_COLORS[device])
        return None


@cites(
    "ISO 32000-2",
    "8.6.5.6",
    introduced="1.1",
    rule="device colour values pass to the default space, clipped to its ranges",
)
def clip_to_ranges(space: ColorSpace, color: tuple[float, ...]) -> tuple[float, ...]:
    ranges = space.component_ranges
    if len(ranges) != len(color):
        return color
    return tuple(
        max(low, min(value, high)) for value, (low, high) in zip(color, ranges, strict=True)
    )


@cites(
    "ISO 32000-2",
    "8.6.5.6",
    introduced="1.1",
    rule="a ColorSpace subdictionary's DefaultGray, DefaultRGB and DefaultCMYK entries",
)
def default_space_values(
    color_spaces: PdfDict | None, resolve: Callable[[object], object]
) -> dict[str, object]:
    """The default colour spaces a resource dictionary's ColorSpace subdictionary names, by
    the device space each remaps, each resolved with resolve; empty with none."""
    if not color_spaces:
        return {}
    found: dict[str, object] = {}
    for device, key in DEFAULT_SPACE_KEYS.items():
        raw = color_spaces.get(key)
        value = resolve(raw) if raw is not None else None
        if value is not None and decoded_name(value) != device:
            found[device] = value
    return found


@cites(
    "ISO 32000-2",
    "8.6.5.6",
    introduced="1.1",
    rule="a default colour space is not Lab, Indexed or Pattern, with the device's components",
)
def valid_default_space(device: str, space: ColorSpace) -> bool:
    """Whether space may stand in for the device space: any space other than a Lab, Indexed
    or Pattern space, with as many components as the device space."""
    return (
        space.kind not in INVALID_DEFAULT_KINDS
        and space.kind not in DEFAULT_SPACE_KEYS
        and len(space.component_ranges) == DEVICE_SPACE_COMPONENTS[device]
    )


@cites(
    "ISO 32000-2",
    "8.6.5.6",
    introduced="1.1",
    rule="a device space remaps where selected and as an Indexed, Pattern or tint space's base",
)
def remap_color_space(value: object, defaults: Mapping[str, object]) -> object:
    """A colour space value, as resolved from a PDF, with each device space it selects
    replaced by its default space from defaults (raw values by device space name): the space
    itself, the base of an Indexed space, the underlying space of a Pattern space and the
    alternate of a Separation or DeviceN space, at any depth. Anything else, a CIE-based
    space's insides included, is returned as it is; so is the value when nothing changes."""
    if not defaults:
        return value
    name = decoded_name(value)
    if name is not None:
        return defaults.get(name, value)
    if not isinstance(value, (list, tuple)) or not value:
        return value
    return remap_array(value, defaults, set())


def remap_array(
    value: list[object] | tuple[object, ...], defaults: Mapping[str, object], active: set[int]
) -> object:
    marker = id(value)
    if marker in active:
        return value
    kind = decoded_name(value[0])
    if kind in {"Indexed", "Pattern"}:
        index = 1
    elif kind in {"Separation", "DeviceN"}:
        index = 2
    else:
        return value
    if len(value) <= index:
        return value
    inner = value[index]
    inner_name = decoded_name(inner)
    if inner_name is not None:
        replaced = defaults.get(inner_name, inner)
    elif isinstance(inner, (list, tuple)) and inner:
        active.add(marker)
        try:
            replaced = remap_array(inner, defaults, active)
        finally:
            active.remove(marker)
    else:
        return value
    if replaced is inner:
        return value
    remapped = list(value)
    remapped[index] = replaced
    return remapped


__all__ = (
    "DEFAULT_SPACE_KEYS",
    "DEVICE_INITIAL_COLORS",
    "DEVICE_SPACE_COMPONENTS",
    "DefaultSpaces",
    "clip_to_ranges",
    "default_space_values",
    "remap_color_space",
    "valid_default_space",
)

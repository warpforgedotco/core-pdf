# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 11.3.4: blending colour spaces.

Which colour space a transparency group composites in (11.3.4, the page group of 11.4.7 and
the group colour space of 11.6.6), and the reference implementation of B(Cb, Cs) in any
blending space, built on the RGB blend functions of
core_pdf.impl.spec.s_11_transparency.blend.

A blending space is one of three families, each named by its device space: DeviceGray (one
additive component: DeviceGray, CalGray, ICCBased gray), DeviceRGB (three additive ones:
DeviceRGB, CalRGB, ICCBased RGB) and DeviceCMYK (four subtractive ones: DeviceCMYK, ICCBased
CMYK). Lab and the other spaces 11.3.4 excludes are none of them.

A subtractive space is blended on the complements of its components (11.3.4). Complementing
is affine, and the compositing formula is a weighted average whose weights sum to one, so
compositing the complements gives the complement of compositing the components themselves.
This module therefore works on a colour's additive form -- its components, or for a
subtractive space their complements -- and additive_blend is B on that form; blend_in_space
complements on the way in and out. The renderer composites the additive form too, which is
what makes the two agree to the bit.

Where the clauses leave a choice open, this resolves it so that the result is the exact
mathematical one:

- Gray blends the non-separable modes "by conversion to RGB, blending in RGB, and then
  converting back to gray" (11.3.5.3). A gray g is the RGB (g, g, g) (10.4.2.2), and each of
  the four modes gives a neutral colour, all three components computed by the same
  operations, for neutral inputs; converting it back takes that component, which the NTSC
  weights of 10.4.2.2 would give in exact arithmetic but may round away from.
- CMYK's K component under Hue, Saturation and Color is the backdrop's, and under Luminosity
  the source's (11.3.5.3), exactly, not through the RGB functions.
"""

from __future__ import annotations

from typing import Literal

from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.exceptions import PdfUnsupportedError
from core_pdf.impl.spec.s_08_graphics.color_spec import ColorSpace
from core_pdf.impl.spec.s_11_transparency.blend import (
    NON_SEPARABLE_BLEND_MODES,
    SEPARABLE_BLEND_MODES,
    BlendMode,
    blend_revision,
    check_unit,
    non_separable_blend,
    separable_blend,
)
from core_pdf.impl.spec.standards import SemanticContext

BlendingSpace = Literal["DeviceGray", "DeviceRGB", "DeviceCMYK"]
BLENDING_SPACE_COMPONENTS: dict[BlendingSpace, int] = {
    "DeviceGray": 1,
    "DeviceRGB": 3,
    "DeviceCMYK": 4,
}
SUBTRACTIVE_BLENDING_SPACES: frozenset[BlendingSpace] = frozenset(("DeviceCMYK",))
# A raster output device's native colour space, which the page group blends in unless the
# page says otherwise (11.4.7): core renders to sRGB.
OUTPUT_BLENDING_SPACE: BlendingSpace = "DeviceRGB"
# The blending space of each family's colour spaces, and of an ICCBased space by its number
# of components.
FAMILY_SPACES: dict[str, BlendingSpace] = {
    "DeviceGray": "DeviceGray",
    "CalGray": "DeviceGray",
    "DeviceRGB": "DeviceRGB",
    "CalRGB": "DeviceRGB",
    "DeviceCMYK": "DeviceCMYK",
}
ICC_SPACES: dict[int, BlendingSpace] = {1: "DeviceGray", 3: "DeviceRGB", 4: "DeviceCMYK"}


@cites("ISO 32000-2", "11.3.4", introduced="1.4", rule="the colour spaces that blend")
def blending_space_family(kind: str, components: int | None = None) -> BlendingSpace | None:
    """The blending space a colour space of this kind (and, for ICCBased, number of
    components) blends as, or None for one that cannot be a blending space: Lab,
    lightness-chromaticity ICC spaces, Indexed, Pattern, Separation and DeviceN."""
    if kind == "ICCBased":
        return ICC_SPACES.get(components) if components is not None else None
    return FAMILY_SPACES.get(kind)


def blending_space_of(space: ColorSpace | None) -> BlendingSpace | None:
    """The blending space a group attributes dictionary's CS names, or None for a missing
    entry or a colour space that cannot blend, which leaves the group in its parent's."""
    if space is None:
        return None
    return blending_space_family(space.kind, len(space.component_ranges))


def blending_profile(space: ColorSpace | None) -> bytes | None:
    """The ICC profile of an ICCBased CMYK blending space: the CMYK its colour means."""
    if space is None or space.kind != "ICCBased" or blending_space_of(space) != "DeviceCMYK":
        return None
    return space.icc_profile


@cites("ISO 32000-2", "11.6.6", introduced="1.4", rule="a transparency group's colour space")
def group_blending_space(
    declared: BlendingSpace | None, *, isolated: bool, parent: BlendingSpace
) -> BlendingSpace:
    """The blending space of a group XObject: its CS entry for an isolated group, otherwise,
    or without one, the space of the group or page it is painted into."""
    if isolated and declared is not None:
        return declared
    return parent


@cites("ISO 32000-2", "11.4.7", introduced="1.4", rule="the page group's blending colour space")
def page_blending_space(declared: BlendingSpace | None) -> BlendingSpace:
    """The page group's blending space: the CS of the page's Group entry, honoured whether
    or not the page group is isolated, or else the output device's native space."""
    return OUTPUT_BLENDING_SPACE if declared is None else declared


def additive_components(space: BlendingSpace, components: tuple[float, ...]) -> tuple[float, ...]:
    """A colour's additive form: its components, complemented in a subtractive space. The
    form is its own inverse."""
    if space in SUBTRACTIVE_BLENDING_SPACES:
        return tuple(1.0 - value for value in components)
    return components


@cites("ISO 32000-2", "11.3.5.3", introduced="1.4", rule="non-separable blending in gray and CMYK")
def additive_blend(
    backdrop: tuple[float, ...],
    source: tuple[float, ...],
    mode: str,
    space: BlendingSpace,
    revised: bool = True,
) -> tuple[float, ...]:
    """B(Cb, Cs) on the additive forms of two colours of a blending space, unchecked.

    A separable mode blends each component. A non-separable one blends an RGB colour as
    Table 135 does; a gray as the neutral RGB colour it equals, taking back the common
    component; and CMYK's first three (complemented) components as an RGB colour, its K
    the backdrop's for Hue, Saturation and Color and the source's for Luminosity.
    """
    if mode in SEPARABLE_BLEND_MODES:
        return tuple(
            separable_blend(cb, cs, mode, revised) for cb, cs in zip(backdrop, source, strict=True)
        )
    if mode not in NON_SEPARABLE_BLEND_MODES:
        raise PdfUnsupportedError(f"unsupported blend mode: {mode}")
    if space == "DeviceGray":
        gray, level = backdrop[0], source[0]
        return (non_separable_blend((gray, gray, gray), (level, level, level), mode)[0],)
    color = non_separable_blend(
        (backdrop[0], backdrop[1], backdrop[2]), (source[0], source[1], source[2]), mode
    )
    if space == "DeviceRGB":
        return color
    return (*color, source[3] if mode == "Luminosity" else backdrop[3])


@cites("ISO 32000-2", "11.3.4", introduced="1.4", rule="blend a subtractive space's complements")
def blend_in_space(
    backdrop: tuple[float, ...],
    source: tuple[float, ...],
    mode: BlendMode,
    space: BlendingSpace,
    *,
    context: SemanticContext | None = None,
) -> tuple[float, ...]:
    """B(Cb, Cs) of any standard mode on two colours of a blending space, each given as the
    space's own components (a subtractive space's colorant amounts)."""
    revised = blend_revision(mode, context)
    count = BLENDING_SPACE_COMPONENTS[space]
    if len(backdrop) != count or len(source) != count:
        raise ValueError(f"a {space} colour has {count} components")
    check_unit(*backdrop, *source)
    blended = additive_blend(
        additive_components(space, backdrop),
        additive_components(space, source),
        mode,
        space,
        revised,
    )
    return additive_components(space, blended)


@cites("ISO 32000-2", "10.4.2.2", rule="the gray level of an RGB colour")
def gray_from_rgb(red: float, green: float, blue: float) -> float:
    return 0.3 * red + 0.59 * green + 0.11 * blue


@cites("ISO 32000-2", "10.4.2.3", rule="the CMYK colour of a gray level")
def cmyk_from_gray(gray: float) -> tuple[float, float, float, float]:
    return 0.0, 0.0, 0.0, 1.0 - gray


__all__ = (
    "BLENDING_SPACE_COMPONENTS",
    "OUTPUT_BLENDING_SPACE",
    "SUBTRACTIVE_BLENDING_SPACES",
    "BlendingSpace",
    "additive_blend",
    "additive_components",
    "blend_in_space",
    "blending_profile",
    "blending_space_family",
    "blending_space_of",
    "cmyk_from_gray",
    "gray_from_rgb",
    "group_blending_space",
    "page_blending_space",
)

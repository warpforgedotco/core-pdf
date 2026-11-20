# SPDX-License-Identifier: AGPL-3.0-only

"""RasterTarget methods: transparency groups that blend in a colour space other than their
parent's (ISO 32000-2 11.3.4, 11.4.7, 11.6.6).

The raster always holds three colour components and an alpha, and its paths blend them as
additive components. A group whose blending space is DeviceRGB holds sRGB, as everything
else does. A DeviceGray group holds its gray level in all three components: every colour
painted into it becomes gray first, and blending three equal components is blending the
gray (each separable mode works per component, and the non-separable ones give the neutral
colour 11.3.5.3 converts back from). A DeviceCMYK group is painted twice, from the same
display items, once holding the complements of cyan, magenta and yellow (SpaceMode.CMY) and
once the complement of black in all three components (SpaceMode.BLACK): its colour's
additive form, which 11.3.4 blends and core_pdf.impl.spec.s_11_transparency.blending_space
composites. Alpha and shape do not depend on colour, so both passes compute the same ones,
and on the K pass the non-separable modes blend as 11.3.5.3 has K blend (resolved_blend).
When the group ends, its colour is converted to its parent's space and the group composites
as any other.

Colours enter a space as they reach the raster. In a CMYK group a colour that comes down to
CMYK keeps that CMYK: a DeviceCMYK colour or image as it is, the group's own CMYK (11.7.2),
an ICCBased CMYK one converted from its profile to the group's unless the two are the same,
and a Separation, DeviceN or Indexed colour through its tint transform or lookup, which
capture applies inside a CMYK group instead of converting to sRGB (capture.paint_sources). A gray
level becomes black (10.4.2.3), and anything else -- the sRGB capture gives every other
colour, and other decoded images -- goes through the group's CMYK profile. In a gray group
every colour becomes the gray level of its sRGB (10.4.2.2). A CMYK group's colour converts
to sRGB through its profile: an ICCBased space's own, or for DeviceCMYK the built-in one a
DeviceCMYK colour converts through anywhere, so an opaque DeviceCMYK fill renders the same
bytes inside a DeviceCMYK group as outside one.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from functools import cache
from typing import TYPE_CHECKING, Any

lazy import numpy

from core_pdf.impl.base.arrays import UInt8Array
from core_pdf.impl.base.geometry import clamp01
from core_pdf.impl.graphics.device_profiles import (
    cmyk_bytes_luminance,
    cmyk_bytes_to_srgb,
    component_byte,
    srgb_bytes_to_cmyk,
)
from core_pdf.impl.graphics.icc_profiles import IccProfileError
from core_pdf.impl.graphics.images import decode_cmyk_samples, decode_process_samples
from core_pdf.impl.render.blend import color_rgba
from core_pdf.impl.render.model import (
    ControlItem,
    DisplayItem,
    GroupBeginItem,
    PaintItemBase,
    RasterGroup,
    SpaceMode,
)
from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_08_graphics.image_spec import ImageSource
from core_pdf.impl.spec.s_11_transparency.blending_space import BlendingSpace, group_blending_space
lazy from core_pdf.impl.graphics.image_samples import d50_xyz_to_srgb_matrix
lazy from core_pdf_cythonized import luminosity_channel

if TYPE_CHECKING:
    from core_pdf.impl.capture.records import CapturedSoftMask
    from core_pdf.impl.render.target import RasterTarget

type Rgba = tuple[int, int, int, int]

MODE_SPACES: dict[SpaceMode, BlendingSpace] = {
    SpaceMode.RGB: "DeviceRGB",
    SpaceMode.GRAY: "DeviceGray",
    SpaceMode.CMY: "DeviceCMYK",
    SpaceMode.BLACK: "DeviceCMYK",
}
# The passes that paint a group of each blending space.
SPACE_PASSES: dict[BlendingSpace, tuple[SpaceMode, ...]] = {
    "DeviceRGB": (SpaceMode.RGB,),
    "DeviceGray": (SpaceMode.GRAY,),
    "DeviceCMYK": (SpaceMode.CMY, SpaceMode.BLACK),
}
SRGB_CMYK_CACHE_LIMIT = 4096


def gray_bytes(rgb: UInt8Array) -> UInt8Array:
    """The gray level of each sRGB byte triple, (n, 3) to (n,), by ISO 32000-2 10.4.2.2."""
    values = rgb.astype(numpy.float64) / 255.0
    # gray_from_rgb's weights, in its order, on arrays.
    gray = 0.3 * values[:, 0] + 0.59 * values[:, 1] + 0.11 * values[:, 2]
    return numpy.clip(numpy.rint(gray * 255.0), 0.0, 255.0).astype(numpy.uint8)


def plane_bytes(cmyk: UInt8Array, mode: SpaceMode) -> UInt8Array:
    """(n, 4) CMYK bytes as one pass's (n, 3) components: the complements of C, M and Y, or
    of K three times."""
    if mode is SpaceMode.CMY:
        return 255 - cmyk[:, :3]
    return numpy.repeat(255 - cmyk[:, 3:4], 3, axis=1)


def set_space_mode(self: RasterTarget, mode: SpaceMode, profile: bytes | None) -> None:
    """Paint from now on in mode, and for a CMYK mode with profile's CMYK (None: the
    built-in DeviceCMYK profile's)."""
    if profile is not self.space_profile:
        self.space_colors.clear()
        self.space_images.clear()
    self.space_mode = mode
    self.space_profile = profile
    self.fill_memo = None


def srgb_cmyk(self: RasterTarget, red: int, green: int, blue: int) -> tuple[int, int, int, int]:
    key = (red, green, blue)
    cache = self.space_colors
    found = cache.get(key)
    if found is None:
        if len(cache) >= SRGB_CMYK_CACHE_LIMIT:
            cache.clear()
        row = srgb_bytes_to_cmyk(numpy.array([key], dtype=numpy.uint8), self.space_profile)[0]
        found = cache[key] = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
    return found


def color_cmyk(self: RasterTarget, color: Any, rgba: Rgba) -> tuple[int, int, int, int]:
    """The CMYK bytes of a colour painted into a CMYK group, given the sRGB it paints as."""
    if isinstance(color, (list, tuple)):
        if len(color) == 4:
            try:
                cyan, magenta, yellow, black = (component_byte(clamp01(float(c))) for c in color)
            except TypeError, ValueError:
                return self.srgb_cmyk(rgba[0], rgba[1], rgba[2])
            return cyan, magenta, yellow, black
        if len(color) == 1:
            return 0, 0, 0, 255 - rgba[0]
    return self.srgb_cmyk(rgba[0], rgba[1], rgba[2])


def space_color(self: RasterTarget, color: Any, rgba: Rgba) -> Rgba:
    """rgba, the colour painted as sRGB, in the current space's components."""
    mode = self.space_mode
    if mode is SpaceMode.GRAY:
        if isinstance(color, (list, tuple)) and len(color) == 1:
            return rgba
        gray = int(gray_bytes(numpy.array([rgba[:3]], dtype=numpy.uint8))[0])
        return gray, gray, gray, rgba[3]
    cyan, magenta, yellow, black = self.color_cmyk(color, rgba)
    if mode is SpaceMode.CMY:
        return 255 - cyan, 255 - magenta, 255 - yellow, rgba[3]
    return 255 - black, 255 - black, 255 - black, rgba[3]


def paint_rgba(self: RasterTarget, color: Any, opacity: Any) -> Rgba:
    rgba = color_rgba(color, opacity)
    if self.space_mode is SpaceMode.RGB:
        return rgba
    return self.space_color(color, rgba)


def item_fill_rgba(self: RasterTarget, item: PaintItemBase) -> Rgba:
    rgba = item.fill_rgba()
    if self.space_mode is SpaceMode.RGB:
        return rgba
    return self.space_color(item.fill, rgba)


def shading_space_color(self: RasterTarget, color_model: str, components: Any, rgba: Rgba) -> Rgba:
    """A shading's colour, rgba as sRGB, in the current space's components."""
    name = color_model or "DeviceRGB"
    count = len(components)
    if name.endswith("DeviceGray") or count == 1:
        return self.space_color((0.0,), rgba)
    if name.endswith("DeviceCMYK") and count >= 4:
        return self.space_color(tuple(float(c) for c in components[:4]), rgba)
    return self.space_color(None, rgba)


def space_samples(
    self: RasterTarget,
    source: ImageSource,
    samples: UInt8Array,
    components: int,
    *,
    process: bool = False,
) -> tuple[UInt8Array, int]:
    """An image's decoded samples, sRGB (components 3) or gray (1), in the current space's
    components. A CMYK image keeps its own components in a CMYK group; a CMYK group finds
    each image's CMYK once for both of its passes. process: the image overprints in a
    space naming process colorants alone, whose tints a CMYK group takes directly
    (images.decode_process_samples)."""
    mode = self.space_mode
    cmyk = (
        self.process_samples(source, len(samples) // components)
        if process and mode is not SpaceMode.GRAY
        else None
    )
    if cmyk is not None:
        if mode is SpaceMode.CMY:
            return numpy.ascontiguousarray(255 - cmyk[:, :3]).reshape(-1), 3
        return numpy.ascontiguousarray(255 - cmyk[:, 3]), 1
    if components == 1:
        if mode is SpaceMode.CMY:
            return numpy.full(samples.shape, 255, dtype=numpy.uint8), 1
        return samples, 1
    rgb = samples.reshape(-1, 3)
    if mode is SpaceMode.GRAY:
        return gray_bytes(rgb), 1
    key = (id(source), len(rgb))
    cached = self.space_images.get(key)
    if cached is None or cached[0] is not source:
        native = decode_cmyk_samples(source, self.space_profile)
        cmyk = (
            native.reshape(-1, 4)
            if native is not None and native.size == len(rgb) * 4
            else srgb_bytes_to_cmyk(rgb, self.space_profile)
        )
        cached = self.space_images[key] = (source, cmyk)
    cmyk = cached[1]
    assert cmyk is not None
    if mode is SpaceMode.CMY:
        return numpy.ascontiguousarray(255 - cmyk[:, :3]).reshape(-1), 3
    return numpy.ascontiguousarray(255 - cmyk[:, 3]), 1


def process_samples(self: RasterTarget, source: ImageSource, pixels: int) -> UInt8Array | None:
    """The CMYK, (pixels, 4), an image in a space naming process colorants alone paints
    directly (images.decode_process_samples), found once for both passes of a CMYK group;
    None when its tints cannot be had for its pixels: a matte, or samples of another size
    than the image it is painted from."""
    key = (id(source), -1)
    cached = self.space_images.get(key)
    if cached is None or cached[0] is not source:
        native = decode_process_samples(source)
        found = native.reshape(-1, 4) if native is not None and native.size == pixels * 4 else None
        cached = self.space_images[key] = (source, found)
    return cached[1]


def switches_space(self: RasterTarget, item: GroupBeginItem) -> bool:
    """Whether a group blends in another space than the one it is painted into."""
    if item.blending_space is None:
        return False
    parent = MODE_SPACES[self.space_mode]
    return (
        group_blending_space(item.blending_space, isolated=item.isolated is True, parent=parent)
        != parent
    )


def group_members(items: Iterator[DisplayItem]) -> list[DisplayItem]:
    """The items of the group whose group-begin was just taken from items, up to and taking
    its group-end."""
    members: list[DisplayItem] = []
    depth = 0
    for item in items:
        if type(item) is GroupBeginItem:
            depth += 1
        elif type(item) is ControlItem and item.kind == "group-end":
            if depth == 0:
                break
            depth -= 1
        members.append(item)
    return members


def paint_space_group(
    self: RasterTarget, item: GroupBeginItem, members: Iterable[DisplayItem]
) -> None:
    """Paint a group that blends in another space than its parent: once per pass of its
    space, each into a group of its own, then its colour converted to the parent's space and
    the group composited."""
    space = item.blending_space
    assert space is not None
    parent_mode = self.space_mode
    parent_profile = self.space_profile
    profile = item.blending_profile if space == "DeviceCMYK" else None
    members = list(members)
    passes: list[RasterGroup] = []
    try:
        for mode in SPACE_PASSES[space]:
            self.set_space_mode(mode, profile)
            self.paint_group_begin(item)
            self.push_scope()
            try:
                self.paint_items(members)
            finally:
                self.pop_scope()
                passes.append(self.pop_group())
    finally:
        self.set_space_mode(parent_mode, parent_profile)
        if parent_mode is SpaceMode.RGB:
            self.space_images.clear()
    convert_group_space(passes, space, profile, parent_mode, parent_profile)
    self.composite_group(passes[0])


# The luminosity_channel code of each blending space.
LUMINOSITY_SPACES: dict[BlendingSpace, int] = {"DeviceRGB": 0, "DeviceGray": 1, "DeviceCMYK": 2}


def luminosity_mask_channel(
    self: RasterTarget, mask: CapturedSoftMask, items: Sequence[DisplayItem]
) -> tuple[UInt8Array, Any]:
    """A Luminosity soft mask's values before its transfer function, a byte per pixel, and
    if it has one, which bytes occur (ISO 32000-2 11.5.3, 11.6.5.1). Its group composites
    in its blending space over a fully opaque backdrop of the BC colour, a raster of the
    parent's size for each pass of the space, so outside the group's bounding box the BC
    colour stays; and each pixel's colour, C = (1 - ag) C0 + ag Cg, becomes its
    luminosity. A CIE-based gray or RGB group composites in sRGB, which every colour painted
    into it converts to colorimetrically, and an ICCBased CMYK one in its profile's CMYK;
    the luminosity of either is the colour's CIE 1931 Y."""
    space = mask.blending_space
    assert space is not None
    profile = mask.blending_profile if space == "DeviceCMYK" else None
    if mask.colorimetric and space == "DeviceGray":
        space = "DeviceRGB"
    views: list[UInt8Array] = []
    for mode in SPACE_PASSES[space]:
        nested, view = self.blank_sibling()
        nested.set_space_mode(mode, profile)
        red, green, blue, _ = nested.paint_rgba(mask.backdrop, 1.0)
        pixel = numpy.array((red, green, blue, 255), dtype=numpy.uint8).view(numpy.uint32)
        view.reshape(-1).view(numpy.uint32).fill(pixel[0])
        nested.paint_items(items, translation=mask.offset)
        views.append(view)
    if mask.colorimetric:
        return colorimetric_luminosity_channel(views, profile, mask.transfer is not None)
    return luminosity_channel(
        views[0],
        views[1] if len(views) > 1 else None,
        LUMINOSITY_SPACES[space],
        mask.transfer is not None,
    )


@cache
def srgb_luminance_tables() -> numpy.ndarray[Any, numpy.dtype[numpy.float64]]:
    """For each sRGB byte of each of red, green and blue, (3, 256), its share of the Y of
    the colour under the D50 white of the PCS: the byte linearized, times the Y row of the
    linear sRGB to D50 XYZ matrix, the inverse of the one sRGB is found through
    (image_samples.d50_xyz_to_srgb_matrix)."""
    encoded = numpy.arange(256, dtype=numpy.float64) / 255.0
    linear = numpy.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)
    row = numpy.linalg.inv(d50_xyz_to_srgb_matrix().astype(numpy.float64))[1]
    return row[:, None] * linear[None, :]


@cites(
    "ISO 32000-2",
    "11.5.3",
    introduced="1.4",
    rule="a CIE-based group's luminosity is the CIE 1931 Y of its colour",
)
def colorimetric_luminosity_channel(
    views: list[UInt8Array], profile: bytes | None, presence: bool
) -> tuple[UInt8Array, Any]:
    """Each pixel's luminosity as a byte, the CIE 1931 Y of its colour relative to white,
    and with presence, which bytes occur: of an sRGB raster (one view), or of the CMYK of
    profile that a CMYK group's two passes hold (Y through the profile)."""
    first = views[0]
    height, width = first.shape[:2]
    if len(views) == 1:
        red, green, blue = srgb_luminance_tables()
        luminance = red[first[..., 0]]
        luminance += green[first[..., 1]]
        luminance += blue[first[..., 2]]
    else:
        cmyk = numpy.empty((height, width, 4), dtype=numpy.uint8)
        numpy.subtract(255, first[..., :3], out=cmyk[..., :3])
        numpy.subtract(255, views[1][..., 0], out=cmyk[..., 3])
        try:
            luminance = cmyk_bytes_luminance(cmyk.reshape(-1, 4), profile).reshape(height, width)
        except IccProfileError:
            # A profile whose Y cannot be found: the device formula, as for DeviceCMYK.
            return luminosity_channel(first, views[1], LUMINOSITY_SPACES["DeviceCMYK"], presence)
    luminance *= 255.0
    numpy.rint(luminance, out=luminance)
    channel = numpy.clip(luminance, 0.0, 255.0, out=luminance).astype(numpy.uint8)
    present = numpy.bincount(channel.reshape(-1), minlength=256) > 0 if presence else None
    return channel, present


def convert_group_space(
    passes: list[RasterGroup],
    space: BlendingSpace,
    profile: bytes | None,
    parent: SpaceMode,
    parent_profile: bytes | None,
) -> None:
    """Rewrite the first pass's colour, where it has any alpha, as the group's colour in the
    parent's space; the passes share alpha and shape."""
    group = passes[0]
    for other in passes[1:]:
        bounds = other.paint_window.bounds()
        if bounds:
            group.paint_window.extend(*bounds)
    window = group.paint_window.slices()
    if window is None:
        return
    rows, columns = window
    view = group.view[rows, columns]
    visible = view[..., 3] > 0
    if not numpy.any(visible):
        return
    if space == "DeviceCMYK":
        cmyk = numpy.column_stack(
            (255 - view[visible, :3], 255 - passes[1].view[rows, columns][visible, 0])
        )
        rgb = cmyk_bytes_to_srgb(cmyk, profile)
        if parent is SpaceMode.GRAY:
            rgb = numpy.repeat(gray_bytes(rgb)[:, None], 3, axis=1)
    elif space == "DeviceGray":
        if parent is SpaceMode.RGB:
            return
        gray = view[visible, 0]
        if parent is SpaceMode.CMY:
            rgb = numpy.full((len(gray), 3), 255, dtype=numpy.uint8)
        else:
            rgb = numpy.repeat(gray[:, None], 3, axis=1)
    elif parent is SpaceMode.GRAY:
        rgb = numpy.repeat(gray_bytes(view[visible, :3])[:, None], 3, axis=1)
    else:
        rgb = plane_bytes(srgb_bytes_to_cmyk(view[visible, :3], parent_profile), parent)
    view[visible, :3] = rgb

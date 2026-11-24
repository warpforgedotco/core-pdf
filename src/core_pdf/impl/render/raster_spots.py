# SPDX-License-Identifier: AGPL-3.0-only

"""RasterTarget methods: spot colorant plates in a DeviceCMYK blending space (ISO 32000-2
11.7.3, 11.7.4.3, Table 146).

A CMYK group's colour space has no component for a spot colorant, which a press prints from
a plate of its own: "The group shall maintain a separate colour value for each spot colour
component, independently of the group's colour space ... it shall still be subject to
blending and compositing with other objects that use the same spot colour" (11.7.3). While a
CMYK group paints, each pass of it (raster_spaces) keeps such a plate for each set of spot
colorants a paint names (graphics.overprint.spot_plate): the ink its alternate colour gives
the pass's components that the paint does not mark, which stand for the spots -- three
components on the CMY pass, one on the K pass -- a byte each. A plate shares its group's
alpha and composites as a colour component of the group does (11.3.6), by the special blend
mode of Table 146:

- a paint naming the plate's spots takes its own value, "Spot colourant named in source
  space" being Cs with overprint on or off: a spot over the same spot replaces it;
- an overprinting paint naming other colorants keeps it, Cb, even one that marks every
  process component;
- a paint that does not overprint erases it: the value of a colorant it does not name is 0
  ("1.0" additive);
- a group, which "is considered to paint all colour components, both process and spot"
  (11.7.4.5, NOTE 2), composites its own plates, empty where it has none.

When the CMYK group ends, its plates add to its process components, clamped at full, as the
plate of a spot printed over the process inks would, before its colour is converted to its
parent's space: the composite MuPDF's overprint simulation makes of its separations.

A paint naming a spot paints through the layer of raster_overprint, whose colour gives the
plate's new value; any other paint takes its usual path, unchanged, and one that does not
overprint, where it reaches a plate, is drawn again on a blank page for the alpha its plates
are scaled by. A paint with a pattern lands through a group of its own, which composites
the plates its cells leave. A plate covers the boxes of the paints that touched it, grown
as they come, not the page; it is copied a logarithmic number of times as it grows. A
group starts from the plates of its backdrop, which it reads through and copies only where it
changes them; an isolated group starts from none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

lazy import numpy

from core_pdf.impl.base.arrays import UInt8Array
from core_pdf.impl.base.types import frozen_setattr
from core_pdf.impl.render.model import (
    ImagePaintItem,
    PaintItemBase,
    PathPaintItem,
    PathPaintKind,
    PixelWindow,
    RasterGroup,
    SpaceMode,
)
from core_pdf.impl.spec.citations import cites
lazy from core_pdf_cythonized._spot_plates import (
    ERASED,
    GROUP,
    KEPT,
    NAMED,
    apply_spot_plate,
    composite_spot_plate,
)

if TYPE_CHECKING:
    from core_pdf.impl.render.target import RasterTarget

type Box = tuple[int, int, int, int]

# A plate that grows takes at least half its size again on each side it grows on, so a
# plate painted in many small steps is copied a logarithmic number of times.
GROWTH = 0.5
# Plates composite in bands of this many rows, so the floats they composite in stay small
# however large the paint.
BAND = 128


def bands(rows: slice) -> list[slice]:
    """rows in bands of BAND rows, relative to rows.start."""
    height = rows.stop - rows.start
    return [slice(top, min(height, top + BAND)) for top in range(0, height, BAND)]


def intersect(first: Box | None, second: Box | None) -> Box | None:
    """The intersection of two (x0, y0, x1, y1) boxes, None when empty."""
    if first is None or second is None:
        return None
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    return (x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None


def union(first: Box | None, second: Box | None) -> Box | None:
    if first is None:
        return second
    if second is None:
        return first
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


class Plate:
    """One plate's ink, a byte per component, over box, (x0, y0, x1, y1) on the page; extent,
    within it, is the union of the boxes painted on it, outside which it is its base's."""

    __slots__ = ("box", "data", "extent")

    def __init__(self, box: Box, data: UInt8Array, extent: Box) -> None:
        self.box = box
        self.data = data
        self.extent = extent


class SpotPlates:
    """A group's plates in one pass, by plate number: own, those it changed, over the boxes
    it changed them in, and elsewhere base's, the plates it started from, which it never
    changes; channels components each."""

    __slots__ = ("base", "channels", "keys", "own", "union_box")

    def __init__(self, base: SpotPlates | None, channels: int) -> None:
        self.base = base
        self.channels = channels
        self.own: dict[int, Plate] = {}
        # The box outside which every plate is empty: base's, and the boxes changed since.
        self.union_box: Box | None = base.union_box if base is not None else None
        # The numbers of the plates, its own and base's, which base never adds to.
        self.keys: frozenset[int] = base.keys if base is not None else frozenset()

    def numbers(self) -> frozenset[int]:
        return self.keys

    def touches(self, window: Box) -> bool:
        """Whether a plate may hold ink within window: its box meets it."""
        if intersect(self.union_box, window) is None:
            return False
        return any(intersect(self.box(key), window) is not None for key in self.keys)

    def box(self, key: int) -> Box | None:
        """The box outside which plate key is empty."""
        plate = self.own.get(key)
        below = self.base.box(key) if self.base is not None else None
        return union(plate.extent if plate is not None else None, below)

    def read(self, key: int, box: Box) -> UInt8Array | None:
        """Plate key's ink over box, or None where it is empty there."""
        out = self.base.read(key, box) if self.base is not None else None
        plate = self.own.get(key)
        if plate is None:
            return out
        common = intersect(plate.box, box)
        if common is None:
            return out
        if out is None:
            out = numpy.zeros((box[3] - box[1], box[2] - box[0], self.channels), numpy.uint8)
        x0, y0, x1, y1 = common
        px0, py0 = plate.box[0], plate.box[1]
        out[y0 - box[1] : y1 - box[1], x0 - box[0] : x1 - box[0]] = plate.data[
            y0 - py0 : y1 - py0, x0 - px0 : x1 - px0
        ]
        return out

    def writable(self, key: int, box: Box, width: int, height: int) -> UInt8Array:
        """Plate key's own ink over box, to change: the plate grows to cover it, from base's
        ink where it was not its own, within a page width by height."""
        plate = self.own.get(key)
        if plate is None or intersect(plate.box, box) != box:
            grown = box
            if plate is not None:
                ox0, oy0, ox1, oy1 = plate.box
                grow_x = max(1, round((ox1 - ox0) * GROWTH))
                grow_y = max(1, round((oy1 - oy0) * GROWTH))
                x0, y0, x1, y1 = union(plate.box, box) or box
                grown = (
                    max(0, min(x0, ox0 - grow_x)) if x0 < ox0 else ox0,
                    max(0, min(y0, oy0 - grow_y)) if y0 < oy0 else oy0,
                    min(width, max(x1, ox1 + grow_x)) if x1 > ox1 else ox1,
                    min(height, max(y1, oy1 + grow_y)) if y1 > oy1 else oy1,
                )
            data = self.base.read(key, grown) if self.base is not None else None
            if data is None:
                data = numpy.zeros(
                    (grown[3] - grown[1], grown[2] - grown[0], self.channels), numpy.uint8
                )
            if plate is not None:
                ox0, oy0, ox1, oy1 = plate.box
                data[oy0 - grown[1] : oy1 - grown[1], ox0 - grown[0] : ox1 - grown[0]] = plate.data
            extent = box if plate is None else union(plate.extent, box) or box
            plate = self.own[key] = Plate(grown, data, extent)
            if key not in self.keys:
                self.keys = self.keys | {key}
        else:
            plate.extent = union(plate.extent, box) or box
        self.union_box = union(self.union_box, box)
        px0, py0 = plate.box[0], plate.box[1]
        return plate.data[box[1] - py0 : box[3] - py0, box[0] - px0 : box[2] - px0]


def plate_channels(mode: SpaceMode) -> int:
    """The components of a plate on a pass of mode: three on the CMY pass, one on the K
    pass, whose three components are the same."""
    return 3 if mode is SpaceMode.CMY else 1


def initial_plates(parent: RasterGroup) -> SpotPlates | None:
    """The plates a non-isolated group pushed onto parent starts from: parent's, or in a
    knockout group, those parent started from, as its backdrop is parent's initial one
    (11.4.8)."""
    spots = parent.spots
    if spots is None:
        return None
    return spots.base if parent.knockout else spots


def child_plates(self: RasterTarget, parent: RasterGroup, isolated: bool) -> SpotPlates | None:
    """The plates of a group pushed onto parent, sharing those it starts from (none for an
    isolated group, or one in another space than a CMYK pass)."""
    if isolated or (
        self.space_mode is not SpaceMode.CMY and self.space_mode is not SpaceMode.BLACK
    ):
        return None
    base = initial_plates(parent)
    return None if base is None else SpotPlates(base, base.channels)


def group_plates(self: RasterTarget, group: RasterGroup) -> SpotPlates:
    """group's plates, made empty when it has none: it started from none."""
    spots = group.spots
    if spots is None:
        spots = SpotPlates(None, plate_channels(self.space_mode))
        frozen_setattr(group, "spots", spots)
    return spots


def paint_marks_off(item: PaintItemBase) -> bool:
    """Whether item does not overprint (its marks None), and so erases spot plates. A path
    painted with a pattern paints it through a group (raster_shading), which composites
    the plates its own paint leaves."""
    if type(item) is PathPaintItem:
        if item.fill_pattern is not None or item.stroke_pattern is not None:
            return False
        if item.paint_kind is PathPaintKind.STROKE:
            return item.stroke_overprint is None
        if item.paint_kind is PathPaintKind.FILL:
            return item.fill_overprint is None
        return item.fill_overprint is None and item.stroke_overprint is None
    from core_pdf.impl.render.raster_overprint import paint_marks  # noqa: PLC0415

    return paint_marks(item) is None


def erases_plates(self: RasterTarget, item: object) -> bool:
    """Whether item, painted by its usual path into the current group, erases spot plates
    there: it does not overprint, and the group has plates where it may paint."""
    spots = self.buffer_stack[-1].spots
    if (
        spots is None
        or not self.overprint_enabled
        or not isinstance(item, PaintItemBase)
        or not paint_marks_off(item)
    ):
        return False
    if (
        type(item) is PathPaintItem
        and item.paint_kind is PathPaintKind.FILL_STROKE
        and self.group_source_shape is not None
    ):
        # paint_typed_path paints its fill and stroke in a knockout group of their own,
        # each erasing there, which then composites the plates.
        return False
    return reaches_plates(self, item, spots)


def reaches_plates(self: RasterTarget, item: PaintItemBase, spots: SpotPlates) -> bool:
    """Whether item may paint where spots has a plate."""
    if type(item) is not PathPaintItem and type(item) is not ImagePaintItem:
        # Other items' boxes are not all on the page's grid: they may reach any plate.
        return True
    # A path's page box takes in its stroke's pen.
    bbox = item.page_box(self.scale)
    if bbox is None:
        return True
    clipped = self.clip.clipped_pixel_box(bbox)
    if clipped is None:
        return False
    return spots.touches(clipped[1])


def paints_apart(self: RasterTarget, item: PathPaintItem) -> bool:
    """Whether a fill and stroke, one of which does not overprint, is painted as its fill
    and then its stroke, for that one alone to erase the spot plates it reaches."""
    spots = self.buffer_stack[-1].spots
    return (
        spots is not None
        and (item.fill_overprint is None) != (item.stroke_overprint is None)
        and reaches_plates(self, item, spots)
    )


def coverage_page(self: RasterTarget) -> bytearray:
    """The blank page a paint that erases plates is drawn on again to find its alpha, zero
    but while it is."""
    page = self.plate_page
    if page is None:
        page = self.plate_page = bytearray(self.width * self.height * 4)
    return page


@cites(
    "ISO 32000-2",
    "11.7.4.5",
    introduced="1.4",
    rule="a paint that does not overprint takes 0 for every spot colourant its space does not name",
)
def erase_under(self: RasterTarget, item: Any) -> None:
    """Once item has painted by its usual path, scale the current group's plates where it
    landed by 1 - alpha_s / alpha_r, as for a component whose source value is 0 (11.3.6).
    Its alpha is found by drawing it again on a blank page under the same clip, so the paint
    itself takes the path it would without plates."""
    group = self.buffer_stack[-1]
    spots = group.spots
    if spots is None:
        return
    plates = spots.union_box
    item_box = None
    if type(item) is PathPaintItem or type(item) is ImagePaintItem:
        # Their page boxes are on the page's grid, a path's taking in its stroke's pen; any
        # other item's is found from the window its drawing widens.
        page_box = item.page_box(self.scale)
        if page_box is not None:
            clipped = self.clip.clipped_pixel_box(page_box)
            if clipped is None:
                return
            item_box = clipped[1]
    if not spots.touches(item_box or (0, 0, self.width, self.height)):
        return
    page = coverage_page(self)
    sibling, view = self.blank_sibling(self.clip, page)
    window = PixelWindow()
    sibling.paint_window = window
    try:
        sibling.paint_display_item(item)
        touched = union(item_box, window_box(window))
        region = intersect(plates, touched or (0, 0, self.width, self.height))
        if region is None:
            return
        x0, y0, x1, y1 = region
        painted = view[y0:y1, x0:x1, 3].copy()
    finally:
        # The paint stays inside its box, or the window it widened; without either, the
        # whole page is cleared.
        touched = union(item_box, window_box(window))
        if touched is None:
            view[...] = 0
        else:
            view[touched[1] : touched[3], touched[0] : touched[2]] = 0
    erase_plates(self, group, slice(y0, y1), slice(x0, x1), painted)


def window_box(window: PixelWindow) -> Box | None:
    bounds = window.bounds()
    if not bounds:
        return None
    y0, y1, x0, x1 = bounds
    return (x0, y0, x1, y1)


def erase_plates(
    self: RasterTarget, group: RasterGroup, rows: slice, columns: slice, painted: UInt8Array
) -> None:
    """Scale group's plates within rows and columns by 1 - alpha_s / alpha_r, painted being
    alpha_s, a byte each."""
    spots = group.spots
    if spots is None:
        return
    window = (columns.start, rows.start, columns.stop, rows.stop)
    result = group.view[rows, columns, 3]
    for key in spots.numbers():
        box = intersect(spots.box(key), window)
        if box is None:
            continue
        values = spots.writable(key, box, self.width, self.height)
        part = (
            slice(box[1] - window[1], box[3] - window[1]),
            slice(box[0] - window[0], box[2] - window[0]),
        )
        alpha = result[part]
        composite_spot_plate(
            values,
            values,
            None,
            None,
            None,
            0,
            painted[part],
            alpha,
            alpha,
            alpha,
            alpha,
            None,
            None,
            ERASED,
        )


def involves_plates(
    self: RasterTarget, parent: RasterGroup, group: RasterGroup, rows: slice, columns: slice
) -> bool:
    """Whether compositing group onto parent within rows and columns may change parent's
    plates: the paint names a spot, or a plate of either reaches there and the group may
    change it. An overprinting paint naming no spot keeps them where its backdrop is opaque
    already, and a group that started from its parent's plates and changed none leaves them,
    in a knockout parent where its initial backdrop is opaque and the parent has not changed
    them there (composite_plates)."""
    if self.space_mode is not SpaceMode.CMY and self.space_mode is not SpaceMode.BLACK:
        return False
    channels = group.overprint
    if channels is not None and channels[3]:
        return True
    window = (columns.start, rows.start, columns.stop, rows.stop)
    spots = parent.spots
    own = group.spots
    if (spots is None or not spots.touches(window)) and (own is None or not own.touches(window)):
        return False
    keeps = channels is not None and not channels[4]
    unchanged = channels is None and own is not None and own.base is not None and not own.own
    if not keeps and not unchanged:
        return True
    if not parent.knockout:
        return not unchanged and int(parent.view[rows, columns, 3].min()) < 255
    if parent.backdrop is None or changed_in(parent, window):
        return True
    return int(self.pixel_view(parent.backdrop)[rows, columns, 3].min()) < 255


class KnockoutElement:
    """What a knockout group composited group's element from (composite_group): the initial
    backdrop's alpha, the element's, and its shape, each over the window."""

    __slots__ = ("element_alpha", "initial_alpha", "shape")

    def __init__(self, initial_alpha: Any, element_alpha: Any, shape: Any) -> None:
        self.initial_alpha = initial_alpha
        self.element_alpha = element_alpha
        self.shape = shape


@cites(
    "ISO 32000-2",
    "11.7.3",
    introduced="1.4",
    rule="a group keeps a separate value for each spot colour component, which composites "
    "with objects using the same spot colour",
)
def composite_plates(
    self: RasterTarget,
    parent: RasterGroup,
    group: RasterGroup,
    rows: slice,
    columns: slice,
    before: UInt8Array,
    effective: UInt8Array,
    knockout: KnockoutElement | None,
) -> None:
    """Composite group's plates, or its one paint's, onto parent's within rows and columns,
    once its colour has composited: before is parent's alpha before, effective the alpha
    group composited with, a byte each. In a knockout parent, the element composites with
    the initial backdrop's plates and replaces what earlier elements left by its shape, as
    its colour does (11.4.7.2); the result is clamped to 0 to 1."""
    channels = group.overprint
    window = (columns.start, rows.start, columns.stop, rows.stop)
    own = group.spots
    # A group that started from its parent's plates (a non-isolated one, or one an
    # overprinting pattern paints through) and changed none of them leaves its parent's as
    # they were, and in a knockout group whose initial backdrop is opaque,
    # wherever the parent has not changed them since.
    unchanged = channels is None and own is not None and own.base is not None
    if (
        unchanged
        and own is not None
        and not own.own
        and (
            knockout is None
            or (
                bool(numpy.all(numpy.asarray(knockout.initial_alpha) == 255))
                and not changed_in(parent, window)
            )
        )
    ):
        return
    if channels is not None and not channels[3] and not channels[4]:
        # An overprinting paint naming no spot keeps every plate where its backdrop was as
        # opaque as the result (alpha_b / alpha_r is 1), and in a knockout group whose
        # initial backdrop is opaque, wherever the group has not changed the plate yet.
        result = parent.view[rows, columns, 3]
        if knockout is None:
            if numpy.array_equal(before, result):
                return
        elif bool(numpy.all(numpy.asarray(knockout.initial_alpha) == 255)) and not changed_in(
            parent, window
        ):
            return
    effective = numpy.asarray(effective)
    for band in bands(rows):
        composite_band(
            self,
            parent,
            group,
            slice(rows.start + band.start, rows.start + band.stop),
            columns,
            before[band],
            effective[band],
            None
            if knockout is None
            else KnockoutElement(
                numpy.asarray(knockout.initial_alpha)[band],
                numpy.asarray(knockout.element_alpha)[band],
                numpy.asarray(knockout.shape)[band],
            ),
        )


def changed_in(group: RasterGroup, window: Box) -> bool:
    """Whether group changed any of the plates it started from within window."""
    spots = group.spots
    return spots is not None and any(
        intersect(plate.extent, window) is not None for plate in spots.own.values()
    )


def composite_band(
    self: RasterTarget,
    parent: RasterGroup,
    group: RasterGroup,
    rows: slice,
    columns: slice,
    before: UInt8Array,
    effective: UInt8Array,
    knockout: KnockoutElement | None,
) -> None:
    """composite_plates within a band of its rows, each plate by composite_spot_plate."""
    window = (columns.start, rows.start, columns.stop, rows.stop)
    result = parent.view[rows, columns, 3]
    if knockout is None:
        backdrop_plates = parent.spots
        backdrop_alpha = before
        element_alpha = result
        cover = None
    else:
        backdrop_plates = initial_plates(parent)
        backdrop_alpha = numpy.asarray(knockout.initial_alpha, dtype=numpy.uint8)
        element_alpha = numpy.asarray(knockout.element_alpha, dtype=numpy.uint8)
        cover = numpy.asarray(knockout.shape, dtype=numpy.float32)
    channels = group.overprint
    layer_key = channels[3] if channels is not None else 0
    own = group.spots
    keys: set[int] = set()
    for spots in (parent.spots, backdrop_plates, own):
        if spots is not None:
            keys |= spots.numbers()
    if layer_key:
        keys.add(layer_key)
    if not keys:
        return
    # Whether an overprinting paint leaves every plate it does not name as it was: its
    # backdrop was as opaque as the result, so alpha_b / alpha_r is 1. In a knockout group
    # whose initial backdrop is opaque, so are the element and the result, which keeps the
    # initial plate wherever the group has not changed it yet.
    keeps = initial_kept = False
    marked = 0
    layer = None
    if channels is not None:
        layer = group.view[rows, columns]
        marked = sum(1 << index for index, mark in enumerate(channels[:3]) if mark)
        if not channels[4]:
            if knockout is None:
                keeps = bool(numpy.array_equal(before, result))
            else:
                initial_kept = bool(numpy.all(backdrop_alpha == 255))
        mode = ERASED if channels[4] else KEPT
    else:
        mode = GROUP
    group_alpha = (
        group.source_alpha[rows, columns]
        if channels is None and group.backdrop is not None and group.source_alpha is not None
        else None
    )
    source = numpy.asarray(effective, dtype=numpy.uint8)
    spots = group_plates(self, parent)
    for key in keys:
        if keeps and key != layer_key:
            continue
        if initial_kept and key != layer_key and parent.spots is not None:
            changed = parent.spots.own.get(key)
            if changed is None or intersect(changed.extent, window) is None:
                continue
        box = union(
            union(
                parent.spots.box(key) if parent.spots is not None else None,
                backdrop_plates.box(key) if backdrop_plates is not None else None,
            ),
            own.box(key) if own is not None else None,
        )
        if key == layer_key:
            box = window
        box = intersect(box, window)
        if box is None:
            continue
        if knockout is None and channels is None and own is not None and own.base is not None:
            # A group that started from its parent's plates leaves those it did not change
            # as they were.
            if key not in own.own:
                continue
            box = intersect(box, own.own[key].extent)
            if box is None:
                continue
        part = (
            slice(box[1] - window[1], box[3] - window[1]),
            slice(box[0] - window[0], box[2] - window[0]),
        )
        below = backdrop_plates.read(key, box) if backdrop_plates is not None else None
        inner = own.read(key, box) if own is not None and channels is None else None
        previous = (
            parent.spots.read(key, box)
            if knockout is not None and parent.spots is not None
            else None
        )
        key_mode = NAMED if key == layer_key else mode
        if (
            key_mode != NAMED
            and (below is None or not below.any())
            and (inner is None or not inner.any())
            and (previous is None or not previous.any())
        ):
            # Erasing, keeping or compositing empty plates leaves the plate empty: nothing to
            # write, and the parent's plate stays the one it started from.
            continue
        values = spots.writable(key, box, self.width, self.height)
        if knockout is not None and previous is None:
            previous = numpy.zeros_like(values)
        composite_spot_plate(
            values,
            below,
            inner,
            previous,
            layer[part] if layer is not None else None,
            marked,
            source[part],
            result[part],
            before[part],
            backdrop_alpha[part],
            element_alpha[part],
            group_alpha[part] if group_alpha is not None else None,
            cover[part] if cover is not None else None,
            key_mode,
        )


@cites(
    "ISO 32000-2",
    "11.7.3",
    introduced="1.4",
    rule="a spot colour's plate prints over the process colorants when its CMYK group ends",
)
def apply_plates(group: RasterGroup) -> None:
    """Add group's plates, a CMYK pass's, to its components, which hold the complements of
    the inks, clamped at full where the group has any alpha; the group then has none."""
    spots = group.spots
    if spots is None:
        return
    frozen_setattr(group, "spots", None)
    view = group.view
    for key in spots.numbers():
        plate = spots.own.get(key) if spots.base is None else None
        box = plate.extent if plate is not None else spots.box(key)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        if plate is not None:
            px0, py0 = plate.box[0], plate.box[1]
            ink = plate.data[y0 - py0 : y1 - py0, x0 - px0 : x1 - px0]
        else:
            ink = spots.read(key, box)
            if ink is None:
                continue
        # A pixel without alpha shows no colour, so it may take the ink too.
        apply_spot_plate(view[y0:y1, x0:x1], ink)

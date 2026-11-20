# SPDX-License-Identifier: AGPL-3.0-only

"""RasterTarget methods: spot colorant plates in a DeviceCMYK blending space (ISO 32000-2
11.7.3, 11.7.4.3, 11.7.4.5).

A group's colour space has no component for a spot colorant, and a press prints the spot
from a plate of its own. "The group shall maintain a separate colour value for each spot
colour component, independently of the group's colour space ... it shall still be subject
to blending and compositing with other objects that use the same spot colour" (11.7.3). A
paint whose Separation or DeviceN space names spot colorants
overprinting in a CMYK pass (raster_overprint) keeps them on such a plate, in the group it
paints into: the tints its alternate colour gives the pass's components that the paint does
not mark, which stand for the spots. Each plate holds one set of spot colorants
(graphics.overprint.spot_plate), and composites as a colour component of the group does
(11.3.6), sharing the group's alpha, by the special blend mode of Table 146:

- a paint naming the plate's spots takes its own tints: "Spot colourant named in source
  space" is Cs, so a spot overprinting the same spot replaces it;
- an overprinting paint naming other colorants keeps the plate: Cb;
- a paint that does not overprint erases it, Cs = 0, as for every colorant it does not name
  (Table 146, "OP false");
- a group, which "is considered to paint all colour components, both process and spot"
  (11.7.4.5, NOTE 2), composites its own plates, empty where it painted none.

When the CMYK group ends, its plates add to its process components, clamped at full, as the
plate of a spot printed over the process inks would, before its colour is converted to its
parent's space: the composite MuPDF's overprint simulation makes of its spot plates. A paint
naming several spots keeps one plate for them all, which a paint naming some of them alone
replaces only where it names exactly the same set; a separable blend mode other than Normal
blends the paint's process components, not its plate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

lazy import numpy

from core_pdf.impl.base.arrays import UInt8Array
from core_pdf.impl.base.types import frozen_setattr
from core_pdf.impl.render.model import RasterGroup
from core_pdf.impl.spec.citations import cites

if TYPE_CHECKING:
    from core_pdf.impl.render.target import RasterTarget

type Plates = dict[int, numpy.ndarray[Any, numpy.dtype[numpy.float32]]]


def inherited_spots(parent: RasterGroup) -> Plates | None:
    """The plates a non-isolated group starts with: its backdrop's (11.4.8)."""
    if not parent.spots:
        return None
    return {plate: values.copy() for plate, values in parent.spots.items()}


def group_plates(self: RasterTarget, group: RasterGroup) -> Plates:
    """group's plates, made empty when it has none."""
    spots = group.spots
    if spots is None:
        spots = {}
        frozen_setattr(group, "spots", spots)
    return spots


def plate(self: RasterTarget, spots: Plates, key: int) -> numpy.ndarray[Any, Any]:
    values = spots.get(key)
    if values is None:
        values = spots[key] = numpy.zeros((self.height, self.width, 3), dtype=numpy.float32)
    return values


@cites(
    "ISO 32000-2",
    "11.7.4.5",
    introduced="1.4",
    rule="an overprinting paint takes the source's value for a spot colourant its space names "
    "and keeps the backdrop's for any other",
)
def composite_spots(
    self: RasterTarget,
    parent: RasterGroup,
    group: RasterGroup,
    window: tuple[slice, slice],
    before: numpy.ndarray[Any, numpy.dtype[numpy.float32]],
    effective: UInt8Array,
) -> None:
    """Composite the plates group's paint leaves onto parent's, within window, once its
    colour has composited: before is parent's alpha before (0 to 1), effective the alpha
    the group composited with (a byte per pixel)."""
    rows, columns = window
    after = parent.view[rows, columns, 3].astype(numpy.float32) / 255.0
    source = numpy.asarray(effective, dtype=numpy.float32) / 255.0
    painted = after > 0.0
    safe = numpy.where(painted, after, 1.0)
    # The share of the result that is the source's, alpha_s / alpha_r (11.3.6).
    share = numpy.where(painted, source / safe, 0.0)[..., None]
    spots = self.group_plates(parent)
    overprint = group.overprint
    if overprint is not None:
        key, erases = overprint[3], overprint[4]
        if key:
            # The paint's tints in the components it does not mark stand for its spots.
            tints = 1.0 - group.view[rows, columns, :3].astype(numpy.float32) / 255.0
            tints[..., numpy.array(overprint[:3])] = 0.0
            values = self.plate(spots, key)[rows, columns]
            values += share * (tints - values)
        # Any other plate: erased under a paint that does not overprint (Cs = 0), kept
        # under one that does (Cb), its colour weighted by alpha_b / alpha_r.
        kept = (1.0 - share) if erases else numpy.where(painted, before / safe, 0.0)[..., None]
        for other, values in spots.items():
            if other != key:
                values[rows, columns] *= kept
        return
    own = group.spots or {}
    if group.backdrop is not None and group.source_alpha is not None:
        # A non-isolated group started from parent's plates: where its own paint shows,
        # parent takes its plates, by how much of that paint survives the group's opacity
        # and mask.
        alpha = group.source_alpha[rows, columns]
        share = numpy.where(
            alpha > 0.0, numpy.clip(source / numpy.maximum(alpha, 1e-6), 0.0, 1.0), 0.0
        )[..., None]
    for key in set(spots) | set(own):
        values = self.plate(spots, key)[rows, columns]
        inner = own.get(key)
        target = inner[rows, columns] if inner is not None else 0.0
        values += share * (target - values)


@cites(
    "ISO 32000-2",
    "11.7.3",
    introduced="1.4",
    rule="a group keeps a separate value for each spot colour component, which composites "
    "with objects using the same spot colour",
)
def apply_spot_plates(view: UInt8Array, spots: Plates | None) -> None:
    """Add a CMYK pass's spot plates to its components, view's three, which hold the
    complements of the tints, clamped at full ink where the pass has any alpha."""
    if not spots:
        return
    total = sum(spots.values())
    assert isinstance(total, numpy.ndarray)
    visible = view[..., 3] > 0
    tints = (255.0 - view[..., :3].astype(numpy.float32)) / 255.0 + total
    combined = 255.0 - numpy.rint(numpy.clip(tints, 0.0, 1.0) * 255.0)
    view[..., :3] = numpy.where(visible[..., None], combined, view[..., :3]).astype(numpy.uint8)

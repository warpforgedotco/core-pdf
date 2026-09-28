from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import chain
from math import inf
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

import numpy

from core_pdf.impl.fonts_helpers import LEGITIMATE_MULTI_CHAR_GLYPHS
from core_pdf.impl.fonts_raster_kernel import (
    rasterize_contours,
)
from core_pdf.impl.geometry import points_bbox
from core_pdf.impl.types import FrozenFields, ReplaceFields, ReprFields, frozen_setattr
from core_pdf_cythonized import cell_distance_map

if TYPE_CHECKING:
    from core_pdf.impl.fonts_program_cff import CFFFont


class CFFGlyphFeature(FrozenFields, ReplaceFields, ReprFields):
    cells: tuple[tuple[int, int], ...]
    aspect: float
    contours: int
    bitmap: tuple[int, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("cells", "aspect", "contours", "bitmap")
    __match_args__ = ("cells", "aspect", "contours", "bitmap")

    def __init__(
        self,
        cells: tuple[tuple[int, int], ...],
        aspect: float,
        contours: int,
        bitmap: tuple[int, ...] = (),
    ) -> None:
        frozen_setattr(self, "cells", cells)
        frozen_setattr(self, "aspect", aspect)
        frozen_setattr(self, "contours", contours)
        frozen_setattr(self, "bitmap", bitmap)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.cells == other.cells
            and self.aspect == other.aspect
            and self.contours == other.contours
            and self.bitmap == other.bitmap
        )

    def __hash__(self) -> int:
        return hash((self.cells, self.aspect, self.contours, self.bitmap))


EMPTY_FEATURE = CFFGlyphFeature((), 0.0, 0, ())


def feature_from_contours(
    contours: tuple[tuple[tuple[float, float], ...], ...] | list[list[tuple[float, float]]],
) -> CFFGlyphFeature:
    if not contours:
        return EMPTY_FEATURE

    coordinates = numpy.fromiter(
        chain.from_iterable(chain.from_iterable(contours)), dtype=numpy.float64
    )
    if not len(coordinates):
        return EMPTY_FEATURE
    if not numpy.isfinite(coordinates).all():
        return feature_from_points(contours)
    xs = coordinates[0::2]
    ys = coordinates[1::2]
    min_x = float(xs.min())
    min_y = float(ys.min())
    width = max(float(xs.max()) - min_x, 1.0)
    height = max(float(ys.max()) - min_y, 1.0)
    cell_x = numpy.rint((xs - min_x) / width * 17).astype(numpy.int64)
    cell_y = numpy.rint((ys - min_y) / height * 23).astype(numpy.int64)
    keys = numpy.unique((cell_x << 5) | cell_y)
    cells = tuple(zip((keys >> 5).tolist(), (keys & 31).tolist(), strict=True))
    bitmap = rasterize_contours(contours, width=18, height=24)
    return CFFGlyphFeature(cells, round(width / height, 2), len(contours), bitmap)


def feature_from_points(
    contours: tuple[tuple[tuple[float, float], ...], ...] | list[list[tuple[float, float]]],
) -> CFFGlyphFeature:
    points = [point for contour in contours for point in contour]
    bbox = points_bbox(points)
    if bbox is None:
        return EMPTY_FEATURE
    min_x, min_y, max_x, max_y = bbox
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    cells: set[tuple[int, int]] = set()
    add_cell = cells.add
    for px, py in points:
        cell_x = round((px - min_x) / width * 17)
        cell_y = round((py - min_y) / height * 23)
        add_cell((cell_x, cell_y))
    bitmap = rasterize_contours(contours, width=18, height=24)
    return CFFGlyphFeature(tuple(sorted(cells)), round(width / height, 2), len(contours), bitmap)


def glyph_feature_distance(left: CFFGlyphFeature, right: CFFGlyphFeature) -> float:
    return feature_distance(
        left.cells,
        left.bitmap,
        left.aspect,
        left.contours,
        right.cells,
        right.bitmap,
        right.aspect,
        right.contours,
    )


SUSPICIOUS_TO_UNICODE = {"\ufffd", "£", "•"}
REPAIRABLE_TO_UNICODE = SUSPICIOUS_TO_UNICODE | {"5", "H"}


def is_repairable_to_unicode_label(label: str) -> bool:
    if len(label) == 1:
        return label in REPAIRABLE_TO_UNICODE
    if label in LEGITIMATE_MULTI_CHAR_GLYPHS:
        return False
    if any(ch in SUSPICIOUS_TO_UNICODE for ch in label):
        return True
    if len(label) > 3:
        return True
    return any(not (ch.isalnum() or ch.isspace()) for ch in label)


def repair_candidate(
    glyph_id: int,
    label: str,
    features: dict[int, CFFGlyphFeature],
    labels: dict[int, str],
    distance_lookup: dict[int, float] | None = None,
) -> str | None:
    feature = features.get(glyph_id, EMPTY_FEATURE)
    if not feature.cells:
        return None
    candidates: list[tuple[float, str]] = []
    same_label = inf
    for other_id, other_label in labels.items():
        if other_id == glyph_id or len(other_label) != 1:
            continue
        if not (other_label.isalnum() or other_label in ".-+"):
            continue
        other_feature = features.get(other_id, EMPTY_FEATURE)
        if not other_feature.cells:
            continue
        distance = (
            distance_lookup[other_id]
            if distance_lookup is not None
            else glyph_feature_distance(feature, other_feature)
        )
        if other_label == label:
            same_label = min(same_label, distance)
        else:
            candidates.append((distance, other_label))
    if not candidates:
        return None
    best_distance, best_label = min(candidates, key=lambda item: item[0])
    if (label in SUSPICIOUS_TO_UNICODE or len(label) > 1) and best_distance < 2.3:
        return best_label
    if label == "5" and best_label == "S" and best_distance < 1.9:
        return best_label
    if label == "H" and best_label == "M" and best_distance < 1.8:
        return best_label
    if same_label < inf and best_distance + 0.35 < same_label and best_distance < 2.0:
        return best_label
    return None


class CFFUnicodeRepairIndex:
    __slots__ = (
        "resolve_candidate_gids",
        "code_to_gid_map",
        "make_font",
        "label_names",
        "repairable_gids",
        "feature_cache",
        "candidate_arrays_cache",
        "decisions",
    )

    def __init__(
        self,
        font: CFFFont,
        mapping_items: tuple[tuple[bytes, int, str], ...],
    ) -> None:
        glyph_count = len(font.charstrings)
        labels: dict[int, str] = {}
        code_to_gid: dict[bytes, int] = {}
        if glyph_count >= 2:
            for code_bytes, cid, value in mapping_items:
                gid = font.glyph_id_for_cid(cid)
                if gid >= glyph_count:
                    continue
                labels[gid] = value
                code_to_gid[code_bytes] = gid

        self.make_font = font
        self.feature_cache: dict[int, CFFGlyphFeature] = {}
        self.candidate_arrays_cache: FeatureArrays | None = None
        self.decisions: dict[int, str | None] = {}
        self.label_names = labels
        self.code_to_gid_map = code_to_gid
        self.repairable_gids = frozenset(
            gid for gid, label in labels.items() if is_repairable_to_unicode_label(label)
        )
        self.resolve_candidate_gids = tuple(
            gid
            for gid, label in labels.items()
            if len(label) == 1 and (label.isalnum() or label in ".-+")
        )

    def repairs_for_codes(self, codes: Iterable[bytes]) -> dict[bytes, str]:
        requested_codes = tuple(dict.fromkeys(codes))
        if not requested_codes or not self.repairable_gids:
            return {}
        target_gids = tuple(
            dict.fromkeys(
                gid
                for code in requested_codes
                if (gid := self.code_to_gid_map.get(code)) in self.repairable_gids
            )
        )
        if not target_gids:
            return {}
        repairs = self.repairs_for_gids(target_gids)
        return {
            code: replacement
            for code in requested_codes
            if (gid := self.code_to_gid_map.get(code)) is not None
            and (replacement := repairs.get(gid)) is not None
        }

    def repairs_for_gids(self, requested_gids: tuple[int, ...]) -> dict[int, str]:
        decisions = self.decisions
        pending = tuple(gid for gid in requested_gids if gid not in decisions)
        if pending:
            decided = self.decide_repairs(pending)
            for gid in pending:
                decisions[gid] = decided.get(gid)
        return {
            gid: replacement
            for gid in requested_gids
            if (replacement := decisions[gid]) is not None
        }

    def decide_repairs(self, requested_gids: tuple[int, ...]) -> dict[int, str]:
        feature_cache = self.feature_cache
        glyph_feature = self.make_font.glyph_feature
        features: dict[int, CFFGlyphFeature] = {}
        for gid in dict.fromkeys((*self.resolve_candidate_gids, *requested_gids)):
            feature = feature_cache.get(gid)
            if feature is None:
                feature = feature_cache[gid] = glyph_feature(gid)
            features[gid] = feature

        candidate_gids = tuple(gid for gid in self.resolve_candidate_gids if features[gid].cells)
        target_gids = tuple(gid for gid in requested_gids if features[gid].cells)
        distance_lookups: dict[int, dict[int, float]] = {}
        if (
            target_gids
            and candidate_gids
            and (len(self.repairable_gids) * len(candidate_gids) >= 512)
        ):
            target_features = [features[gid] for gid in target_gids]
            candidate_features = [features[gid] for gid in candidate_gids]
            candidate_arrays = self.candidate_arrays_cache
            if candidate_arrays is None:
                candidate_arrays = self.candidate_arrays_cache = feature_arrays(
                    [feature.cells for feature in candidate_features],
                    [feature.bitmap for feature in candidate_features],
                    [feature.aspect for feature in candidate_features],
                    [feature.contours for feature in candidate_features],
                )
            distance_matrix = feature_distance_matrix(
                [feature.cells for feature in target_features],
                [feature.bitmap for feature in target_features],
                [feature.aspect for feature in target_features],
                [feature.contours for feature in target_features],
                [feature.cells for feature in candidate_features],
                [feature.bitmap for feature in candidate_features],
                [feature.aspect for feature in candidate_features],
                [feature.contours for feature in candidate_features],
                right_arrays=candidate_arrays,
            )
            distance_lookups = {
                target_gid: {
                    candidate_gid: float(distance_matrix[target_index, candidate_index])
                    for candidate_index, candidate_gid in enumerate(candidate_gids)
                }
                for target_index, target_gid in enumerate(target_gids)
            }

        repairs: dict[int, str] = {}
        for glyph_id in target_gids:
            label = self.label_names[glyph_id]
            replacement = repair_candidate(
                glyph_id,
                label,
                features,
                self.label_names,
                distance_lookups.get(glyph_id),
            )
            if replacement is not None and replacement != label:
                repairs[glyph_id] = replacement
        return repairs


FEATURE_GRID_WIDTH = 18
FEATURE_GRID_HEIGHT = 24
FeatureArrays: TypeAlias = tuple[
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
]


def average_nearest_distance(
    cells: tuple[tuple[int, int], ...], distance_map: tuple[int, ...]
) -> float:
    total = 0.0
    count = 0
    for x, y in cells:
        if 0 <= x < FEATURE_GRID_WIDTH and 0 <= y < FEATURE_GRID_HEIGHT:
            total += distance_map[y * FEATURE_GRID_WIDTH + x]
            count += 1
    return total / count if count else inf


def bitmap_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    intersection = 0
    union = 0
    for left_row, right_row in zip(left, right, strict=True):
        intersection += (left_row & right_row).bit_count()
        union += (left_row | right_row).bit_count()
    if union == 0:
        return 0.0
    return 1.0 - intersection / union


def feature_distance(
    left_cells: tuple[tuple[int, int], ...],
    left_bitmap: tuple[int, ...],
    left_aspect: float,
    left_contours: int,
    right_cells: tuple[tuple[int, int], ...],
    right_bitmap: tuple[int, ...],
    right_aspect: float,
    right_contours: int,
) -> float:
    if not left_cells or not right_cells:
        return inf
    left_map = cell_distance_map(left_cells)
    right_map = cell_distance_map(right_cells)
    return (
        average_nearest_distance(left_cells, right_map)
        + average_nearest_distance(right_cells, left_map)
        + bitmap_distance(left_bitmap, right_bitmap) * 0.75
        + abs(left_aspect - right_aspect) * 2.0
        + abs(left_contours - right_contours) * 0.2
    )


def feature_arrays(
    cells: Sequence[tuple[tuple[int, int], ...]],
    bitmaps: Sequence[tuple[int, ...]],
    aspects: Sequence[float],
    contours: Sequence[int],
) -> FeatureArrays:
    count = len(cells)
    masks = numpy.zeros((count, FEATURE_GRID_HEIGHT, FEATURE_GRID_WIDTH), dtype=numpy.float64)
    distance_maps = numpy.zeros_like(masks)
    valid_counts = numpy.zeros(count, dtype=numpy.float64)
    for index, feature_cells in enumerate(cells):
        valid_cells = tuple(
            (x, y)
            for x, y in feature_cells
            if 0 <= x < FEATURE_GRID_WIDTH and 0 <= y < FEATURE_GRID_HEIGHT
        )
        if not valid_cells:
            continue
        valid_counts[index] = len(valid_cells)
        for x, y in valid_cells:
            masks[index, y, x] += 1.0
        distance_maps[index] = numpy.asarray(
            cell_distance_map(feature_cells), dtype=numpy.float64
        ).reshape(FEATURE_GRID_HEIGHT, FEATURE_GRID_WIDTH)

    bitmap_width = max((len(bitmap) for bitmap in bitmaps), default=0)
    bitmap_rows = numpy.zeros((count, bitmap_width), dtype=numpy.uint64)
    for index, bitmap in enumerate(bitmaps):
        if bitmap:
            bitmap_rows[index, : len(bitmap)] = bitmap
    return (
        masks,
        distance_maps,
        valid_counts,
        numpy.asarray(aspects, dtype=numpy.float64),
        numpy.asarray(contours, dtype=numpy.float64),
        bitmap_rows,
    )


def feature_distance_matrix(
    left_cells: Sequence[tuple[tuple[int, int], ...]],
    left_bitmaps: Sequence[tuple[int, ...]],
    left_aspects: Sequence[float],
    left_contours: Sequence[int],
    right_cells: Sequence[tuple[tuple[int, int], ...]],
    right_bitmaps: Sequence[tuple[int, ...]],
    right_aspects: Sequence[float],
    right_contours: Sequence[int],
    *,
    right_arrays: FeatureArrays | None = None,
) -> numpy.ndarray[Any, Any]:
    (
        left_masks,
        left_maps,
        left_counts,
        left_aspects_array,
        left_contours_array,
        left_bitmap_rows,
    ) = feature_arrays(left_cells, left_bitmaps, left_aspects, left_contours)
    if right_arrays is None:
        right_arrays = feature_arrays(right_cells, right_bitmaps, right_aspects, right_contours)
    (
        right_masks,
        right_maps,
        right_counts,
        right_aspects_array,
        right_contours_array,
        right_bitmap_rows,
    ) = right_arrays

    distance = numpy.full((len(left_cells), len(right_cells)), numpy.inf, dtype=numpy.float64)
    valid = (left_counts[:, None] > 0) & (right_counts[None, :] > 0)
    if not numpy.any(valid):
        return distance

    left_to_right = numpy.einsum("lxy,rxy->lr", left_masks, right_maps)
    left_to_right = numpy.divide(
        left_to_right,
        left_counts[:, None],
        out=numpy.zeros_like(left_to_right),
        where=left_counts[:, None] > 0,
    )
    right_to_left = numpy.einsum("rxy,lxy->rl", right_masks, left_maps).T
    right_to_left = numpy.divide(
        right_to_left,
        right_counts[None, :],
        out=numpy.zeros_like(right_to_left),
        where=right_counts[None, :] > 0,
    )

    if left_bitmap_rows.shape[1] == 0 and right_bitmap_rows.shape[1] == 0:
        bitmap_distance = numpy.zeros_like(distance)
    else:
        bitmap_width = max(left_bitmap_rows.shape[1], right_bitmap_rows.shape[1])
        if left_bitmap_rows.shape[1] != bitmap_width:
            left_bitmap_rows = numpy.pad(
                left_bitmap_rows,
                ((0, 0), (0, bitmap_width - left_bitmap_rows.shape[1])),
            )
        if right_bitmap_rows.shape[1] != bitmap_width:
            right_bitmap_rows = numpy.pad(
                right_bitmap_rows,
                ((0, 0), (0, bitmap_width - right_bitmap_rows.shape[1])),
            )
        intersection = numpy.bitwise_count(
            left_bitmap_rows[:, None, :] & right_bitmap_rows[None, :, :]
        ).sum(axis=2)
        union = numpy.bitwise_count(
            left_bitmap_rows[:, None, :] | right_bitmap_rows[None, :, :]
        ).sum(axis=2)
        same_bitmap_shape = numpy.equal(
            numpy.asarray([len(bitmap) for bitmap in left_bitmaps])[:, None],
            numpy.asarray([len(bitmap) for bitmap in right_bitmaps])[None, :],
        )
        bitmap_ratio = numpy.divide(
            intersection,
            union,
            out=numpy.zeros_like(intersection, dtype=numpy.float64),
            where=union != 0,
        )
        bitmap_distance = numpy.where(
            same_bitmap_shape & (union != 0),
            1.0 - bitmap_ratio,
            0.0,
        )

    combined = (
        left_to_right
        + right_to_left
        + bitmap_distance * 0.75
        + numpy.abs(left_aspects_array[:, None] - right_aspects_array[None, :]) * 2.0
        + numpy.abs(left_contours_array[:, None] - right_contours_array[None, :]) * 0.2
    )
    distance[valid] = combined[valid]
    return distance

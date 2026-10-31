# join_edge_group, words_to_edges_h, words_to_edges_v and edges_to_intersections are
# ported from pdfplumber, Copyright (c) 2015, Jeremy Singer-Vine, under the MIT License.
from collections.abc import Iterable
from operator import itemgetter
from typing import Any

from . import ObjectDict, Table, TableFinder, TableSettings, merge_edges, utils

DEFAULT_JOIN_TOLERANCE = 3
DEFAULT_MIN_WORDS_VERTICAL = 3
DEFAULT_MIN_WORDS_HORIZONTAL = 1


def join_edge_group(
    edges: Iterable[ObjectDict], orientation: str, tolerance: float = DEFAULT_JOIN_TOLERANCE
) -> list[ObjectDict]:
    """
    Given a list of edges along the same infinite line, join those that
    are within `tolerance` pixels of one another.
    """
    if orientation == "h":
        min_prop, max_prop = "x0", "x1"
    elif orientation == "v":
        min_prop, max_prop = "top", "bottom"
    else:
        raise ValueError("Orientation must be 'v' or 'h'")

    sorted_edges = sorted(edges, key=itemgetter(min_prop))
    joined: list[ObjectDict] = [sorted_edges[0]]
    for e in sorted_edges[1:]:
        last = joined[-1]
        if e[min_prop] <= (last[max_prop] + tolerance):
            if e[max_prop] > last[max_prop]:
                # Extend current edge to new extremity
                joined[-1] = utils.resize_object(last, max_prop, e[max_prop])
        else:
            # Edge is separate from previous edges
            joined.append(e)

    return joined


def words_to_edges_h(
    words: list[ObjectDict], word_threshold: int = DEFAULT_MIN_WORDS_HORIZONTAL
) -> list[ObjectDict]:
    """
    Find (imaginary) horizontal lines that connect the tops
    of at least `word_threshold` words.
    """
    by_top = utils.cluster_objects(words, itemgetter("top"), 1)
    large_clusters = filter(lambda x: len(x) >= word_threshold, by_top)
    rects = list(map(utils.objects_to_rect, large_clusters))
    if len(rects) == 0:
        return []
    min_x0 = min(map(itemgetter("x0"), rects))
    max_x1 = max(map(itemgetter("x1"), rects))

    edges = []
    for r in rects:
        edges += [
            # Top of text
            {
                "x0": min_x0,
                "x1": max_x1,
                "top": r["top"],
                "bottom": r["top"],
                "width": max_x1 - min_x0,
                "orientation": "h",
            },
            # For each detected row, we also add the 'bottom' line.  This will
            # generate extra edges, (some will be redundant with the next row
            # 'top' line), but this catches the last row of every table.
            {
                "x0": min_x0,
                "x1": max_x1,
                "top": r["bottom"],
                "bottom": r["bottom"],
                "width": max_x1 - min_x0,
                "orientation": "h",
            },
        ]

    return edges


def words_to_edges_v(
    words: list[ObjectDict], word_threshold: int = DEFAULT_MIN_WORDS_VERTICAL
) -> list[ObjectDict]:
    """
    Find (imaginary) vertical lines that connect the left, right, or
    center of at least `word_threshold` words.
    """
    # Find words that share the same left, right, or centerpoints
    by_x0 = utils.cluster_objects(words, itemgetter("x0"), 1)
    by_x1 = utils.cluster_objects(words, itemgetter("x1"), 1)

    def get_center(word: ObjectDict) -> float:
        return float(word["x0"] + word["x1"]) / 2

    by_center = utils.cluster_objects(words, get_center, 1)
    clusters = by_x0 + by_x1 + by_center

    # Find the points that align with the most words
    sorted_clusters = sorted(clusters, key=lambda x: -len(x))
    large_clusters = filter(lambda x: len(x) >= word_threshold, sorted_clusters)

    # For each of those points, find the bboxes fitting all matching words
    bboxes = list(map(utils.objects_to_bbox, large_clusters))

    # Iterate through those bboxes, condensing overlapping bboxes
    condensed_bboxes: list[Any] = []
    for bbox in bboxes:
        overlap = any(utils.get_bbox_overlap(bbox, c) for c in condensed_bboxes)
        if not overlap:
            condensed_bboxes.append(bbox)

    if len(condensed_bboxes) == 0:
        return []

    condensed_rects = map(utils.bbox_to_rect, condensed_bboxes)
    sorted_rects = sorted(condensed_rects, key=itemgetter("x0"))

    max_x1 = max(map(itemgetter("x1"), sorted_rects))
    min_top = min(map(itemgetter("top"), sorted_rects))
    max_bottom = max(map(itemgetter("bottom"), sorted_rects))

    return [
        {
            "x0": b["x0"],
            "x1": b["x0"],
            "top": min_top,
            "bottom": max_bottom,
            "height": max_bottom - min_top,
            "orientation": "v",
        }
        for b in sorted_rects
    ] + [
        {
            "x0": max_x1,
            "x1": max_x1,
            "top": min_top,
            "bottom": max_bottom,
            "height": max_bottom - min_top,
            "orientation": "v",
        }
    ]


def edges_to_intersections(
    edges: list[ObjectDict], x_tolerance: float = 1, y_tolerance: float = 1
) -> dict[tuple[float, float], dict[str, list[Any]]]:
    """
    Given a list of edges, return the points at which they intersect
    within `tolerance` pixels.
    """
    intersections: dict[tuple[float, float], dict[str, list[Any]]] = {}
    v_edges, h_edges = [list(filter(lambda x: x["orientation"] == o, edges)) for o in ("v", "h")]
    for v in sorted(v_edges, key=itemgetter("x0", "top")):
        for h in sorted(h_edges, key=itemgetter("top", "x0")):
            if (
                (v["top"] <= (h["top"] + y_tolerance))
                and (v["bottom"] >= (h["top"] - y_tolerance))
                and (v["x0"] >= (h["x0"] - x_tolerance))
                and (v["x0"] <= (h["x1"] + x_tolerance))
            ):
                vertex = (v["x0"], h["top"])
                if vertex not in intersections:
                    intersections[vertex] = {"v": [], "h": []}
                intersections[vertex]["v"].append(v)
                intersections[vertex]["h"].append(h)
    return intersections


__all__ = (
    "Table",
    "TableFinder",
    "TableSettings",
    "edges_to_intersections",
    "join_edge_group",
    "merge_edges",
    "words_to_edges_h",
    "words_to_edges_v",
)

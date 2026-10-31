from collections.abc import Iterable

from ...pdfminer.utils import PDFDocEncoding
from .. import (
    BBox,
    ObjectDict,
    _edges,
    bbox_overlap,
    crop_to_bbox,
    decode_psl_list,
    filter_edges,
    intersects_bbox,
    merge_bboxes,
    move_object,
    obj_to_bbox,
    outside_bbox,
    resize_object,
    resolve,
    resolve_all,
    snap_objects,
    to_list,
    within_bbox,
)
from . import exceptions, text
from .clustering import cluster_list, cluster_objects, make_cluster_dict
from .text import (
    DEFAULT_X_DENSITY,
    DEFAULT_X_TOLERANCE,
    DEFAULT_Y_DENSITY,
    DEFAULT_Y_TOLERANCE,
    chars_to_textmap,
    collate_line,
    dedupe_chars,
    extract_text,
    extract_text_simple,
    extract_words,
)


def decode_text(value: bytes | str) -> str:
    if isinstance(value, bytes) and value.startswith(b"\xfe\xff"):
        return str(value[2:], "utf-16be", "ignore")
    try:
        ordinals = (ord(char) if isinstance(char, str) else char for char in value)
        return "".join(PDFDocEncoding[ordinal] for ordinal in ordinals)
    except IndexError:
        return str(value)


get_bbox_overlap = bbox_overlap
obj_to_edges = _edges


def objects_to_bbox(objects: Iterable[ObjectDict]) -> BBox:
    return merge_bboxes([obj_to_bbox(obj) for obj in objects])


def bbox_to_rect(bbox: BBox) -> dict[str, float]:
    return {"x0": bbox[0], "top": bbox[1], "x1": bbox[2], "bottom": bbox[3]}


def objects_to_rect(objects: Iterable[ObjectDict]) -> dict[str, float]:
    return bbox_to_rect(objects_to_bbox(objects))


__all__ = (
    "decode_text",
    "DEFAULT_X_DENSITY",
    "DEFAULT_X_TOLERANCE",
    "DEFAULT_Y_DENSITY",
    "DEFAULT_Y_TOLERANCE",
    "chars_to_textmap",
    "collate_line",
    "dedupe_chars",
    "make_cluster_dict",
    "bbox_to_rect",
    "cluster_list",
    "cluster_objects",
    "crop_to_bbox",
    "decode_psl_list",
    "exceptions",
    "extract_text",
    "extract_text_simple",
    "extract_words",
    "filter_edges",
    "get_bbox_overlap",
    "intersects_bbox",
    "merge_bboxes",
    "move_object",
    "obj_to_bbox",
    "obj_to_edges",
    "objects_to_bbox",
    "objects_to_rect",
    "outside_bbox",
    "resize_object",
    "resolve",
    "resolve_all",
    "snap_objects",
    "text",
    "to_list",
    "within_bbox",
)

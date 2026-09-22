from copy import deepcopy
from typing import Any

import pytest

from core_pdf_compat import pdfplumber as compat

reference = pytest.importorskip("pdfplumber")
pytestmark = pytest.mark.compat_differential


def rectangle(x: float = 10, top: float = 20) -> dict[str, Any]:
    return {
        "object_type": "rect",
        "x0": x,
        "x1": x + 20,
        "top": top,
        "bottom": top + 30,
        "width": 20,
        "height": 30,
        "doctop": top + 200,
        "y0": 150 - top,
        "y1": 180 - top,
    }


@pytest.mark.parametrize("axis", ["h", "v"])
@pytest.mark.parametrize("delta", [-5, 0, 8])
@pytest.mark.parametrize("minimal", [False, True])
def test_move_preserves_size_and_optional_coordinates(
    axis: str, delta: float, minimal: bool
) -> None:
    obj = rectangle()
    if minimal:
        for key in ("doctop", "y0", "y1"):
            del obj[key]
    original = deepcopy(obj)
    assert compat.utils.move_object(obj, axis, delta) == reference.utils.move_object(
        obj, axis, delta
    )
    assert obj == original


@pytest.mark.parametrize(("key", "value"), [("x0", 12), ("x1", 25), ("top", 25), ("bottom", 45)])
@pytest.mark.parametrize("pdf_coordinates", [False, True])
def test_resize_updates_dimensions_and_vertical_coordinates(
    key: str,
    value: float,
    pdf_coordinates: bool,
) -> None:
    obj = rectangle()
    if not pdf_coordinates:
        del obj["y0"], obj["y1"]
    original = deepcopy(obj)
    assert compat.utils.resize_object(obj, key, value) == reference.utils.resize_object(
        obj, key, value
    )
    assert obj == original


@pytest.mark.parametrize("attr", ["x0", "x1", "top", "bottom"])
@pytest.mark.parametrize("tolerance", [0, 3])
def test_snap_moves_whole_objects_in_cluster_order(attr: str, tolerance: float) -> None:
    objects = [rectangle(50, 60), rectangle(10, 20), rectangle(12, 22), rectangle(10, 20)]
    original = deepcopy(objects)
    assert compat.utils.snap_objects(objects, attr, tolerance) == reference.utils.snap_objects(
        objects, attr, tolerance
    )
    assert objects == original


@pytest.mark.parametrize("method", ["within_bbox", "outside_bbox", "crop_to_bbox"])
@pytest.mark.parametrize(
    "bbox",
    [(0, 0, 100, 100), (15, 25, 25, 45), (30, 20, 40, 50), (30, 50, 40, 60), (100, 100, 110, 110)],
)
def test_bbox_selection_and_clipping(method: str, bbox: tuple[int, int, int, int]) -> None:
    objects = [rectangle(), rectangle(50, 60)]
    original = deepcopy(objects)
    assert getattr(compat.utils, method)(objects, bbox) == getattr(reference.utils, method)(
        objects, bbox
    )
    assert objects == original


@pytest.mark.parametrize("tolerance", [0, 2])
@pytest.mark.parametrize("values", [[], [1], [5, 1, 2, 2, 10, 12]])
def test_numeric_clusters_match_reference(values: list[int], tolerance: float) -> None:
    assert compat.utils.cluster_list(iter(values), tolerance) == reference.utils.cluster_list(
        values, tolerance
    )
    objects = [{"position": value, "index": i} for i, value in enumerate(values)]
    assert compat.utils.cluster_objects(
        objects, "position", tolerance
    ) == reference.utils.cluster_objects(objects, "position", tolerance)


@pytest.mark.parametrize("orientation", [None, "h", "v"])
@pytest.mark.parametrize("edge_type", [None, "line", "rect_edge"])
def test_edge_filter_honors_type_orientation_and_minimum_length(
    orientation: str | None,
    edge_type: str | None,
) -> None:
    edges = [
        {**rectangle(), "orientation": "h", "object_type": "line"},
        {**rectangle(), "orientation": "v", "object_type": "rect_edge"},
        {**rectangle(), "orientation": "h", "width": 0.5, "object_type": "line"},
    ]
    options = {"orientation": orientation, "edge_type": edge_type, "min_length": 1}
    assert compat.utils.filter_edges(edges, **options) == reference.utils.filter_edges(
        edges, **options
    )


@pytest.mark.parametrize("orientation", ["h", "v"])
@pytest.mark.parametrize("snap", [0, 1, 3])
@pytest.mark.parametrize("join", [0, 1, 3])
def test_merge_snaps_and_joins_without_mutating_edges(orientation, snap, join):
    from pdfplumber.table import merge_edges

    edges = []
    for start, end, offset in [
        (20, 30, 0),
        (10, 15, 0),
        (14, 18, 1),
        (10, 12, 0),
        (31, 40, 0),
        (50, 60, 4),
    ]:
        edge = rectangle()
        edge.update(orientation=orientation, object_type="line")
        if orientation == "h":
            edge.update(x0=start, x1=end, top=offset, bottom=offset, width=end - start, height=0)
        else:
            edge.update(x0=offset, x1=offset, top=start, bottom=end, width=0, height=end - start)
        edges.append(edge)
    original = deepcopy(edges)
    options = {
        "snap_x_tolerance": snap,
        "snap_y_tolerance": snap,
        "join_x_tolerance": join,
        "join_y_tolerance": join,
    }
    assert compat.merge_edges(iter(edges), **options) == merge_edges(edges, **options)
    assert edges == original


def test_merge_defaults_and_empty_iterators():
    assert compat.merge_edges(iter(())) == []
    with pytest.raises(TypeError):
        options: dict[str, Any] = {"unsupported": True}
        compat.merge_edges([], **options)
    with pytest.raises(ValueError, match="orientation"):
        compat.merge_edges([{"orientation": "diagonal"}])

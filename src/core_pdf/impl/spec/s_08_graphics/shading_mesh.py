# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 8.7.4.5.5: the stream data of triangle and patch mesh shadings (types 4 to 7).

A mesh shading's stream holds its vertices or patches as packed unsigned integers, each
mapped to a number by the Decode array the way an image sample is (8.9.5.2). This reads them
into arrays: the triangles of a free-form (8.7.4.5.5) or lattice-form (8.7.4.5.6) mesh, each
vertex with its point and its values, and the control points and corner values of each
Coons (8.7.4.5.7) or tensor-product (8.7.4.5.8) patch. The values are a vertex's colour
components, or the single parametric t a Function maps to a colour.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

lazy import numpy

from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import require_pdf_number_array
from core_records import GeneratedRecord

COORDINATE_BITS = frozenset({1, 2, 4, 8, 12, 16, 24, 32})
COMPONENT_BITS = frozenset({1, 2, 4, 8, 12, 16})
FLAG_BITS = frozenset({2, 4, 8})

# The order a tensor-product patch's 16 control points come in (8.7.4.5.8), as indexes
# i * 4 + j of p_ij, column i along u and row j along v; a Coons patch gives the first 12.
TENSOR_ORDER = (0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 4, 5, 6, 10, 9)
# For edge flags 1, 2 and 3, the previous patch's points and corners that the new patch's
# p00, p01, p02, p03 and its corners c00, c03 repeat (Tables 84 and 85).
SHARED_POINTS = {1: (3, 7, 11, 15), 2: (15, 14, 13, 12), 3: (12, 8, 4, 0)}
SHARED_CORNERS = {1: (1, 2), 2: (2, 3), 3: (3, 0)}


class MeshData(GeneratedRecord):
    """A mesh's geometry in the shading's target space. triangles is (T, 3, 2 + k): each
    vertex's x, y and k values, for types 4 and 5. patches is (P, 16, 2), p_ij at i * 4 + j,
    and corners (P, 4, k), the values at p00, p03, p33 and p30, for types 6 and 7."""

    triangles: Any
    patches: Any
    corners: Any
    # Each value's Decode range: t's for a Function, else each colour component's.
    ranges: tuple[tuple[float, float], ...]


class BitReader:
    """Unsigned integers read from data most significant bit first."""

    __slots__ = ("data", "position", "limit")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0
        self.limit = len(data) * 8

    def read(self, bits: int) -> int | None:
        end = self.position + bits
        if end > self.limit:
            return None
        first = self.position >> 3
        last = (end + 7) >> 3
        chunk = int.from_bytes(self.data[first:last], "big")
        self.position = end
        return (chunk >> ((last << 3) - end)) & ((1 << bits) - 1)

    def align(self) -> None:
        self.position = (self.position + 7) & ~7


def required_bits(dictionary: dict[Any, Any], key: str, allowed: frozenset[int]) -> int:
    value = dictionary.get(key)
    if type(value) is not int or value not in allowed:
        raise ValueError(f"invalid shading {key}")
    return value


def decode_ranges(dictionary: dict[Any, Any], value_count: int) -> tuple[tuple[float, float], ...]:
    """The Decode array's pairs: x, y, then one per value."""
    decode = require_pdf_number_array(dictionary.get("Decode"), "invalid shading Decode")
    if len(decode) < 4 + 2 * value_count:
        raise ValueError("invalid shading Decode")
    return tuple((decode[2 * index], decode[2 * index + 1]) for index in range(2 + value_count))


def decoded(raw: Any, bits: int, low: float, high: float) -> Any:
    return low + raw * ((high - low) / float((1 << bits) - 1))


@cites(
    "ISO 32000-2",
    "8.7.4.5.5",
    introduced="1.3",
    rule="each vertex of a triangle mesh fills a whole number of bytes",
)
def vertex_fields(data: bytes, field_bits: Sequence[int]) -> list[Any]:
    """Each vertex's fields as uint64 columns: the vertices are packed one after another,
    each padded to a whole byte (8.7.4.5.5); a partial vertex at the end is dropped."""
    record_bits = sum(field_bits)
    record_bytes = (record_bits + 7) // 8
    count = len(data) // record_bytes
    rows = numpy.frombuffer(data, dtype=numpy.uint8, count=count * record_bytes).reshape(
        count, record_bytes
    )
    columns = []
    offset = 0
    for bits in field_bits:
        first = offset >> 3
        last = (offset + bits + 7) >> 3
        chunk = numpy.zeros(count, dtype=numpy.uint64)
        for index in range(first, last):
            chunk = (chunk << numpy.uint64(8)) | rows[:, index].astype(numpy.uint64)
        shift = numpy.uint64((last << 3) - offset - bits)
        mask = numpy.uint64((1 << bits) - 1)
        columns.append((chunk >> shift) & mask)
        offset += bits
    return columns


def vertex_array(
    dictionary: dict[Any, Any], data: bytes, value_count: int, *, flagged: bool
) -> tuple[Any, Any, tuple[tuple[float, float], ...]]:
    """The vertices of a type 4 or 5 mesh as a (V, 2 + k) array, their edge flags (or None),
    and the values' ranges."""
    coordinate_bits = required_bits(dictionary, "BitsPerCoordinate", COORDINATE_BITS)
    component_bits = required_bits(dictionary, "BitsPerComponent", COMPONENT_BITS)
    flag_bits = required_bits(dictionary, "BitsPerFlag", FLAG_BITS) if flagged else 0
    ranges = decode_ranges(dictionary, value_count)
    field_bits = [coordinate_bits, coordinate_bits] + [component_bits] * value_count
    if flagged:
        field_bits.insert(0, flag_bits)
    columns = vertex_fields(data, field_bits)
    flags = columns.pop(0) & numpy.uint64(3) if flagged else None
    count = len(columns[0])
    vertices = numpy.empty((count, 2 + value_count), dtype=numpy.float64)
    for index, column in enumerate(columns):
        bits = coordinate_bits if index < 2 else component_bits
        low, high = ranges[index]
        vertices[:, index] = decoded(column.astype(numpy.float64), bits, low, high)
    return vertices, flags, ranges[2:]


@cites(
    "ISO 32000-2",
    "8.7.4.5.5",
    introduced="1.3",
    rule="a free-form mesh's edge flags join each vertex to the previous triangle",
)
def free_form_triangles(vertices: Any, flags: Any) -> Any:
    """The triangles of a type 4 mesh: flag 0 starts a triangle with the next two vertices,
    1 joins a vertex to side bc of the previous triangle, 2 to side ac."""
    order: list[int] = []
    flag_list = flags.tolist()
    count = len(flag_list)
    index = 0
    a = b = c = -1
    while index < count:
        flag = flag_list[index]
        if flag == 0 or a < 0:
            if index + 2 >= count:
                break
            a, b, c = index, index + 1, index + 2
            index += 3
        elif flag == 1:
            a, b, c = b, c, index
            index += 1
        elif flag == 2:
            a, b, c = a, c, index
            index += 1
        else:
            # A flag of 3 is not valid for a triangle mesh: nothing after it can be placed.
            break
        order.extend((a, b, c))
    width = vertices.shape[1]
    return vertices[numpy.asarray(order, dtype=numpy.intp)].reshape(-1, 3, width)


@cites(
    "ISO 32000-2",
    "8.7.4.5.6",
    introduced="1.3",
    rule="a lattice-form mesh splits each cell of its lattice into two triangles",
)
def lattice_triangles(vertices: Any, per_row: int) -> Any:
    """The triangles of a type 5 mesh: (V[i,j], V[i,j+1], V[i+1,j]) and
    (V[i,j+1], V[i+1,j], V[i+1,j+1]) for each cell of the lattice."""
    rows = len(vertices) // per_row
    width = vertices.shape[1]
    if rows < 2:
        return numpy.empty((0, 3, width), dtype=numpy.float64)
    grid = numpy.arange(rows * per_row).reshape(rows, per_row)
    top_left = grid[:-1, :-1].reshape(-1)
    top_right = grid[:-1, 1:].reshape(-1)
    bottom_left = grid[1:, :-1].reshape(-1)
    bottom_right = grid[1:, 1:].reshape(-1)
    first = numpy.stack((top_left, top_right, bottom_left), axis=1)
    second = numpy.stack((top_right, bottom_left, bottom_right), axis=1)
    order = numpy.stack((first, second), axis=1).reshape(-1)
    return vertices[order].reshape(-1, 3, width)


@cites(
    "ISO 32000-2",
    "8.7.4.5.8",
    introduced="1.3",
    rule="a Coons patch's inner control points follow from its boundary curves",
)
def coons_inner_points(points: list[tuple[float, float]]) -> None:
    """Fill in p11, p12, p21 and p22 of a Coons patch, given as a tensor-product patch
    (8.7.4.5.8); points holds p_ij at i * 4 + j."""

    def combine(*terms: tuple[float, int]) -> tuple[float, float]:
        x = sum(weight * points[index][0] for weight, index in terms) / 9.0
        y = sum(weight * points[index][1] for weight, index in terms) / 9.0
        return (x, y)

    p00, p01, p02, p03 = 0, 1, 2, 3
    p10, p13 = 4, 7
    p20, p23 = 8, 11
    p30, p31, p32, p33 = 12, 13, 14, 15
    points[5] = combine(
        (-4, p00), (6, p01), (6, p10), (-2, p03), (-2, p30), (3, p31), (3, p13), (-1, p33)
    )
    points[6] = combine(
        (-4, p03), (6, p02), (6, p13), (-2, p00), (-2, p33), (3, p32), (3, p10), (-1, p30)
    )
    points[9] = combine(
        (-4, p30), (6, p31), (6, p20), (-2, p33), (-2, p00), (3, p01), (3, p23), (-1, p03)
    )
    points[10] = combine(
        (-4, p33), (6, p32), (6, p23), (-2, p30), (-2, p03), (3, p02), (3, p20), (-1, p00)
    )


@cites(
    "ISO 32000-2",
    "8.7.4.5.7",
    introduced="1.3",
    rule="a patch with edge flag 1, 2 or 3 repeats an edge of the previous patch",
)
def patch_arrays(
    dictionary: dict[Any, Any], data: bytes, value_count: int, *, tensor: bool
) -> tuple[Any, Any, tuple[tuple[float, float], ...]]:
    """The control points (P, 16, 2) and corner values (P, 4, k) of a type 6 or 7 mesh, and
    the values' ranges. Each patch's data is padded to a whole byte, as a vertex's is
    (8.7.4.5.5); a partial patch at the end is dropped."""
    coordinate_bits = required_bits(dictionary, "BitsPerCoordinate", COORDINATE_BITS)
    component_bits = required_bits(dictionary, "BitsPerComponent", COMPONENT_BITS)
    flag_bits = required_bits(dictionary, "BitsPerFlag", FLAG_BITS)
    ranges = decode_ranges(dictionary, value_count)
    (x_low, x_high), (y_low, y_high) = ranges[0], ranges[1]
    coordinate_max = float((1 << coordinate_bits) - 1)
    x_step = (x_high - x_low) / coordinate_max
    y_step = (y_high - y_low) / coordinate_max
    component_max = float((1 << component_bits) - 1)
    component_steps = [(low, (high - low) / component_max) for low, high in ranges[2:]]
    reader = BitReader(data)
    explicit_points = 16 if tensor else 12
    patches: list[list[tuple[float, float]]] = []
    corners: list[list[tuple[float, ...]]] = []
    while True:
        flag = reader.read(flag_bits)
        if flag is None:
            break
        flag &= 3
        if not patches:
            flag = 0
        points: list[tuple[float, float]] = [(0.0, 0.0)] * 16
        values: list[tuple[float, ...]] = [()] * 4
        order = TENSOR_ORDER[:explicit_points]
        corner_order: tuple[int, ...] = (0, 1, 2, 3)
        if flag:
            previous_points = patches[-1]
            previous_values = corners[-1]
            for target, source in zip((0, 1, 2, 3), SHARED_POINTS[flag], strict=True):
                points[target] = previous_points[source]
            first, second = SHARED_CORNERS[flag]
            values[0] = previous_values[first]
            values[1] = previous_values[second]
            order = order[4:]
            corner_order = (2, 3)
        complete = True
        for target in order:
            raw_x = reader.read(coordinate_bits)
            raw_y = reader.read(coordinate_bits)
            if raw_x is None or raw_y is None:
                complete = False
                break
            points[target] = (x_low + raw_x * x_step, y_low + raw_y * y_step)
        if not complete:
            break
        for corner in corner_order:
            components: list[float] = []
            for low, step in component_steps:
                raw = reader.read(component_bits)
                if raw is None:
                    complete = False
                    break
                components.append(low + raw * step)
            if not complete:
                break
            values[corner] = tuple(components)
        if not complete:
            break
        if not tensor:
            coons_inner_points(points)
        reader.align()
        patches.append(points)
        corners.append(values)
    point_array = numpy.asarray(patches, dtype=numpy.float64).reshape(-1, 16, 2)
    corner_array = numpy.asarray(corners, dtype=numpy.float64).reshape(-1, 4, value_count)
    return point_array, corner_array, ranges[2:]


def decode_mesh(
    shading_type: int, dictionary: dict[Any, Any], data: bytes, value_count: int
) -> MeshData:
    """The geometry of a type 4 to 7 shading whose vertices or corners carry value_count
    values each; ValueError when its dictionary is not valid."""
    if shading_type in {4, 5}:
        vertices, flags, ranges = vertex_array(
            dictionary, data, value_count, flagged=shading_type == 4
        )
        if shading_type == 4:
            triangles = free_form_triangles(vertices, flags)
        else:
            per_row = dictionary.get("VerticesPerRow")
            if type(per_row) is not int or per_row < 2:
                raise ValueError("invalid shading VerticesPerRow")
            triangles = lattice_triangles(vertices, per_row)
        return MeshData(numpy.ascontiguousarray(triangles), None, None, ranges)
    if shading_type in {6, 7}:
        patches, corners, ranges = patch_arrays(
            dictionary, data, value_count, tensor=shading_type == 7
        )
        return MeshData(None, patches, corners, ranges)
    raise ValueError("not a mesh shading")


__all__ = ("BitReader", "MeshData", "decode_mesh")

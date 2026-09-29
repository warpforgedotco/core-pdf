# SPDX-License-Identifier: AGPL-3.0-only

import types
import typing

import cython
from cython.cimports.core_pdf_cythonized._glyphs import (
    G_ALTERNATES,
    G_BITMAP_CODE,
    G_CHAR_CODE,
    G_CID,
    G_CODE_BYTES,
    G_GID,
    G_SPLIT_UNICODE,
    G_UNICODE,
    G_UNICODE_SOURCE,
    G_WIDTH_CODE,
    Py_T_OBJECT_EX,
    PyMemberDef,
    PyMemberDescrObject,
)
from cython.cimports.core_pdf_cythonized._pymath import py_max, py_min
from cython.cimports.cpython.dict import PyDict_Check, PyDict_GetItemWithError
from cython.cimports.cpython.object import PyObject
from cython.cimports.cpython.ref import Py_INCREF
from cython.cimports.libc.math import ceil, isnan
from cython.cimports.libc.stdlib import free, malloc

MEMBER_DESCRIPTOR = cython.declare(object, types.MemberDescriptorType)
NO_BOX = cython.declare(cython.double, float("nan"))


GeometryBasis = cython.struct(
    base_x=cython.double,
    base_y=cython.double,
    a=cython.double,
    b=cython.double,
    c=cython.double,
    d=cython.double,
    axis_aligned=cython.bint,
    transform_a=cython.double,
    transform_b=cython.double,
    transform_c=cython.double,
    transform_d=cython.double,
    rise_offset_x=cython.double,
    rise_offset_y=cython.double,
    rise=cython.double,
    ar=cython.double,
    dr=cython.double,
    axis_y0=cython.double,
    axis_y1=cython.double,
    axis_baseline_y=cython.double,
    font_ascent=cython.double,
    font_descent=cython.double,
    font_scale=cython.double,
    advance_scale=cython.double,
    font_size=cython.double,
    visible=cython.bint,
    has_primary=cython.bint,
    has_page=cython.bint,
    px0=cython.double,
    py0=cython.double,
    px1=cython.double,
    py1=cython.double,
    qx0=cython.double,
    qy0=cython.double,
    qx1=cython.double,
    qy1=cython.double,
)


GlyphGeometry = cython.struct(
    ax0=cython.double,
    ay0=cython.double,
    ax1=cython.double,
    ay1=cython.double,
    bx0=cython.double,
    by0=cython.double,
    bx1=cython.double,
    by1=cython.double,
    tx=cython.double,
    ty=cython.double,
    ix0=cython.double,
    iy0=cython.double,
    ix1=cython.double,
    iy1=cython.double,
    visible=cython.int,
    bitmap_width=cython.int,
    bitmap_height=cython.int,
)


@cython.cfunc
@cython.exceptval(-1, check=False)
def load_basis(
    g: cython.pointer[GeometryBasis],
    basis: tuple,
    font_ascent: cython.double,
    font_descent: cython.double,
    rise: cython.double,
    font_scale: cython.double,
    advance_scale: cython.double,
    font_size: cython.double,
    clip_primary: object,
    clip_page: object,
    visible: cython.bint,
) -> cython.int:
    if len(basis) != 6:
        raise ValueError("basis must hold six floats")
    g.base_x = basis[0]
    g.base_y = basis[1]
    g.a = basis[2]
    g.b = basis[3]
    g.c = basis[4]
    g.d = basis[5]
    g.axis_aligned = (g.b == 0.0) and (g.c == 0.0)
    g.transform_a = advance_scale * g.a
    g.transform_b = advance_scale * g.b
    g.transform_c = font_scale * g.c
    g.transform_d = font_scale * g.d
    g.rise_offset_x = rise * g.c
    g.rise_offset_y = rise * g.d
    g.rise = rise
    g.ar = font_ascent + rise
    g.dr = font_descent + rise
    g.axis_y0 = g.base_y + g.dr * g.d
    g.axis_y1 = g.base_y + g.ar * g.d
    swap: cython.double
    if g.axis_y0 > g.axis_y1:
        swap = g.axis_y0
        g.axis_y0 = g.axis_y1
        g.axis_y1 = swap
    g.axis_baseline_y = g.base_y + rise * g.d
    g.font_ascent = font_ascent
    g.font_descent = font_descent
    g.font_scale = font_scale
    g.advance_scale = advance_scale
    g.font_size = font_size
    g.visible = visible
    g.has_primary = clip_primary is not None
    g.has_page = clip_page is not None
    g.px0 = g.py0 = g.px1 = g.py1 = 0.0
    g.qx0 = g.qy0 = g.qx1 = g.qy1 = 0.0
    if g.has_primary:
        g.px0 = clip_primary[0]
        g.py0 = clip_primary[1]
        g.px1 = clip_primary[2]
        g.py1 = clip_primary[3]
    if g.has_page:
        g.qx0 = clip_page[0]
        g.qy0 = clip_page[1]
        g.qx1 = clip_page[2]
        g.qy1 = clip_page[3]
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def glyph_geometry(
    g: cython.pointer[cython.const[GeometryBasis]],
    offset: cython.double,
    advance: cython.double,
    gx0: cython.double,
    gy0: cython.double,
    gx1: cython.double,
    gy1: cython.double,
    want_bitmap: cython.bint,
    out: cython.pointer[GlyphGeometry],
) -> cython.void:
    base_x: cython.double = g.base_x
    base_y: cython.double = g.base_y
    a: cython.double = g.a
    b: cython.double = g.b
    c: cython.double = g.c
    d: cython.double = g.d
    rise: cython.double = g.rise
    ar: cython.double = g.ar
    dr: cython.double = g.dr
    ax0: cython.double
    ax1: cython.double
    abx0: cython.double
    abx1: cython.double
    aby0: cython.double
    aby1: cython.double
    blx0: cython.double
    bly0: cython.double
    blx1: cython.double
    bly1: cython.double
    tx0: cython.double
    tx1: cython.double
    cx0: cython.double
    cx1: cython.double
    cx2: cython.double
    cx3: cython.double
    cy0: cython.double
    cy1: cython.double
    cy2: cython.double
    cy3: cython.double
    ix0: cython.double
    ix1: cython.double
    iy0: cython.double
    iy1: cython.double
    rx0: cython.double
    rx1: cython.double
    ry0: cython.double
    ry1: cython.double
    ex0: cython.double
    ex1: cython.double
    ex2: cython.double
    ex3: cython.double
    ey0: cython.double
    ey1: cython.double
    ey2: cython.double
    ey3: cython.double
    inx0: cython.double
    iny0: cython.double
    inx1: cython.double
    iny1: cython.double
    fb_w: cython.double
    fb_h: cython.double
    r_w: cython.double
    r_h: cython.double
    width: cython.double
    height: cython.double
    size: cython.double
    scaled: cython.double
    ratio: cython.double
    swap: cython.double
    have_box: cython.bint
    vis: cython.int
    bw: cython.int
    bh: cython.int

    if g.axis_aligned:
        ax0 = base_x + offset * a
        ax1 = base_x + (offset + advance) * a
        if ax1 < ax0:
            abx0 = ax1
            abx1 = ax0
        else:
            abx0 = ax0
            abx1 = ax1
        aby0 = g.axis_y0
        aby1 = g.axis_y1
        blx0 = ax0
        bly0 = g.axis_baseline_y
        blx1 = ax1
        bly1 = g.axis_baseline_y
    else:
        tx0 = offset
        tx1 = offset + advance
        cx0 = tx0 * a + dr * c + base_x
        cx1 = tx1 * a + dr * c + base_x
        cx2 = tx0 * a + ar * c + base_x
        cx3 = tx1 * a + ar * c + base_x
        cy0 = tx0 * b + dr * d + base_y
        cy1 = tx1 * b + dr * d + base_y
        cy2 = tx0 * b + ar * d + base_y
        cy3 = tx1 * b + ar * d + base_y
        abx0 = py_min(py_min(py_min(cx0, cx1), cx2), cx3)
        abx1 = py_max(py_max(py_max(cx0, cx1), cx2), cx3)
        aby0 = py_min(py_min(py_min(cy0, cy1), cy2), cy3)
        aby1 = py_max(py_max(py_max(cy0, cy1), cy2), cy3)
        blx0 = base_x + tx0 * a + rise * c
        bly0 = base_y + tx0 * b + rise * d
        blx1 = base_x + tx1 * a + rise * c
        bly1 = base_y + tx1 * b + rise * d

    out.ax0 = abx0
    out.ay0 = aby0
    out.ax1 = abx1
    out.ay1 = aby1
    out.bx0 = blx0
    out.by0 = bly0
    out.bx1 = blx1
    out.by1 = bly1
    out.tx = base_x + offset * a + g.rise_offset_x
    out.ty = base_y + offset * b + g.rise_offset_y

    vis = 0
    if g.visible:
        vis = 1
        if g.has_primary and (abx1 <= g.px0 or abx0 >= g.px1 or aby1 <= g.py0 or aby0 >= g.py1):
            vis = 0
        elif g.has_page and (abx1 <= g.qx0 or abx0 >= g.qx1 or aby1 <= g.qy0 or aby0 >= g.qy1):
            vis = 0
    out.visible = vis

    have_box = not isnan(gx0)
    if (
        have_box
        and g.axis_aligned
        and gx0 == 0.0
        and gy0 * g.font_scale == g.font_descent
        and gx1 * g.advance_scale == advance
        and gy1 * g.font_scale == g.font_ascent
    ):
        inx0 = abx0
        iny0 = aby0
        inx1 = abx1
        iny1 = aby1
    elif (not have_box) or gx1 <= gx0 or gy1 <= gy0:
        inx0 = abx0
        iny0 = aby0
        inx1 = abx1
        iny1 = aby1
    else:
        ix0 = offset + gx0 * g.advance_scale
        ix1 = offset + gx1 * g.advance_scale
        iy0 = rise + gy0 * g.font_scale
        iy1 = rise + gy1 * g.font_scale
        if g.axis_aligned:
            rx0 = ix0 * a + base_x
            rx1 = ix1 * a + base_x
            ry0 = iy0 * d + base_y
            ry1 = iy1 * d + base_y
            if rx1 < rx0:
                swap = rx0
                rx0 = rx1
                rx1 = swap
            if ry1 < ry0:
                swap = ry0
                ry0 = ry1
                ry1 = swap
        else:
            ex0 = ix0 * a + iy0 * c + base_x
            ex1 = ix1 * a + iy0 * c + base_x
            ex2 = ix0 * a + iy1 * c + base_x
            ex3 = ix1 * a + iy1 * c + base_x
            ey0 = ix0 * b + iy0 * d + base_y
            ey1 = ix1 * b + iy0 * d + base_y
            ey2 = ix0 * b + iy1 * d + base_y
            ey3 = ix1 * b + iy1 * d + base_y
            rx0 = py_min(py_min(py_min(ex0, ex1), ex2), ex3)
            rx1 = py_max(py_max(py_max(ex0, ex1), ex2), ex3)
            ry0 = py_min(py_min(py_min(ey0, ey1), ey2), ey3)
            ry1 = py_max(py_max(py_max(ey0, ey1), ey2), ey3)
        fb_w = abx1 - abx0
        fb_h = aby1 - aby0
        r_w = rx1 - rx0
        r_h = ry1 - ry0
        if (
            r_w <= 0.01
            or r_h <= 0.01
            or (fb_w > 0.0 and r_w > fb_w * 4.0)
            or (fb_h > 0.0 and r_h > fb_h * 1.5)
        ):
            inx0 = abx0
            iny0 = aby0
            inx1 = abx1
            iny1 = aby1
        else:
            inx0 = rx0
            iny0 = ry0
            inx1 = rx1
            iny1 = ry1
    out.ix0 = inx0
    out.iy0 = iny0
    out.ix1 = inx1
    out.iy1 = iny1

    bw = 0
    bh = 0
    if want_bitmap:
        bw = 24
        bh = 32
        if have_box:
            width = gx1 - gx0
            height = gy1 - gy0
            if width > 0.0 and height > 0.0:
                size = g.font_size if g.font_size > 1.0 else 1.0
                scaled = ceil(size * 2.5)
                if scaled < 16.0:
                    bh = 16
                elif scaled > 64.0:
                    bh = 64
                else:
                    bh = cython.cast(cython.int, scaled)
                ratio = ceil(bh * width / height)
                if ratio < 1.0:
                    bw = 1
                elif ratio > 96.0:
                    bw = 96
                else:
                    bw = cython.cast(cython.int, ratio)
    out.bitmap_width = bw
    out.bitmap_height = bh


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def union_into(
    union: cython.p_double,
    x0: cython.double,
    y0: cython.double,
    x1: cython.double,
    y1: cython.double,
    first: cython.bint,
) -> cython.void:
    if first:
        union[0] = x0
        union[1] = y0
        union[2] = x1
        union[3] = y1
    else:
        union[0] = union[0] if union[0] < x0 else x0
        union[1] = union[1] if union[1] < y0 else y0
        union[2] = union[2] if union[2] > x1 else x1
        union[3] = union[3] if union[3] > y1 else y1


def horizontal_glyph_geometry(
    offsets: list,
    advances: list,
    glyph_boxes: list,
    basis: tuple,
    font_ascent: cython.double,
    font_descent: cython.double,
    rise: cython.double,
    font_scale: cython.double,
    advance_scale: cython.double,
    font_size: cython.double,
    clip_primary,
    clip_page,
    visible: cython.bint,
    want_bitmap: list,
    want_transform: cython.bint = True,
):
    n: cython.Py_ssize_t = len(offsets)
    i: cython.Py_ssize_t

    if len(advances) != n or len(want_bitmap) != n:
        raise ValueError("offsets, advances and want_bitmap must be the same length")
    if len(glyph_boxes) != 4 * n:
        raise ValueError("glyph_boxes must hold four floats per glyph")

    g = cython.declare(GeometryBasis)
    load_basis(
        cython.address(g),
        basis,
        font_ascent,
        font_descent,
        rise,
        font_scale,
        advance_scale,
        font_size,
        clip_primary,
        clip_page,
        visible,
    )

    off: cython.p_double = cython.cast(cython.p_double, malloc(n * cython.sizeof(cython.double)))
    adv: cython.p_double = cython.cast(cython.p_double, malloc(n * cython.sizeof(cython.double)))
    box: cython.p_double = cython.cast(
        cython.p_double, malloc(4 * n * cython.sizeof(cython.double))
    )
    bmp: cython.p_char = cython.cast(cython.p_char, malloc(n * cython.sizeof(cython.char)))
    if off is cython.NULL or adv is cython.NULL or box is cython.NULL or bmp is cython.NULL:
        free(off)
        free(adv)
        free(box)
        free(bmp)
        raise MemoryError

    out_advance: list = [None] * n
    out_baseline: list = [None] * n
    out_transform: list = [None] * n
    out_ink: list = [None] * n
    out_visible: list = [0] * n
    out_bitmap: list = [0] * (2 * n)
    glyph = cython.declare(GlyphGeometry)
    advance_union = cython.declare(cython.double[4])
    ink_union = cython.declare(cython.double[4])

    try:
        for i in range(n):
            off[i] = offsets[i]
            adv[i] = advances[i]
            bmp[i] = 1 if want_bitmap[i] else 0
        for i in range(4 * n):
            box[i] = glyph_boxes[i]

        for i in range(n):
            glyph_geometry(
                cython.address(g),
                off[i],
                adv[i],
                box[4 * i],
                box[4 * i + 1],
                box[4 * i + 2],
                box[4 * i + 3],
                bmp[i],
                cython.address(glyph),
            )
            out_advance[i] = (glyph.ax0, glyph.ay0, glyph.ax1, glyph.ay1)
            out_baseline[i] = (glyph.bx0, glyph.by0, glyph.bx1, glyph.by1)
            if want_transform:
                out_transform[i] = (
                    g.transform_a,
                    g.transform_b,
                    g.transform_c,
                    g.transform_d,
                    glyph.tx,
                    glyph.ty,
                )
            out_visible[i] = glyph.visible
            out_ink[i] = (glyph.ix0, glyph.iy0, glyph.ix1, glyph.iy1)
            union_into(advance_union, glyph.ax0, glyph.ay0, glyph.ax1, glyph.ay1, i == 0)
            union_into(ink_union, glyph.ix0, glyph.iy0, glyph.ix1, glyph.iy1, i == 0)
            if bmp[i]:
                out_bitmap[2 * i] = glyph.bitmap_width
                out_bitmap[2 * i + 1] = glyph.bitmap_height
    finally:
        free(off)
        free(adv)
        free(box)
        free(bmp)

    return (
        out_advance,
        out_baseline,
        out_transform,
        out_ink,
        out_visible,
        out_bitmap,
        (advance_union[0], advance_union[1], advance_union[2], advance_union[3]) if n else None,
        (ink_union[0], ink_union[1], ink_union[2], ink_union[3]) if n else None,
    )


@cython.cclass
class SlotLayout:
    """Offsets of a class's __slots__ object members, for direct reads and writes."""

    cls = cython.declare(type, visibility="readonly")
    offsets: cython.p_Py_ssize_t
    count = cython.declare(cython.Py_ssize_t, visibility="readonly")

    def __cinit__(self, cls: type, names: tuple):
        self.cls = cls
        self.count = len(names)
        self.offsets = cython.cast(
            cython.p_Py_ssize_t, malloc((self.count or 1) * cython.sizeof(cython.Py_ssize_t))
        )
        if self.offsets is cython.NULL:
            raise MemoryError
        i: cython.Py_ssize_t
        member: cython.pointer[PyMemberDef]
        for i in range(self.count):
            descriptor = getattr(cls, names[i])
            if type(descriptor) is not MEMBER_DESCRIPTOR:
                raise TypeError(f"{cls.__name__}.{names[i]} is not a slot")
            member = cython.cast(cython.pointer[PyMemberDescrObject], descriptor).d_member
            if member.type != Py_T_OBJECT_EX:
                raise TypeError(f"{cls.__name__}.{names[i]} is not an object slot")
            self.offsets[i] = member.offset

    def __dealloc__(self):
        free(self.offsets)

    def build(self, values: tuple):
        """A new instance of the class with the named slots set to values, in order,
        and no __init__ run: for classes whose __init__ only assigns its fields."""
        if len(values) != self.count:
            raise ValueError("one value is needed per slot")
        instance = self.cls.__new__(self.cls)
        index: cython.Py_ssize_t
        for index in range(self.count):
            slot_init(instance, self.offsets[index], values[index])
        return instance


@cython.cfunc
@cython.inline
def slot_get(obj: object, offset: cython.Py_ssize_t) -> object:
    value: cython.pointer[PyObject] = cython.cast(
        cython.pointer[cython.pointer[PyObject]],
        cython.cast(cython.p_char, cython.cast(cython.pointer[PyObject], obj)) + offset,
    )[0]
    if value is cython.NULL:
        raise AttributeError("slot is not set")
    return cython.cast(object, value)


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def slot_init(obj: object, offset: cython.Py_ssize_t, value: object) -> cython.void:
    Py_INCREF(value)
    cython.cast(
        cython.pointer[cython.pointer[PyObject]],
        cython.cast(cython.p_char, cython.cast(cython.pointer[PyObject], obj)) + offset,
    )[0] = cython.cast(cython.pointer[PyObject], value)


# DecodedGlyph fields, in this order: G_CODE_BYTES .. G_GID, declared in _glyphs.pxd.

DECODED_GLYPH_FIELDS = (
    "code_bytes", "width_code", "unicode", "bitmap_code", "split_unicode",
    "unicode_source", "alternates", "char_code", "cid", "gid",
)  # fmt: skip

# GlyphObservation.styled's arguments, in this order.
OBSERVATION_FIELDS = (
    "style", "text", "ink_bbox", "advance_bbox", "seqno", "code_bytes", "char_code", "cid",
    "gid", "font_name", "baseline", "visible", "confidence", "unicode_source", "alternates",
    "bitmap", "bitmap_width", "bitmap_height", "bitmap_code", "glyph_transform", "paint_glyph",
    "cluster_key",
)  # fmt: skip


@cython.cfunc
@cython.inline
@cython.exceptval(-1, check=False)
def wants_glyph_bitmap(text: typing.Optional[str], labels: frozenset) -> cython.bint:
    if len(text) != 1:
        return False
    if text in labels:
        return True
    code: cython.Py_UCS4 = text[0]
    return (0xE000 <= code <= 0xF8FF) or code < 32


def capture_horizontal_glyphs(
    text: str,
    glyphs: tuple,
    glyph_layout: SlotLayout,
    observation_layout: SlotLayout,
    glyph_width,
    glyph_bbox,
    font_size: cython.double,
    char_space,
    word_space,
    horizontal_scale,
    want_render: cython.bint,
    want_runs: cython.bint,
    basis: tuple,
    font_ascent: cython.double,
    font_descent: cython.double,
    rise: cython.double,
    font_scale: cython.double,
    advance_scale: cython.double,
    clip_primary,
    clip_page,
    visible: cython.bint,
    style,
    seqno,
    font_name,
    cluster_start: cython.Py_ssize_t,
    confidence_of,
    suspicious_multi,
    bitmap_labels: frozenset,
    out_glyphs: list,
    out_clusters: list,
):
    """capture_glyphs for horizontal text, or None when a glyph needs its text split.

    glyph_width is a callable from width code to width, or a (widths, default) pair
    that it would compute as widths.get(code, default). confidence_of is a callable,
    or a (cache, callable) pair whose cache maps (text, source, alternates) to what
    the callable returns for them. Appends one observation per
    kept glyph to out_glyphs (and out_clusters when want_runs) and returns
    (kept, advance_union, ink_union, run_confidence).
    """
    n: cython.Py_ssize_t = len(glyphs)
    i: cython.Py_ssize_t
    kept: cython.Py_ssize_t = 0
    cursor: cython.Py_ssize_t = 0
    chunk_length: cython.Py_ssize_t
    glyph_type: type = glyph_layout.cls
    observation_type: type = observation_layout.cls
    gs: cython.p_Py_ssize_t = glyph_layout.offsets
    os: cython.p_Py_ssize_t = observation_layout.offsets
    for i in range(n):
        if type(glyphs[i]) is not glyph_type:
            return None

    g = cython.declare(GeometryBasis)
    load_basis(
        cython.address(g),
        basis,
        font_ascent,
        font_descent,
        rise,
        font_scale,
        advance_scale,
        font_size,
        clip_primary,
        clip_page,
        visible,
    )
    off: cython.p_double = cython.cast(
        cython.p_double, malloc((n or 1) * cython.sizeof(cython.double))
    )
    adv: cython.p_double = cython.cast(
        cython.p_double, malloc((n or 1) * cython.sizeof(cython.double))
    )
    box: cython.p_double = cython.cast(
        cython.p_double, malloc((4 * n or 1) * cython.sizeof(cython.double))
    )
    bmp: cython.p_char = cython.cast(cython.p_char, malloc((n or 1) * cython.sizeof(cython.char)))
    if off is cython.NULL or adv is cython.NULL or box is cython.NULL or bmp is cython.NULL:
        free(off)
        free(adv)
        free(box)
        free(bmp)
        raise MemoryError
    kept_glyphs: list = []
    chunk_texts: list = []
    offset: cython.double = 0.0
    advance: cython.double
    suspicious: cython.bint
    geometry = cython.declare(GlyphGeometry)
    advance_union = cython.declare(cython.double[4])
    ink_union = cython.declare(cython.double[4])
    chunk_text: object
    glyph: object
    spacing: object
    bbox: object
    observation: object
    confidence: object
    run_confidence: object = None
    empty: object = ()
    transform: object
    width_code: object
    width_table: dict = None
    default_width: object = None
    if type(glyph_width) is tuple:
        width_table, default_width = glyph_width
    confidence_cache: object = None
    cached_confidence: cython.pointer[PyObject]
    if type(confidence_of) is tuple:
        confidence_cache, confidence_of = confidence_of
        if not PyDict_Check(confidence_cache):
            raise TypeError("a confidence cache must be a dict")
    try:
        for i in range(n):
            glyph = glyphs[i]
            spacing = char_space + (
                word_space if slot_get(glyph, gs[G_CODE_BYTES]) == b" " else 0.0
            )
            width_code = slot_get(glyph, gs[G_WIDTH_CODE])
            advance = (
                (
                    (
                        width_table.get(width_code, default_width)
                        if width_table is not None
                        else glyph_width(width_code)
                    )
                    * font_size
                    / 1000.0
                    + spacing
                )
                * horizontal_scale
                / 100.0
            )
            chunk_text = slot_get(glyph, gs[G_UNICODE])
            if not chunk_text:
                chunk_text = text[cursor : cursor + 1]
            chunk_length = len(chunk_text)
            cursor += chunk_length if chunk_length > 1 else 1
            if not chunk_text:
                offset += advance
                continue
            suspicious = False if chunk_length == 1 else bool(suspicious_multi(chunk_text))
            if slot_get(glyph, gs[G_SPLIT_UNICODE]) and chunk_length != 1 and not suspicious:
                return None
            bbox = (
                glyph_bbox(slot_get(glyph, gs[G_BITMAP_CODE])) if glyph_bbox is not None else None
            )
            if bbox is None:
                box[4 * kept] = NO_BOX
                box[4 * kept + 1] = NO_BOX
                box[4 * kept + 2] = NO_BOX
                box[4 * kept + 3] = NO_BOX
            else:
                x0, y0, x1, y1 = bbox
                box[4 * kept] = x0
                box[4 * kept + 1] = y0
                box[4 * kept + 2] = x1
                box[4 * kept + 3] = y1
            off[kept] = offset
            adv[kept] = advance
            bmp[kept] = (
                1
                if want_render and (wants_glyph_bitmap(chunk_text, bitmap_labels) or suspicious)
                else 0
            )
            kept_glyphs.append(glyph)
            chunk_texts.append(chunk_text)
            kept += 1
            offset += advance

        for i in range(kept):
            glyph = kept_glyphs[i]
            chunk_text = chunk_texts[i]
            glyph_geometry(
                cython.address(g),
                off[i],
                adv[i],
                box[4 * i],
                box[4 * i + 1],
                box[4 * i + 2],
                box[4 * i + 3],
                bmp[i],
                cython.address(geometry),
            )
            union_into(
                advance_union, geometry.ax0, geometry.ay0, geometry.ax1, geometry.ay1, i == 0
            )
            union_into(ink_union, geometry.ix0, geometry.iy0, geometry.ix1, geometry.iy1, i == 0)
            confidence = None
            if confidence_cache is not None:
                try:
                    cached_confidence = PyDict_GetItemWithError(
                        confidence_cache,
                        (
                            chunk_text,
                            slot_get(glyph, gs[G_UNICODE_SOURCE]),
                            slot_get(glyph, gs[G_ALTERNATES]),
                        ),
                    )
                except TypeError:
                    cached_confidence = cython.NULL
                if cached_confidence is not cython.NULL:
                    confidence = cython.cast(object, cached_confidence)
            if confidence is None:
                confidence = confidence_of(
                    chunk_text,
                    slot_get(glyph, gs[G_UNICODE_SOURCE]),
                    slot_get(glyph, gs[G_ALTERNATES]),
                )
            transform = (
                (
                    g.transform_a,
                    g.transform_b,
                    g.transform_c,
                    g.transform_d,
                    geometry.tx,
                    geometry.ty,
                )
                if want_render
                else None
            )
            observation = observation_type.__new__(observation_type)
            slot_init(observation, os[0], style)
            slot_init(observation, os[1], chunk_text)
            slot_init(observation, os[2], (geometry.ix0, geometry.iy0, geometry.ix1, geometry.iy1))
            slot_init(observation, os[3], (geometry.ax0, geometry.ay0, geometry.ax1, geometry.ay1))
            slot_init(observation, os[4], seqno)
            slot_init(observation, os[5], slot_get(glyph, gs[G_CODE_BYTES]))
            slot_init(observation, os[6], slot_get(glyph, gs[G_CHAR_CODE]))
            slot_init(observation, os[7], slot_get(glyph, gs[G_CID]))
            slot_init(observation, os[8], slot_get(glyph, gs[G_GID]))
            slot_init(observation, os[9], font_name)
            slot_init(observation, os[10], (geometry.bx0, geometry.by0, geometry.bx1, geometry.by1))
            slot_init(observation, os[11], True if geometry.visible else False)
            slot_init(observation, os[12], confidence)
            slot_init(observation, os[13], slot_get(glyph, gs[G_UNICODE_SOURCE]))
            slot_init(observation, os[14], slot_get(glyph, gs[G_ALTERNATES]))
            slot_init(observation, os[15], empty)
            slot_init(observation, os[16], geometry.bitmap_width)
            slot_init(observation, os[17], geometry.bitmap_height)
            slot_init(observation, os[18], slot_get(glyph, gs[G_BITMAP_CODE]) if bmp[i] else None)
            slot_init(observation, os[19], transform)
            slot_init(observation, os[20], True)
            slot_init(observation, os[21], (seqno, cluster_start + i))
            out_glyphs.append(observation)
            if want_runs:
                if run_confidence is None:
                    run_confidence = confidence
                elif confidence is not None:
                    run_confidence = min(run_confidence, confidence)
                out_clusters.append(observation)
    finally:
        free(off)
        free(adv)
        free(box)
        free(bmp)

    if not kept:
        return (0, None, None, None)
    return (
        kept,
        (advance_union[0], advance_union[1], advance_union[2], advance_union[3]),
        (ink_union[0], ink_union[1], ink_union[2], ink_union[3]),
        run_confidence,
    )

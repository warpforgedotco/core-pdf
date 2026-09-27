# SPDX-License-Identifier: AGPL-3.0-only

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from libc.math cimport ceil, isnan
from libc.stdlib cimport free, malloc

from core_pdf_cythonized._pymath cimport py_max, py_min

cdef extern from "Python.h":
    ctypedef struct PyMemberDef:
        const char* name
        int type
        Py_ssize_t offset
    ctypedef struct PyMemberDescrObject:
        PyMemberDef* d_member
    int Py_T_OBJECT_EX

import types

cdef object MEMBER_DESCRIPTOR = types.MemberDescriptorType
cdef double NO_BOX = float("nan")


cdef struct GeometryBasis:
    double base_x, base_y, a, b, c, d
    bint axis_aligned
    double transform_a, transform_b, transform_c, transform_d
    double rise_offset_x, rise_offset_y
    double rise, ar, dr
    double axis_y0, axis_y1, axis_baseline_y
    double font_ascent, font_descent, font_scale, advance_scale, font_size
    bint visible, has_primary, has_page
    double px0, py0, px1, py1, qx0, qy0, qx1, qy1


cdef struct GlyphGeometry:
    double ax0, ay0, ax1, ay1
    double bx0, by0, bx1, by1
    double tx, ty
    double ix0, iy0, ix1, iy1
    int visible, bitmap_width, bitmap_height


cdef int load_basis(
    GeometryBasis* g,
    tuple basis,
    double font_ascent,
    double font_descent,
    double rise,
    double font_scale,
    double advance_scale,
    double font_size,
    clip_primary,
    clip_page,
    bint visible,
) except -1:
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
    cdef double swap
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


cdef void glyph_geometry(
    const GeometryBasis* g,
    double offset,
    double advance,
    double gx0,
    double gy0,
    double gx1,
    double gy1,
    bint want_bitmap,
    GlyphGeometry* out,
) noexcept nogil:
    cdef double base_x = g.base_x, base_y = g.base_y, a = g.a, b = g.b, c = g.c, d = g.d
    cdef double rise = g.rise, ar = g.ar, dr = g.dr
    cdef double ax0, ax1, abx0, abx1, aby0, aby1
    cdef double blx0, bly0, blx1, bly1
    cdef double tx0, tx1, cx0, cx1, cx2, cx3, cy0, cy1, cy2, cy3
    cdef double ix0, ix1, iy0, iy1, rx0, rx1, ry0, ry1
    cdef double ex0, ex1, ex2, ex3, ey0, ey1, ey2, ey3
    cdef double inx0, iny0, inx1, iny1
    cdef double fb_w, fb_h, r_w, r_h, width, height, size, scaled, ratio, swap
    cdef bint have_box
    cdef int vis, bw, bh

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
        if g.has_primary and (
            abx1 <= g.px0 or abx0 >= g.px1 or aby1 <= g.py0 or aby0 >= g.py1
        ):
            vis = 0
        elif g.has_page and (
            abx1 <= g.qx0 or abx0 >= g.qx1 or aby1 <= g.qy0 or aby0 >= g.qy1
        ):
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
                    bh = <int> scaled
                ratio = ceil(bh * width / height)
                if ratio < 1.0:
                    bw = 1
                elif ratio > 96.0:
                    bw = 96
                else:
                    bw = <int> ratio
    out.bitmap_width = bw
    out.bitmap_height = bh


cdef inline void union_into(double* union, double x0, double y0, double x1, double y1, bint first) noexcept nogil:
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
    list offsets,
    list advances,
    list glyph_boxes,
    tuple basis,
    double font_ascent,
    double font_descent,
    double rise,
    double font_scale,
    double advance_scale,
    double font_size,
    clip_primary,
    clip_page,
    bint visible,
    list want_bitmap,
    bint want_transform=True,
):
    cdef Py_ssize_t n = len(offsets)
    cdef Py_ssize_t i

    if len(advances) != n or len(want_bitmap) != n:
        raise ValueError("offsets, advances and want_bitmap must be the same length")
    if len(glyph_boxes) != 4 * n:
        raise ValueError("glyph_boxes must hold four floats per glyph")

    cdef GeometryBasis g
    load_basis(
        &g, basis, font_ascent, font_descent, rise, font_scale, advance_scale, font_size,
        clip_primary, clip_page, visible,
    )

    cdef double *off = <double *> malloc(n * sizeof(double))
    cdef double *adv = <double *> malloc(n * sizeof(double))
    cdef double *box = <double *> malloc(4 * n * sizeof(double))
    cdef char *bmp = <char *> malloc(n * sizeof(char))
    if off is NULL or adv is NULL or box is NULL or bmp is NULL:
        free(off); free(adv); free(box); free(bmp)
        raise MemoryError

    cdef list out_advance = [None] * n
    cdef list out_baseline = [None] * n
    cdef list out_transform = [None] * n
    cdef list out_ink = [None] * n
    cdef list out_visible = [0] * n
    cdef list out_bitmap = [0] * (2 * n)
    cdef GlyphGeometry glyph
    cdef double advance_union[4]
    cdef double ink_union[4]

    try:
        for i in range(n):
            off[i] = offsets[i]
            adv[i] = advances[i]
            bmp[i] = 1 if want_bitmap[i] else 0
        for i in range(4 * n):
            box[i] = glyph_boxes[i]

        for i in range(n):
            glyph_geometry(
                &g, off[i], adv[i], box[4 * i], box[4 * i + 1], box[4 * i + 2], box[4 * i + 3],
                bmp[i], &glyph,
            )
            out_advance[i] = (glyph.ax0, glyph.ay0, glyph.ax1, glyph.ay1)
            out_baseline[i] = (glyph.bx0, glyph.by0, glyph.bx1, glyph.by1)
            if want_transform:
                out_transform[i] = (
                    g.transform_a, g.transform_b, g.transform_c, g.transform_d, glyph.tx, glyph.ty
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


cdef class SlotLayout:
    """Offsets of a class's __slots__ object members, for direct reads and writes."""

    cdef readonly type cls
    cdef Py_ssize_t* offsets
    cdef readonly Py_ssize_t count

    def __cinit__(self, type cls, tuple names):
        self.cls = cls
        self.count = len(names)
        self.offsets = <Py_ssize_t*> malloc((self.count or 1) * sizeof(Py_ssize_t))
        if self.offsets is NULL:
            raise MemoryError
        cdef Py_ssize_t i
        cdef PyMemberDef* member
        for i in range(self.count):
            descriptor = getattr(cls, names[i])
            if type(descriptor) is not MEMBER_DESCRIPTOR:
                raise TypeError(f"{cls.__name__}.{names[i]} is not a slot")
            member = (<PyMemberDescrObject*> descriptor).d_member
            if member.type != Py_T_OBJECT_EX:
                raise TypeError(f"{cls.__name__}.{names[i]} is not an object slot")
            self.offsets[i] = member.offset

    def __dealloc__(self):
        free(self.offsets)


cdef inline object slot_get(object obj, Py_ssize_t offset):
    cdef PyObject* value = (<PyObject**> (<char*> <PyObject*> obj + offset))[0]
    if value is NULL:
        raise AttributeError("slot is not set")
    return <object> value


cdef inline void slot_init(object obj, Py_ssize_t offset, object value) noexcept:
    Py_INCREF(value)
    (<PyObject**> (<char*> <PyObject*> obj + offset))[0] = <PyObject*> value


# DecodedGlyph fields, in this order.
cdef enum:
    G_CODE_BYTES = 0
    G_WIDTH_CODE = 1
    G_UNICODE = 2
    G_BITMAP_CODE = 3
    G_SPLIT_UNICODE = 4
    G_UNICODE_SOURCE = 5
    G_ALTERNATES = 6
    G_CHAR_CODE = 7
    G_CID = 8
    G_GID = 9

DECODED_GLYPH_FIELDS = (
    "code_bytes", "width_code", "unicode", "bitmap_code", "split_unicode",
    "unicode_source", "alternates", "char_code", "cid", "gid",
)

# GlyphObservation.styled's arguments, in this order.
OBSERVATION_FIELDS = (
    "style", "text", "ink_bbox", "advance_bbox", "seqno", "code_bytes", "char_code", "cid",
    "gid", "font_name", "baseline", "visible", "confidence", "unicode_source", "alternates",
    "bitmap", "bitmap_width", "bitmap_height", "bitmap_code", "glyph_transform", "paint_glyph",
    "cluster_key",
)


cdef inline bint wants_glyph_bitmap(str text, frozenset labels) except -1:
    if len(text) != 1:
        return False
    if text in labels:
        return True
    cdef Py_UCS4 code = text[0]
    return (0xE000 <= code <= 0xF8FF) or code < 32


def capture_horizontal_glyphs(
    str text,
    tuple glyphs,
    SlotLayout glyph_layout,
    SlotLayout observation_layout,
    glyph_width,
    glyph_bbox,
    double font_size,
    char_space,
    word_space,
    horizontal_scale,
    bint want_render,
    bint want_runs,
    tuple basis,
    double font_ascent,
    double font_descent,
    double rise,
    double font_scale,
    double advance_scale,
    clip_primary,
    clip_page,
    bint visible,
    style,
    seqno,
    font_name,
    Py_ssize_t cluster_start,
    confidence_of,
    suspicious_multi,
    frozenset bitmap_labels,
    list out_glyphs,
    list out_clusters,
):
    """capture_glyphs for horizontal text, or None when a glyph needs its text split.

    Appends one observation per kept glyph to out_glyphs (and out_clusters when
    want_runs) and returns (kept, advance_union, ink_union, run_confidence).
    """
    cdef Py_ssize_t n = len(glyphs)
    cdef Py_ssize_t i, kept = 0, cursor = 0, chunk_length
    cdef type glyph_type = glyph_layout.cls
    cdef type observation_type = observation_layout.cls
    cdef Py_ssize_t* gs = glyph_layout.offsets
    cdef Py_ssize_t* os = observation_layout.offsets
    for i in range(n):
        if type(glyphs[i]) is not glyph_type:
            return None

    cdef GeometryBasis g
    load_basis(
        &g, basis, font_ascent, font_descent, rise, font_scale, advance_scale, font_size,
        clip_primary, clip_page, visible,
    )
    cdef double* off = <double*> malloc((n or 1) * sizeof(double))
    cdef double* adv = <double*> malloc((n or 1) * sizeof(double))
    cdef double* box = <double*> malloc((4 * n or 1) * sizeof(double))
    cdef char* bmp = <char*> malloc((n or 1) * sizeof(char))
    if off is NULL or adv is NULL or box is NULL or bmp is NULL:
        free(off); free(adv); free(box); free(bmp)
        raise MemoryError
    cdef list kept_glyphs = []
    cdef list chunk_texts = []
    cdef double offset = 0.0, advance
    cdef bint suspicious
    cdef GlyphGeometry geometry
    cdef double advance_union[4]
    cdef double ink_union[4]
    cdef object chunk_text, glyph, spacing, bbox, observation, confidence
    cdef object run_confidence = None
    cdef object empty = ()
    cdef object transform
    try:
        for i in range(n):
            glyph = glyphs[i]
            spacing = char_space + (word_space if slot_get(glyph, gs[G_CODE_BYTES]) == b" " else 0.0)
            advance = (
                glyph_width(slot_get(glyph, gs[G_WIDTH_CODE])) * font_size / 1000.0 + spacing
            ) * horizontal_scale / 100.0
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
            bbox = glyph_bbox(slot_get(glyph, gs[G_BITMAP_CODE])) if glyph_bbox is not None else None
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
            bmp[kept] = 1 if want_render and (
                wants_glyph_bitmap(chunk_text, bitmap_labels) or suspicious
            ) else 0
            kept_glyphs.append(glyph)
            chunk_texts.append(chunk_text)
            kept += 1
            offset += advance

        for i in range(kept):
            glyph = kept_glyphs[i]
            chunk_text = chunk_texts[i]
            glyph_geometry(
                &g, off[i], adv[i], box[4 * i], box[4 * i + 1], box[4 * i + 2], box[4 * i + 3],
                bmp[i], &geometry,
            )
            union_into(advance_union, geometry.ax0, geometry.ay0, geometry.ax1, geometry.ay1, i == 0)
            union_into(ink_union, geometry.ix0, geometry.iy0, geometry.ix1, geometry.iy1, i == 0)
            confidence = confidence_of(
                chunk_text,
                slot_get(glyph, gs[G_UNICODE_SOURCE]),
                slot_get(glyph, gs[G_ALTERNATES]),
            )
            transform = (
                (g.transform_a, g.transform_b, g.transform_c, g.transform_d, geometry.tx, geometry.ty)
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

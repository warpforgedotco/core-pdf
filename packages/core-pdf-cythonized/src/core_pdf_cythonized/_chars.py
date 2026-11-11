# SPDX-License-Identifier: AGPL-3.0-only

import cython
from cython.cimports.core_pdf_cythonized._pymath import py_max, py_min

DECLINE = cython.declare(object, object())
ANNOTATION = cython.declare(tuple, ("source", "annotation_appearance"))


@cython.cfunc
@cython.inline
def exact_float(value: object, out: cython.p_double) -> cython.bint:
    # Only a float: Python's arithmetic on ints keeps ints, which doubles would not.
    if type(value) is float:
        out[0] = cython.cast(cython.double, value)
        return True
    return False


@cython.cfunc
@cython.inline
def exact_operand(value: object, out: cython.p_double) -> cython.bint:
    # A float, or an int a double holds exactly: the page's ctm and box, which only ever
    # meet a glyph's floats, so Python turns them into the same double first.
    if type(value) is int and -(2**53) <= value <= 2**53:
        out[0] = cython.cast(cython.double, value)
        return True
    return exact_float(value, out)


@cython.cfunc
def cached_by_identity(cache: dict, key: object, compute: object, argument: object) -> object:
    # compute(argument), kept under id(key) beside key so a reused id is not taken for it.
    known = cache.get(id(key))
    if known is None or cython.cast(tuple, known)[0] is not key:
        known = (key, compute(argument))
        cache[id(key)] = known
    return cython.cast(tuple, known)[1]


def plumber_char_dicts(
    glyphs,
    marks,
    skipped: set,
    ligatures: dict,
    embedded_unusable,
    glyph_text,
    normalized_width,
    descent_of,
    font_name,
    ctm: tuple,
    page_number,
    media_x0_value,
    media_top_value,
    page_height_value,
    doctop_value,
    nfc: cython.bint,
):
    """core_pdf_compat.pdfplumber's char_objects for horizontal text, or None.

    Writes each glyph's char dict as the Python loop does, with the same arithmetic in the
    same order, and calls back for what it caches: whether a decoder's embedded CMap is
    unusable, a glyph's text, width and descent, a decoder's font name. None, having
    returned nothing, for a page with vertical text or with a number that is not a float,
    whose dicts the Python loop writes.
    """
    pa = cython.declare(cython.double)
    pb = cython.declare(cython.double)
    pc = cython.declare(cython.double)
    pd = cython.declare(cython.double)
    pe = cython.declare(cython.double)
    pf = cython.declare(cython.double)
    media_x0 = cython.declare(cython.double)
    media_top = cython.declare(cython.double)
    page_height = cython.declare(cython.double)
    doctop = cython.declare(cython.double)
    a = cython.declare(cython.double)
    b = cython.declare(cython.double)
    c = cython.declare(cython.double)
    d = cython.declare(cython.double)
    e = cython.declare(cython.double)
    f = cython.declare(cython.double)
    font_size = cython.declare(cython.double)
    rise = cython.declare(cython.double)
    scaling = cython.declare(cython.double)
    descent_unit = cython.declare(cython.double)
    width = cython.declare(cython.double)
    base_x = cython.declare(cython.double)
    base_y = cython.declare(cython.double)
    if len(ctm) != 6 or not (
        exact_operand(ctm[0], cython.address(pa))
        and exact_operand(ctm[1], cython.address(pb))
        and exact_operand(ctm[2], cython.address(pc))
        and exact_operand(ctm[3], cython.address(pd))
        and exact_operand(ctm[4], cython.address(pe))
        and exact_operand(ctm[5], cython.address(pf))
        and exact_operand(media_x0_value, cython.address(media_x0))
        and exact_operand(media_top_value, cython.address(media_top))
        and exact_operand(page_height_value, cython.address(page_height))
        and exact_operand(doctop_value, cython.address(doctop))
    ):
        return None
    objects: list = []
    glyph_texts: dict = {}
    widths: dict = {}
    descents: dict = {}
    font_names: dict = {}
    unusable: dict = {}
    provenances: dict = {}
    mark_count: cython.Py_ssize_t = len(marks)
    mark_position: cython.Py_ssize_t = -1
    glyph_index: cython.Py_ssize_t
    count: cython.Py_ssize_t = len(glyphs)
    mark: object
    matrix_value: object
    ma: cython.double
    mb: cython.double
    mc: cython.double
    md: cython.double
    me: cython.double
    mf: cython.double
    left: cython.double
    bottom: cython.double
    right: cython.double
    top: cython.double
    x0: cython.double
    x1: cython.double
    y0: cython.double
    y1: cython.double
    advance: cython.double
    descent: cython.double
    char_x0: cython.double
    char_top: cython.double
    char_x1: cython.double
    char_bottom: cython.double
    for glyph_index in range(count):
        glyph = glyphs[glyph_index]
        decoder = glyph.font_decoder
        if cached_by_identity(unusable, decoder, embedded_unusable, glyph):
            continue
        if id(glyph) in skipped:
            continue
        ligature = ligatures.get(id(glyph))
        key = (
            decoder,
            glyph.char_code,
            glyph.cid,
            glyph.code_bytes,
            glyph.unicode_source,
            glyph.text,
            glyph.alternates,
        )
        if ligature is not None:
            text = ligature[0]
        else:
            text = glyph_texts.get(key)
            if text is None:
                text = glyph_texts[key] = glyph_text(glyph)
        provenance = glyph.provenance
        known = provenances.get(id(provenance))
        if known is None or cython.cast(tuple, known)[0] is not provenance:
            known = provenances[id(provenance)] = (provenance, ANNOTATION in provenance)
        if cython.cast(tuple, known)[1]:
            continue
        if not mark_count:
            continue
        while mark_position + 1 < mark_count and marks[mark_position + 1][0] <= glyph_index:
            mark_position += 1
        mark = marks[mark_position]
        if decoder.is_vertical:
            return None
        matrix_value = mark[6]
        if not (
            len(matrix_value) == 6
            and exact_float(matrix_value[0], cython.address(a))
            and exact_float(matrix_value[1], cython.address(b))
            and exact_float(matrix_value[2], cython.address(c))
            and exact_float(matrix_value[3], cython.address(d))
            and exact_float(matrix_value[4], cython.address(e))
            and exact_float(matrix_value[5], cython.address(f))
            and exact_float(glyph.font_size, cython.address(font_size))
            and exact_float(mark[7], cython.address(rise))
            and exact_float(mark[8], cython.address(scaling))
        ):
            return None
        baseline = ligature[2] if ligature is not None else glyph.baseline
        if glyph_index != mark[0] and baseline is not None:
            if not (
                exact_float(baseline[0], cython.address(base_x))
                and exact_float(baseline[1], cython.address(base_y))
            ):
                return None
            e = base_x - rise * c
            f = base_y - rise * d
        if not exact_float(
            cached_by_identity(descents, decoder, descent_of, glyph), cython.address(descent_unit)
        ):
            return None
        descent = descent_unit * font_size
        projected = None if ligature else text
        width_key = (*key, projected)
        width_value = widths.get(width_key)
        if width_value is None:
            width_value = widths[width_key] = normalized_width(glyph, projected)
        if not exact_float(width_value, cython.address(width)):
            return None
        advance = width * font_size * scaling
        left = 0.0
        bottom = descent + rise
        right = advance
        top = descent + rise + font_size
        ma = a * pa + b * pc
        mb = a * pb + b * pd
        mc = c * pa + d * pc
        md = c * pb + d * pd
        me = e * pa + f * pc + pe
        mf = e * pb + f * pd + pf
        # pdfminer.six's apply_matrix_rect: the box around all four placed corners.
        x0 = py_min(
            py_min(py_min(ma * left + mc * bottom + me, ma * left + mc * top + me),
                   ma * right + mc * bottom + me),
            ma * right + mc * top + me,
        )  # fmt: skip
        x1 = py_max(
            py_max(py_max(ma * left + mc * bottom + me, ma * left + mc * top + me),
                   ma * right + mc * bottom + me),
            ma * right + mc * top + me,
        )  # fmt: skip
        y0 = py_min(
            py_min(py_min(mb * left + md * bottom + mf, mb * left + md * top + mf),
                   mb * right + md * bottom + mf),
            mb * right + md * top + mf,
        )  # fmt: skip
        y1 = py_max(
            py_max(py_max(mb * left + md * bottom + mf, mb * left + md * top + mf),
                   mb * right + md * bottom + mf),
            mb * right + md * top + mf,
        )  # fmt: skip
        if nfc and text == ";":
            text = ";"
        char_x0 = x0 + media_x0
        char_top = page_height - y1 + media_top
        char_x1 = x1 + media_x0
        char_bottom = page_height - y0 + media_top
        objects.append(
            {
                "object_type": "char",
                "page_number": page_number,
                "x0": char_x0,
                "x1": char_x1,
                "top": char_top,
                "bottom": char_bottom,
                "doctop": doctop + char_top,
                "width": char_x1 - char_x0,
                "height": char_bottom - char_top,
                "text": text,
                "y0": y0,
                "y1": y1,
                "fontname": cached_by_identity(font_names, decoder, font_name, decoder),
                "size": y1 - y0,
                "stroking_color": mark[4],
                "non_stroking_color": mark[5],
                "upright": ma * md * scaling > 0 and mb * mc <= 0,
                "adv": advance,
                "seqno": glyph.seqno,
                "matrix": (ma, mb, mc, md, me, mf),
                "ncs": mark[3],
                "mcid": mark[1],
                "tag": mark[2],
            }
        )
    return objects

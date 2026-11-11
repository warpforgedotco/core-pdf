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


# A glyph pdfminer_chars leaves to the Python loop, and one that makes no char.
LEFT = cython.declare(object, None)
NO_CHAR = cython.declare(object, object())


def no_char() -> object:
    """The marker pdfminer_chars returns for a glyph that makes no char."""
    return NO_CHAR


@cython.cfunc
def provenance_entry(cache: dict, provenance: object, is_clipped: object) -> object:
    # What a glyph's provenance gives every glyph that shares it: whether a form clips it
    # away, its four-value text matrix as floats (None for any other), its horizontal scale
    # and its text rise -- or None, which leaves its glyphs to the Python loop, where a
    # value that does not convert raises as it always has.
    known = cache.get(id(provenance))
    if known is not None and cython.cast(tuple, known)[0] is provenance:
        return cython.cast(tuple, known)[1]
    entry: object = None
    try:
        values = dict(provenance)
        clipped = bool(is_clipped(values))
        text_matrix = values.get("text_matrix")
        floats = None
        if isinstance(text_matrix, (tuple, list)) and len(text_matrix) == 4:
            floats = tuple(map(float, text_matrix))
        entry = (
            clipped,
            floats,
            float(values.get("horizontal_scale", 100.0)) * 0.01,
            float(values.get("text_rise", 0.0) or 0.0),
        )
    except Exception:  # noqa: BLE001
        entry = None
    cache[id(provenance)] = (provenance, entry)
    return entry


def pdfminer_chars(
    glyphs,
    glyph_matrices: dict,
    strict_resources: cython.bint,
    skipped: set,
    ligatures: dict,
    embedded_unusable,
    is_clipped,
    glyph_text,
    projected_widths,
    descent_of,
    font_name,
    char_type,
    empty_glyph_text,
) -> list:
    """core_pdf_compat.pdfminer's project_page chars for plain horizontal text.

    One entry per glyph: the LTChar the Python loop makes for it, no_char() for a glyph that
    makes none, or None for a glyph left to the loop -- vertical text, a text matrix other
    than four values, a glyph pdfminer's own char matrix does not place, one right after
    vertical text, one the centring rule for a lone wide flat glyph moves, a provenance
    that does not convert, or a number that is not a float. The arithmetic is the loop's,
    in its order.
    """
    count: cython.Py_ssize_t = len(glyphs)
    results: list = [LEFT] * count
    glyph_index: cython.Py_ssize_t
    glyph_texts: dict = {}
    widths: dict = {}
    descents: dict = {}
    font_names: dict = {}
    unusable: dict = {}
    provenances: dict = {}
    font_size = cython.declare(cython.double)
    width = cython.declare(cython.double)
    descent_unit = cython.declare(cython.double)
    char_a = cython.declare(cython.double)
    char_b = cython.declare(cython.double)
    char_c = cython.declare(cython.double)
    char_d = cython.declare(cython.double)
    char_e = cython.declare(cython.double)
    char_f = cython.declare(cython.double)
    box_x0 = cython.declare(cython.double)
    box_y0 = cython.declare(cython.double)
    box_x1 = cython.declare(cython.double)
    box_y1 = cython.declare(cython.double)
    horizontal_scale: cython.double
    rise: cython.double
    descent: cython.double
    advance: cython.double
    low: cython.double
    high: cython.double
    x0: cython.double
    y0: cython.double
    x1: cython.double
    y1: cython.double
    for glyph_index in range(count):
        glyph = glyphs[glyph_index]
        decoder = glyph.font_decoder
        if strict_resources and cached_by_identity(unusable, decoder, embedded_unusable, glyph):
            results[glyph_index] = NO_CHAR
            continue
        provenance = glyph.provenance
        if not provenance:
            continue
        entry = provenance_entry(provenances, provenance, is_clipped)
        if entry is None:
            continue
        if cython.cast(tuple, entry)[0]:
            results[glyph_index] = NO_CHAR
            continue
        if id(glyph) in skipped:
            results[glyph_index] = NO_CHAR
            continue
        if not glyph.text:
            results[glyph_index] = NO_CHAR
            continue
        ligature = ligatures.get(id(glyph))
        box = ligature[1] if ligature is not None else glyph.advance_bbox
        if len(box) != 4:
            continue
        baseline = ligature[2] if ligature is not None else glyph.baseline
        key = (
            decoder,
            glyph.char_code,
            glyph.cid,
            glyph.code_bytes,
            glyph.unicode_source,
            glyph.text,
            glyph.alternates,
        )
        text_of_glyph = None
        if ligature is None:
            text_of_glyph = glyph_texts.get(key)
            if text_of_glyph is None:
                text_of_glyph = glyph_texts[key] = glyph_text(glyph)
        text = (ligature[0] if ligature is not None else text_of_glyph) or ""
        if not text and glyph.text != empty_glyph_text:
            results[glyph_index] = NO_CHAR
            continue
        if getattr(decoder, "is_vertical", False) or baseline is None:
            continue
        floats = cython.cast(tuple, entry)[1]
        if floats is None:
            continue
        if glyph_index and glyphs[glyph_index - 1].font_decoder.is_vertical:
            continue
        if not exact_float(glyph.font_size, cython.address(font_size)):
            continue
        # The centring rule for a lone wide flat glyph keeps a count the loop must update.
        if (
            glyph.baseline is not None
            and font_size > 0
            and not glyph.effective_font_size
            and exact_float(box[0], cython.address(box_x0))
            and exact_float(box[1], cython.address(box_y0))
            and exact_float(box[2], cython.address(box_x1))
            and exact_float(box[3], cython.address(box_y1))
            and box_x1 - box_x0 >= font_size * 0.8
            and box_y1 - box_y0 <= font_size * 0.1
        ):
            continue
        if not (
            exact_float(box[0], cython.address(box_x0))
            and exact_float(box[1], cython.address(box_y0))
            and exact_float(box[2], cython.address(box_x1))
            and exact_float(box[3], cython.address(box_y1))
        ):
            continue
        exact_matrix = glyph_matrices.get(glyph_index)
        if exact_matrix is None or len(exact_matrix) != 6:
            continue
        if not (
            exact_float(exact_matrix[0], cython.address(char_a))
            and exact_float(exact_matrix[1], cython.address(char_b))
            and exact_float(exact_matrix[2], cython.address(char_c))
            and exact_float(exact_matrix[3], cython.address(char_d))
            and exact_float(exact_matrix[4], cython.address(char_e))
            and exact_float(exact_matrix[5], cython.address(char_f))
        ):
            continue
        width_code = (
            glyph.cid
            if getattr(decoder, "is_cid_font", False)
            else glyph.char_code
            if glyph.char_code is not None
            else glyph.cid
        )
        widths_key = (*key, text_of_glyph is None)
        known_widths = widths.get(widths_key)
        if known_widths is None:
            known_widths = widths[widths_key] = projected_widths(glyph, width_code, text_of_glyph)
        if not exact_float(known_widths[1], cython.address(width)):
            continue
        if not exact_float(
            cached_by_identity(descents, decoder, descent_of, glyph), cython.address(descent_unit)
        ):
            continue
        horizontal_scale = cython.cast(tuple, entry)[2]
        rise = cython.cast(tuple, entry)[3]
        # pdfminer's own arithmetic, from LTChar: its rounding decides which boxes touch.
        descent = descent_unit * font_size
        advance = width * font_size * horizontal_scale
        low = descent + rise
        high = descent + rise + font_size
        x0 = py_min(
            py_min(py_min(char_a * 0.0 + char_c * low + char_e,
                          char_a * advance + char_c * low + char_e),
                   char_a * advance + char_c * high + char_e),
            char_a * 0.0 + char_c * high + char_e,
        )  # fmt: skip
        x1 = py_max(
            py_max(py_max(char_a * 0.0 + char_c * low + char_e,
                          char_a * advance + char_c * low + char_e),
                   char_a * advance + char_c * high + char_e),
            char_a * 0.0 + char_c * high + char_e,
        )  # fmt: skip
        y0 = py_min(
            py_min(py_min(char_b * 0.0 + char_d * low + char_f,
                          char_b * advance + char_d * low + char_f),
                   char_b * advance + char_d * high + char_f),
            char_b * 0.0 + char_d * high + char_f,
        )  # fmt: skip
        y1 = py_max(
            py_max(py_max(char_b * 0.0 + char_d * low + char_f,
                          char_b * advance + char_d * low + char_f),
                   char_b * advance + char_d * high + char_f),
            char_b * 0.0 + char_d * high + char_f,
        )  # fmt: skip
        results[glyph_index] = char_type(
            (x0, y0, x1, y1),
            text,
            cached_by_identity(font_names, decoder, font_name, glyph),
            y1 - y0,
        )
    return results

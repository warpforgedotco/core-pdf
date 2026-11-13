# SPDX-License-Identifier: AGPL-3.0-only

"""TJ arrays shown in one loop.

show_text_array does what the content interpreter's append_tj_array does for a capture
state, with append_text, append_decoded_text and the state's show_text and text_boundary
inlined, so the strings of an array kerned between every pair of characters do not each
cost a chain of Python calls. It computes what they compute in the same order, so each
string gets the same glyphs, run and sequence number. A string it cannot show exactly
goes to the state's own methods, and an array it cannot walk is declined untouched.
"""

import cython

from core_pdf_cythonized._glyphs import capture_horizontal_glyphs

MAX_EXACT_INT = cython.declare(object, 2**53)


@cython.cfunc
@cython.exceptval(-1, check=False)
def exact_operand(value: object, out: cython.p_double) -> cython.int:
    # A float, or an int a double holds exactly: what Python's arithmetic gives for them
    # once they meet a float, which every value computed here does.
    if type(value) is float:
        out[0] = cython.cast(cython.double, value)
        return 1
    if type(value) is int and -MAX_EXACT_INT <= value <= MAX_EXACT_INT:
        out[0] = cython.cast(cython.double, value)
        return 1
    return 0


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def garbage(code: cython.Py_UCS4) -> cython.bint:
    return code < 32 or (0xE000 <= code <= 0xF8FF)


@cython.cfunc
@cython.exceptval(-1, check=False)
def visible_text(text: str) -> cython.int:
    # is_visible_text for non-empty text: some character is neither a control nor private use.
    char: cython.Py_UCS4
    for char in text:
        if not garbage(char):
            return 1
    return 0


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def min4(a: cython.double, b: cython.double, c: cython.double, d: cython.double) -> cython.double:
    # min(a, b, c, d): the first of the smallest, as Python's min keeps it.
    result: cython.double = a
    if b < result:
        result = b
    if c < result:
        result = c
    if d < result:
        result = d
    return result


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def max4(a: cython.double, b: cython.double, c: cython.double, d: cython.double) -> cython.double:
    result: cython.double = a
    if b > result:
        result = b
    if c > result:
        result = c
    if d > result:
        result = d
    return result


@cython.cclass
class ArrayWalk:
    """What one TJ array's strings share: the state, its font and text state, the
    matrices, and the text layout once a string has fetched it."""

    state: object
    decoder: object
    moved_to: object
    text_run: object
    normalize: object
    glyph_layout: object
    observation_layout: object
    confidence: object
    suspicious: object
    labels: object
    widths: dict
    default_width: object
    width_source: object
    glyph_bbox: object
    font_name: object
    current_font: object
    is_vertical: object
    char_space: object
    word_space: object
    horizontal_scale: object
    font_size_value: object
    font_size: cython.double
    horizontal_scale_value: cython.double
    char_space_value: cython.double
    word_space_value: cython.double
    default_width_value: cython.double
    render_details: cython.bint
    keep_glyphs: cython.bint
    out_glyphs: object
    ta: cython.double
    tb: cython.double
    tc: cython.double
    td: cython.double
    linear_identity: cython.bint
    ctm_identity: cython.bint
    ca_value: object
    cb_value: object
    cc_value: object
    cd_value: object
    ce_value: object
    cf_value: object
    A: object
    B: object
    C: object
    D: object
    Ad: cython.double
    Bd: cython.double
    Cd: cython.double
    Dd: cython.double
    matrix: object
    capture_source: object
    stream_order: object
    xobject_depth: object
    clip_bbox: object
    page_clip: object
    pending_line_break: object

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def advance(self, glyphs: tuple, data: bytes, out: cython.p_double) -> cython.int:
        # The font's text_advance_vector for horizontal text, from its widths.
        total: cython.double = 0.0
        width = cython.declare(cython.double)
        spacing: cython.double
        glyph: object
        for glyph in glyphs:
            if not exact_operand(
                self.widths.get(glyph.width_code, self.default_width), cython.address(width)
            ):
                # A width that is neither float nor int: the font computes it.
                total = self.decoder.text_advance_vector(
                    data,
                    font_size=self.font_size_value,
                    char_space=self.char_space,
                    word_space=self.word_space,
                    horizontal_scale=self.horizontal_scale,
                    glyphs=glyphs,
                )[0]
                out[0] = total
                return 1
            spacing = self.char_space_value + (
                self.word_space_value if glyph.code_bytes == b" " else 0.0
            )
            total += (
                (width * self.font_size / 1000.0 + spacing) * self.horizontal_scale_value / 100.0
            )
        out[0] = total
        return 1

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def show(
        self, text: str, glyphs: tuple, te_value: object, tf_value: object, adv_x: cython.double
    ) -> cython.int:
        """show_text for one string at the text matrix's origin (te, tf). Returns 0,
        having changed nothing the state's show_text would not redo the same way,
        when the state must show it."""
        state = self.state
        E: object
        F: object
        if self.ctm_identity:
            E = te_value
            F = tf_value
        else:
            E = te_value * self.ca_value + tf_value * self.cc_value + self.ce_value
            F = te_value * self.cb_value + tf_value * self.cd_value + self.cf_value
        Ed = cython.declare(cython.double)
        Fd = cython.declare(cython.double)
        if not exact_operand(E, cython.address(Ed)) or not exact_operand(F, cython.address(Fd)):
            return 0
        layout = state.text_layout
        if (
            layout is None
            or layout.given_paint is not None
            or layout.decoder is not self.decoder
            or layout.identity_text_matrix is not False
        ):
            layout = state.current_text_layout(
                self.decoder, None, self.A, self.B, self.C, self.D, False
            )
        if layout.pushes_clip_scope or layout.actual_text_span is not None:
            return 0
        ascent = cython.declare(cython.double)
        descent = cython.declare(cython.double)
        rise = cython.declare(cython.double)
        scale_factor = cython.declare(cython.double)
        space_width = cython.declare(cython.double)
        if not (
            exact_operand(layout.ascent, cython.address(ascent))
            and exact_operand(layout.descent, cython.address(descent))
            and exact_operand(layout.rise, cython.address(rise))
            and exact_operand(layout.scale_factor, cython.address(scale_factor))
            and exact_operand(layout.space_width, cython.address(space_width))
        ):
            return 0
        # bool(text) and layout.paints_text and is_visible_text(text), text being non-empty.
        visible = layout.paints_text
        if visible:
            visible = visible_text(text) == 1
        seqno = state.sequence
        style = layout.style
        if style is None:
            style = state.new_layout_style(layout, self.A, self.B, self.C, self.D)
        clusters: list = []
        captured = capture_horizontal_glyphs(
            text,
            glyphs,
            self.glyph_layout,
            self.observation_layout,
            layout.width_source,
            self.glyph_bbox,
            layout.font_size,
            self.char_space,
            self.word_space,
            self.horizontal_scale,
            self.render_details,
            True,
            (E, F, self.A, self.B, self.C, self.D),
            layout.ascent,
            layout.descent,
            layout.rise,
            layout.font_scale,
            layout.advance_scale,
            self.clip_bbox,
            self.page_clip,
            visible,
            style,
            seqno,
            self.font_name,
            state.glyph_cluster_count,
            self.confidence,
            self.suspicious,
            self.labels,
            self.out_glyphs if self.keep_glyphs else [],
            clusters,
        )
        if captured is None:
            return 0
        kept_count, kept_advance, kept_ink, kept_confidence = captured
        state.glyph_cluster_count += kept_count

        A: cython.double = self.Ad
        B: cython.double = self.Bd
        C: cython.double = self.Cd
        D: cython.double = self.Dd
        ar: cython.double = ascent + rise
        dr: cython.double = descent + rise
        c0_x: cython.double = dr * C + Ed
        c0_y: cython.double = dr * D + Fd
        c1_x: cython.double = ar * C + Ed
        c1_y: cython.double = ar * D + Fd
        adv_A: cython.double = adv_x * A
        adv_B: cython.double = adv_x * B
        c2_x: cython.double = adv_A + c0_x
        c2_y: cython.double = adv_B + c0_y
        c3_x: cython.double = adv_A + c1_x
        c3_y: cython.double = adv_B + c1_y
        x0: cython.double = min4(c0_x, c1_x, c2_x, c3_x)
        y0: cython.double = min4(c0_y, c1_y, c2_y, c3_y)
        x1: cython.double = max4(c0_x, c1_x, c2_x, c3_x)
        y1: cython.double = max4(c0_y, c1_y, c2_y, c3_y)
        adv_y: cython.double = 0.0
        baseline = (E, F, Ed + adv_x * A + adv_y * C, Fd + adv_x * B + adv_y * D)
        provenance = (("source", self.capture_source), ("seqno", seqno), *layout.run_provenance)
        advance_bbox = (x0, y0, x1, y1)
        new_run = self.text_run(
            text if text.isascii() else self.normalize(text),
            x0,
            y0,
            x1,
            y1,
            te_value,
            tf_value,
            layout.effective_font_size,
            space_width * scale_factor,
            seqno,
            self.stream_order,
            self.xobject_depth,
            self.current_font,
            self.is_vertical,
            layout.rotation,
            visible,
            True,
            self.pending_line_break,
            seqno,
            layout.fill_color,
            advance_bbox,
            advance_bbox,
            baseline,
            provenance,
            None,
        )
        new_run.glyph_clusters = tuple(clusters)
        if kept_advance is not None and kept_ink is not None:
            new_run.advance_bbox = kept_advance
            new_run.ink_bbox = kept_ink
            new_run.confidence = kept_confidence
        state.update_pending_run(new_run)
        state.sequence = seqno + 1
        return 1

    @cython.cfunc
    def piece(self, data: bytes, te_value: object, tf_value: object) -> tuple:
        """append_text for one string at (te_value, tf_value), then the text matrix's
        move past it and the "shown" boundary. Returns the text matrix's new origin."""
        te = cython.declare(cython.double)
        tf = cython.declare(cython.double)
        if (
            not exact_operand(te_value, cython.address(te))
            or not exact_operand(tf_value, cython.address(tf))
            or (self.linear_identity and te == 0.0 and tf == 0.0)
        ):
            # At the identity text matrix Matrix.multiply returns the ctm itself, and the
            # layout is made for it: the state's own methods show the string.
            self.fall_back(te_value, tf_value)
            self.state.append_text(data, decoder=self.decoder)
            self.resume()
            return (self.matrix[4], self.matrix[5])
        glyphs = self.decoder.decode_glyphs(data)
        if type(glyphs) is not tuple:
            self.fall_back(te_value, tf_value)
            self.state.append_text(data, decoder=self.decoder)
            self.resume()
            return (self.matrix[4], self.matrix[5])
        glyph_tuple: tuple = glyphs
        text: str
        if len(glyph_tuple) == 1:
            text = glyph_tuple[0].unicode
        else:
            text = "".join([glyph.unicode for glyph in glyph_tuple])
        adv_x = cython.declare(cython.double)
        self.advance(glyph_tuple, data, cython.address(adv_x))
        if text and not self.show(text, glyph_tuple, te_value, tf_value, adv_x):
            self.fall_back(te_value, tf_value)
            self.state.show_text(self.state, text, data, glyph_tuple, self.decoder, adv_x, 0.0)
            self.resume()
        adv_y: cython.double = 0.0
        self.pending_line_break = False
        return (te + adv_x * self.ta + adv_y * self.tc, tf + adv_x * self.tb + adv_y * self.td)

    @cython.cfunc
    def fall_back(self, te_value: object, tf_value: object) -> cython.void:
        # Hand the state the text matrix and line-break flag a string's own methods read.
        self.state.text_matrix = self.moved_to(self.matrix, te_value, tf_value)
        self.state.pending_line_break = self.pending_line_break

    @cython.cfunc
    def resume(self) -> cython.void:
        # Take back the state's text matrix and line-break flag after its methods ran.
        self.matrix = self.state.text_matrix
        self.pending_line_break = self.state.pending_line_break


def show_text_array(state, array, decoder, context) -> bool:
    """append_tj_array for a capture state whose text methods are the stock ones.

    context is (string type, moved_to, TextRun, normalize_extracted_text, the decoded
    glyph and observation slot layouts, the confidence cache and function, the
    suspicious multi-glyph test, the bitmap labels). Returns False, having done
    nothing, for an array the interpreter's loop must walk: not a non-empty list or
    tuple of strings and numbers, vertical or Type 3 text, values other than floats
    and exact ints.
    """
    (
        string_type,
        moved_to,
        text_run,
        normalize,
        glyph_layout,
        observation_layout,
        confidence,
        suspicious,
        labels,
    ) = context
    if type(array) is not list and type(array) is not tuple:
        return False
    if not array:
        return False
    number = cython.declare(cython.double)
    item: object
    kind: object
    for item in array:
        kind = type(item)
        if kind is string_type or kind is bytes:
            continue
        if (kind is int or kind is float) and exact_operand(item, cython.address(number)):
            continue
        return False
    if decoder.is_type3 or decoder.is_vertical:
        return False
    metrics = decoder.metrics
    if metrics.is_vertical or type(metrics.widths) is not dict:
        return False

    walk = ArrayWalk()
    walk.state = state
    walk.decoder = decoder
    walk.moved_to = moved_to
    walk.text_run = text_run
    walk.normalize = normalize
    walk.glyph_layout = glyph_layout
    walk.observation_layout = observation_layout
    walk.confidence = confidence
    walk.suspicious = suspicious
    walk.labels = labels
    walk.widths = metrics.widths
    walk.default_width = metrics.default_width
    graphics = state.graphics
    walk.font_size_value = graphics.font_size
    walk.horizontal_scale = graphics.horizontal_scale
    walk.char_space = graphics.char_space
    walk.word_space = graphics.word_space
    if not (
        exact_operand(walk.default_width, cython.address(walk.default_width_value))
        and exact_operand(walk.font_size_value, cython.address(walk.font_size))
        and exact_operand(walk.horizontal_scale, cython.address(walk.horizontal_scale_value))
        and exact_operand(walk.char_space, cython.address(walk.char_space_value))
        and exact_operand(walk.word_space, cython.address(walk.word_space_value))
    ):
        return False
    matrix = state.text_matrix
    ctm = graphics.ctm
    if not (isinstance(matrix, tuple) and len(matrix) == 6):
        return False
    if not (isinstance(ctm, tuple) and len(ctm) == 6):
        return False
    te = cython.declare(cython.double)
    tf = cython.declare(cython.double)
    ca = cython.declare(cython.double)
    cb = cython.declare(cython.double)
    cc = cython.declare(cython.double)
    cd = cython.declare(cython.double)
    ce = cython.declare(cython.double)
    cf = cython.declare(cython.double)
    m = cython.cast(tuple, matrix)
    c = cython.cast(tuple, ctm)
    if not (
        exact_operand(m[0], cython.address(walk.ta))
        and exact_operand(m[1], cython.address(walk.tb))
        and exact_operand(m[2], cython.address(walk.tc))
        and exact_operand(m[3], cython.address(walk.td))
        and exact_operand(m[4], cython.address(te))
        and exact_operand(m[5], cython.address(tf))
        and exact_operand(c[0], cython.address(ca))
        and exact_operand(c[1], cython.address(cb))
        and exact_operand(c[2], cython.address(cc))
        and exact_operand(c[3], cython.address(cd))
        and exact_operand(c[4], cython.address(ce))
        and exact_operand(c[5], cython.address(cf))
    ):
        return False

    options = state.options
    walk.render_details = bool(options.render_details)
    walk.keep_glyphs = bool(options.glyphs)
    walk.out_glyphs = state.glyphs
    walk.glyph_bbox = decoder.glyph_bbox if options.ink_bounds else None
    walk.current_font = graphics.current_font
    walk.font_name = decoder.font_name or walk.current_font
    walk.is_vertical = decoder.is_vertical
    walk.capture_source = state.capture_source
    walk.stream_order = state.stream_order
    walk.xobject_depth = state.xobject_depth
    walk.clip_bbox = state.clip_bbox
    walk.page_clip = state.page_clip
    walk.matrix = matrix
    walk.pending_line_break = state.pending_line_break
    walk.linear_identity = walk.ta == 1.0 and walk.tb == 0.0 and walk.tc == 0.0 and walk.td == 1.0
    # Matrix.multiply returns the text matrix itself when the ctm is the identity, and
    # multiply_affine's products otherwise; their linear part is the same for every string.
    walk.ctm_identity = (
        ca == 1.0 and cb == 0.0 and cc == 0.0 and cd == 1.0 and ce == 0.0 and cf == 0.0
    )
    walk.ca_value, walk.cb_value, walk.cc_value = c[0], c[1], c[2]
    walk.cd_value, walk.ce_value, walk.cf_value = c[3], c[4], c[5]
    if walk.ctm_identity:
        walk.A, walk.B, walk.C, walk.D = m[0], m[1], m[2], m[3]
    else:
        walk.A = m[0] * c[0] + m[1] * c[2]
        walk.B = m[0] * c[1] + m[1] * c[3]
        walk.C = m[2] * c[0] + m[3] * c[2]
        walk.D = m[2] * c[1] + m[3] * c[3]
    if not (
        exact_operand(walk.A, cython.address(walk.Ad))
        and exact_operand(walk.B, cython.address(walk.Bd))
        and exact_operand(walk.C, cython.address(walk.Cd))
        and exact_operand(walk.D, cython.address(walk.Dd))
    ):
        return False

    te_value: object = m[4]
    tf_value: object = m[5]
    pending = bytearray()
    try:
        for item in array:
            kind = type(item)
            if kind is string_type:
                pending.extend(item.data)
                continue
            if kind is bytes:
                pending.extend(item)
                continue
            if pending:
                te_value, tf_value = walk.piece(bytes(pending), te_value, tf_value)
                pending.clear()
                exact_operand(te_value, cython.address(te))
                exact_operand(tf_value, cython.address(tf))
            # text_adjustment_vector, then te += advance_x * ta + advance_y * tc. An int
            # is negated as an int, so 0 gives 0.0 rather than -0.0.
            if kind is int:
                number = cython.cast(cython.double, -item)
            else:
                number = -cython.cast(cython.double, item)
            advance_x: cython.double = (
                number * walk.font_size / 1000.0 * walk.horizontal_scale_value / 100.0
            )
            advance_y: cython.double = 0.0
            te += advance_x * walk.ta + advance_y * walk.tc
            tf += advance_x * walk.tb + advance_y * walk.td
            te_value = te
            tf_value = tf
        if pending:
            te_value, tf_value = walk.piece(bytes(pending), te_value, tf_value)
    except BaseException:
        walk.fall_back(te_value, tf_value)
        raise
    state.text_matrix = moved_to(walk.matrix, te_value, tf_value)
    state.pending_line_break = walk.pending_line_break
    return True

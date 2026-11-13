# SPDX-License-Identifier: AGPL-3.0-only

"""Text showing: the content interpreter's Tj and TJ, a capture state's show_text, and
the accumulator that merges the runs they show.

These are the only implementations. The interpreter's append_tj_array, append_text and
append_decoded_text call show_text_array, show_string and show_decoded, and a capture
state's show_text calls show_text, so the strings of an array kerned between every pair
of characters do not each cost a chain of Python calls. A capture state hands its
collaborators over as its text_capture_context; one whose show_text, text_boundary,
current_text_layout and update_pending_run are the stock ones, and which is its own sink,
has its strings shown inline, and any other interpreter has its sink's methods called.

Numbers are read as doubles, which is what Python's arithmetic makes of them once they
meet the floats every position is computed with; where Python would keep an int, such as
the text matrix times the ctm, the value is computed on the objects.

RunAccumulator is core's only run accumulator: a capture state appends each shown run to
it, and it merges a run into the pending one when it continues it. Shown inline, a string
is merged with its values before a TextRun is made, which only a string starting a run
needs.
"""

import cython

from core_pdf_cythonized._glyphs import capture_horizontal_glyphs


@cython.cfunc
@cython.inline
@cython.exceptval(-1.0, check=True)
def as_double(value: object) -> cython.double:
    # What Python's float arithmetic reads a number as: an int is rounded to a double first.
    if type(value) is float:
        return cython.cast(cython.double, value)
    return cython.cast(cython.double, value)


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def garbage(code: cython.Py_UCS4) -> cython.bint:
    return code < 32 or (0xE000 <= code <= 0xF8FF)


@cython.cfunc
@cython.exceptval(-1, check=False)
def visible_text(text: str) -> cython.int:
    # Some character is neither a control nor private use.
    char: cython.Py_UCS4
    for char in text:
        if not garbage(char):
            return 1
    return 0


def is_visible_text(text: str) -> bool:
    """Whether text holds a character other than controls and private use."""
    return visible_text(text) == 1


@cython.cfunc
@cython.exceptval(-1.0, check=True)
def horizontal_advance(
    widths: object,
    default_width: object,
    glyphs: object,
    font_size: cython.double,
    char_space: cython.double,
    word_space: cython.double,
    horizontal_scale: cython.double,
) -> cython.double:
    total: cython.double = 0.0
    glyph: object
    for glyph in glyphs:
        spacing: cython.double = char_space + (word_space if glyph.code_bytes == b" " else 0.0)
        total += (
            (as_double(widths.get(glyph.width_code, default_width)) * font_size / 1000.0 + spacing)
            * horizontal_scale
            / 100.0
        )
    return total


def text_advance(
    widths: object,
    default_width: object,
    glyphs: object,
    font_size: object,
    char_space: object,
    word_space: object,
    horizontal_scale: object,
) -> float:
    """A horizontal font's advance across glyphs: each width (widths.get(width_code,
    default_width)) times the font size over 1000, plus the character spacing and, after a
    single-byte space, the word spacing, scaled horizontally."""
    return horizontal_advance(
        widths,
        default_width,
        glyphs,
        as_double(font_size),
        as_double(char_space),
        as_double(word_space),
        as_double(horizontal_scale),
    )


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


# gap_separator's punctuation: no space before a closing mark, nor after an opening one.
NO_SPACE_BEFORE = cython.declare(frozenset, frozenset(".,;:!?)]}%"))
NO_SPACE_AFTER = cython.declare(frozenset, frozenset("([{"))

# The edge a merge widens the pending run on.
WIDEN_X0 = cython.declare(cython.int, 0)
WIDEN_X1 = cython.declare(cython.int, 1)
WIDEN_Y1 = cython.declare(cython.int, 2)


@cython.cfunc
@cython.exceptval(-1, check=False)
def can_merge_cross_font_word(left: str, right: str) -> cython.int:
    # Whether a word continues across a font change: a word character on both sides.
    if not left or not right:
        return 0
    last: cython.Py_UCS4 = left[len(left) - 1]
    first: cython.Py_UCS4 = right[0]
    return (last.isalnum() or last == "_") and (first.isalnum() or first == "_")


@cython.cfunc
@cython.inline
def last_char(text: str) -> str:
    # text[-1:], spelled without a negative index: the package compiles with wraparound off.
    return text[len(text) - 1 :] if text else ""


@cython.cfunc
@cython.inline
def first_max(a: object, b: object) -> object:
    # max(a, b): b only when it is greater, as Python's max keeps the first of equals.
    return b if b > a else a


@cython.cfunc
@cython.inline
def first_min(a: object, b: object) -> object:
    return b if b < a else a


@cython.cclass
class RunAccumulator:
    """The runs a capture state shows, each merged into the pending run when it continues
    it on the same line in the same style, and every finished run put in output.

    word_gap_threshold(space_width, font_size) is core's word gap, past which a merge
    puts a space between the two texts.
    """

    output = cython.declare(list, visibility="readonly")
    word_gap_threshold: object
    # The pending run, its texts and glyph clusters merged before it (nearest last) and
    # after it (its own first), and the first and last characters of its joined text.
    run: object
    before: list
    after: list
    clusters_before: list
    clusters_after: list
    head: str
    tail: str

    def __init__(self, output: list, word_gap_threshold: object) -> None:
        self.output = output
        self.word_gap_threshold = word_gap_threshold
        self.run = None

    def append(self, run: object) -> None:
        """Merge run into the pending run, or flush that and make run the pending one."""
        if self.run is not None and self.try_merge(
            run.text,
            run.x0,
            run.y0,
            run.x1,
            run.y1,
            run.font_size,
            run.rotation_angle,
            run.visible,
            run.line_break_before,
            run.font_name,
            run.fill_color,
            run.ink_bbox,
            run.glyph_clusters,
        ):
            return
        self.start(run)

    def flush(self) -> None:
        """Put the pending run, its merged text and glyph clusters joined, in output."""
        run = self.run
        if run is None:
            return
        if len(self.before) + len(self.after) > 1:
            self.before.reverse()
            run.set_text("".join(self.before + self.after))
            clusters: list = []
            part: object
            for part in reversed(self.clusters_before):
                clusters.extend(part)
            for part in self.clusters_after:
                clusters.extend(part)
            run.glyph_clusters = tuple(clusters)
        self.output.append(run)
        self.run = None

    @cython.cfunc
    def start(self, run: object) -> cython.void:
        # Flush the pending run and make run the pending one.
        self.flush()
        text: str = run.text
        self.run = run
        self.before = []
        self.after = [text]
        self.clusters_before = []
        self.clusters_after = [run.glyph_clusters]
        self.head = text[:1]
        self.tail = last_char(text)

    @cython.cfunc
    def gap_separator(self, left: str, right: str, gap: object) -> str:
        # The space a merge puts between left and right across gap, if any.
        run = self.run
        if gap <= self.word_gap_threshold(run.space_width, run.font_size):
            return ""
        if not left or not right:
            return ""
        last = last_char(left)
        first = right[:1]
        if last.isspace() or first.isspace():
            return ""
        if first in NO_SPACE_BEFORE or last in NO_SPACE_AFTER:
            return ""
        return " "

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def merge(
        self,
        text: str,
        clusters: object,
        ink_bbox: object,
        x0: object,
        x1: object,
        y1: object,
        gap: object,
        threshold: object,
        widen: cython.int,
        reverse: cython.bint,
    ) -> cython.int:
        # Merge a run with these values into the pending one across gap, after it or,
        # reversed, before it, when the gap is from -2 up to the threshold.
        if not (-2.0 <= gap and gap < threshold):
            return 0
        added: str
        if reverse:
            added = text + self.gap_separator(text, self.head, gap)
            self.before.append(added)
            self.clusters_before.append(clusters)
            if added:
                self.head = added[:1]
            if not self.tail:
                self.tail = last_char(added)
        else:
            added = self.gap_separator(self.tail, text, gap) + text
            self.after.append(added)
            self.clusters_after.append(clusters)
            if not self.head:
                self.head = added[:1]
            if added:
                self.tail = last_char(added)
        run = self.run
        run.union_ink_bbox(ink_bbox)
        if widen == WIDEN_X1:
            run.x1 = first_max(x1, run.x1)
        elif widen == WIDEN_X0:
            run.x0 = first_min(x0, run.x0)
        else:
            run.y1 = first_max(y1, run.y1)
        return 1

    @cython.cfunc
    @cython.exceptval(-1, check=False)
    def try_merge(
        self,
        text: str,
        x0: object,
        y0: object,
        x1: object,
        y1: object,
        font_size: object,
        rotation_angle: object,
        visible: object,
        line_break_before: object,
        font_name: object,
        fill_color: object,
        ink_bbox: object,
        clusters: object,
    ) -> cython.int:
        """Merge a run with these values into the pending run when it continues it: the
        same rotation, visibility, size, font (or a word going on across fonts) and fill,
        no line break before it, on the pending run's line, within the merge gap."""
        pending = self.run
        pending_size = pending.font_size
        space_width = pending.space_width
        rotation = pending.rotation_angle
        threshold = first_max(space_width * 0.45, 2.0)
        if not (
            rotation == rotation_angle
            and pending.visible == visible
            and not line_break_before
            and pending_size == font_size
            and (
                pending.font_name == font_name
                or can_merge_cross_font_word(self.tail, text)
                or can_merge_cross_font_word(text, self.head)
            )
            and pending.fill_color == fill_color
        ):
            return 0
        if rotation == 90:
            if abs(y0 - pending.y1) > first_max(
                first_max(space_width * 0.5, pending_size * 0.8), 2.0
            ):
                return 0
        elif rotation == 0 and abs(pending.y0 - y0) > pending_size * 0.5:
            return 0
        merged: cython.int
        if rotation == 0:
            merged = self.merge(
                text, clusters, ink_bbox, x0, x1, y1, x0 - pending.x1, threshold, WIDEN_X1, False
            ) or self.merge(
                text, clusters, ink_bbox, x0, x1, y1, pending.x0 - x1, threshold, WIDEN_X0, True
            )
        elif rotation == 90:
            merged = self.merge(
                text, clusters, ink_bbox, x0, x1, y1, y0 - pending.y1, threshold, WIDEN_Y1, False
            )
        else:
            merged = self.merge(
                text, clusters, ink_bbox, x0, x1, y1, pending.x0 - x1, threshold, WIDEN_X0, False
            ) or self.merge(
                text, clusters, ink_bbox, x0, x1, y1, x0 - pending.x1, threshold, WIDEN_X1, True
            )
        if merged:
            pending.advance_bbox = (pending.x0, pending.y0, pending.x1, pending.y1)
        return merged


@cython.cclass
class TextShow:
    """What showing a string shares with the strings around it: the state, its font and
    text state, the text matrix and ctm, and, for a capture state, what it captures with.

    piece, decoded and show are the interpreter's append_text and append_decoded_text and
    a capture state's show_text, each computing what the Python did, in the same order.
    """

    state: object
    sink: object
    decoder: object
    string_type: object
    moved_to: object
    # The state's text_capture_context, and whether it shows strings inline.
    capture: cython.bint
    inline: cython.bint
    stock_layout: cython.bint
    stock_pending: cython.bint
    stock_advance: object
    text_run: object
    normalize: object
    glyph_layout: object
    observation_layout: object
    confidence: object
    suspicious: object
    labels: object
    capture_glyphs: object
    # The font.
    is_type3: cython.bint
    vertical: cython.bint
    vertical_value: object
    inline_advance: cython.bint
    widths: object
    default_width: object
    font_name: object
    # The text state.
    font_size_value: object
    horizontal_scale_value: object
    char_space_value: object
    word_space_value: object
    font_size: cython.double
    horizontal_scale: cython.double
    char_space: cython.double
    word_space: cython.double
    current_font: object
    # The text matrix (its linear part, which showing never changes) and the ctm.
    matrix: object
    ta: cython.double
    tb: cython.double
    tc: cython.double
    td: cython.double
    linear_identity: cython.bint
    ctm_identity: cython.bint
    ca: object
    cb: object
    cc: object
    cd: object
    ce: object
    cf: object
    # The linear part of the text matrix times the ctm, for a text matrix not the identity.
    A: object
    B: object
    C: object
    D: object
    # What the capture state captures with.
    options: object
    want_glyphs: cython.bint
    want_runs: cython.bint
    render_details: cython.bint
    out_glyphs: object
    glyph_bbox: object
    capture_source: object
    stream_order: object
    xobject_depth: object
    clip_bbox: object
    page_clip: object
    accumulator: RunAccumulator
    pending_line_break: object
    # The text matrix origin the Python last assigned to state.text_matrix.
    committed: cython.bint
    commit_e: object
    commit_f: object

    @cython.cfunc
    def setup(self, state: object, decoder: object, spec_context: object) -> cython.void:
        self.state = state
        self.sink = state.sink
        self.decoder = decoder
        if spec_context is not None:
            self.string_type, self.moved_to = spec_context
        context = getattr(state, "text_capture_context", None)
        self.capture = context is not None
        if self.capture:
            (
                stock,
                stock_layout,
                stock_pending,
                self.stock_advance,
                self.text_run,
                self.normalize,
                self.glyph_layout,
                self.observation_layout,
                self.confidence,
                self.suspicious,
                self.labels,
                self.capture_glyphs,
            ) = context
            kind = type(state)
            self.inline = self.sink is state and bool(stock(kind))
            self.stock_layout = getattr(kind, "current_text_layout", None) is stock_layout
            self.stock_pending = getattr(kind, "update_pending_run", None) is stock_pending
        self.is_type3 = bool(decoder.is_type3)
        self.vertical_value = decoder.is_vertical
        self.vertical = bool(self.vertical_value)
        self.inline_advance = False
        if self.capture and type(decoder).text_advance_vector is self.stock_advance:
            metrics = decoder.metrics
            if not metrics.is_vertical:
                self.inline_advance = True
                self.widths = metrics.widths
                self.default_width = metrics.default_width
        self.load_text_state()
        self.load_matrices()
        if self.capture:
            self.load_capture()
            self.pending_line_break = state.pending_line_break

    @cython.cfunc
    def load_text_state(self) -> cython.void:
        graphics = self.state.graphics
        self.font_size_value = graphics.font_size
        self.horizontal_scale_value = graphics.horizontal_scale
        self.char_space_value = graphics.char_space
        self.word_space_value = graphics.word_space
        self.font_size = as_double(self.font_size_value)
        self.horizontal_scale = as_double(self.horizontal_scale_value)
        self.char_space = as_double(self.char_space_value)
        self.word_space = as_double(self.word_space_value)
        self.current_font = graphics.current_font
        ctm = graphics.ctm
        self.ca, self.cb, self.cc, self.cd, self.ce, self.cf = ctm
        # Matrix.multiply returns the text matrix itself when the ctm is the identity.
        self.ctm_identity = (
            self.ca == 1.0
            and self.cb == 0.0
            and self.cc == 0.0
            and self.cd == 1.0
            and self.ce == 0.0
            and self.cf == 0.0
        )

    @cython.cfunc
    def load_matrices(self) -> cython.void:
        matrix = self.state.text_matrix
        self.matrix = matrix
        a, b, c, d = matrix[0], matrix[1], matrix[2], matrix[3]
        self.ta = as_double(a)
        self.tb = as_double(b)
        self.tc = as_double(c)
        self.td = as_double(d)
        self.linear_identity = a == 1.0 and b == 0.0 and c == 0.0 and d == 1.0
        if self.ctm_identity:
            self.A, self.B, self.C, self.D = a, b, c, d
        else:
            # multiply_affine's linear part, on the objects as Python multiplies them.
            self.A = a * self.ca + b * self.cc
            self.B = a * self.cb + b * self.cd
            self.C = c * self.ca + d * self.cc
            self.D = c * self.cb + d * self.cd

    @cython.cfunc
    def load_capture(self) -> cython.void:
        state = self.state
        options = state.options
        self.options = options
        self.want_glyphs = bool(options.glyphs)
        self.want_runs = bool(options.text_runs)
        self.render_details = bool(options.render_details)
        self.out_glyphs = state.glyphs
        self.glyph_bbox = self.decoder.glyph_bbox if options.ink_bounds else None
        self.font_name = self.decoder.font_name or self.current_font
        self.capture_source = state.capture_source
        self.stream_order = state.stream_order
        self.xobject_depth = state.xobject_depth
        self.clip_bbox = state.clip_bbox
        self.page_clip = state.page_clip
        accumulator = state.run_accumulator
        self.accumulator = (
            accumulator if type(accumulator) is RunAccumulator and self.stock_pending else None
        )

    @cython.cfunc
    def commit(self, e: object, f: object) -> cython.void:
        # The text matrix origin the Python assigns to state.text_matrix here; a state shown
        # through its sink's methods gets it at once, for them to read.
        self.committed = True
        self.commit_e = e
        self.commit_f = f
        if not self.inline:
            self.state.text_matrix = self.moved_to(self.matrix, e, f)

    @cython.cfunc
    def hand_over(self) -> cython.void:
        # Give the state the text matrix and line-break flag its own methods read.
        if self.committed:
            self.state.text_matrix = self.moved_to(self.matrix, self.commit_e, self.commit_f)
        if self.inline:
            self.state.pending_line_break = self.pending_line_break

    @cython.cfunc
    def piece(self, data: bytes, te_value: object, tf_value: object) -> tuple:
        """append_text at (te_value, tf_value). Returns the text matrix origin after it."""
        glyphs = self.decoder.decode_glyphs(data)
        text: str
        if len(glyphs) == 1:
            text = glyphs[0].unicode
        else:
            text = "".join([glyph.unicode for glyph in glyphs])
        return self.decoded(text, data, glyphs, te_value, tf_value)

    @cython.cfunc
    def decoded(
        self, text: str, data: object, glyphs: object, te_value: object, tf_value: object
    ) -> tuple:
        """append_decoded_text at (te_value, tf_value): a Type 3 font's glyph procedures run,
        the string shown, the text matrix moved past it and the "shown" boundary sent.
        Returns the text matrix origin after it."""
        state = self.state
        if self.is_type3 and data:
            self.hand_over()
            text_matrix = self.moved_to(self.matrix, te_value, tf_value)
            line_matrix = state.line_matrix
            state.text_matrix = text_matrix
            state.render_type3_glyphs(data, self.decoder)
            state.text_matrix = text_matrix
            state.line_matrix = line_matrix
            if self.inline:
                self.pending_line_break = state.pending_line_break
            self.load_text_state()
            if self.capture:
                self.load_capture()
        if not text and not data:
            return (te_value, tf_value)
        adv_x: cython.double
        adv_y: cython.double
        if self.inline_advance:
            adv_x = horizontal_advance(
                self.widths,
                self.default_width,
                glyphs,
                self.font_size,
                self.char_space,
                self.word_space,
                self.horizontal_scale,
            )
            adv_y = 0.0
        else:
            advance = self.decoder.text_advance_vector(
                data,
                font_size=self.font_size_value,
                char_space=self.char_space_value,
                word_space=self.word_space_value,
                horizontal_scale=self.horizontal_scale_value,
                glyphs=glyphs,
            )
            adv_x = as_double(advance[0])
            adv_y = as_double(advance[1])
        te: cython.double = as_double(te_value)
        tf: cython.double = as_double(tf_value)
        if text:
            if self.inline:
                self.show(text, glyphs, te_value, tf_value, adv_x, adv_y, None)
            else:
                self.sink.show_text(state, text, data, glyphs, self.decoder, adv_x, adv_y)
        moved_e = te + adv_x * self.ta + adv_y * self.tc
        moved_f = tf + adv_x * self.tb + adv_y * self.td
        self.commit(moved_e, moved_f)
        if self.inline:
            self.pending_line_break = False
        else:
            self.sink.text_boundary(state, "shown")
        return (self.commit_e, self.commit_f)

    @cython.cfunc
    def show(
        self,
        text: str,
        glyphs: object,
        te_value: object,
        tf_value: object,
        adv_x: cython.double,
        adv_y: cython.double,
        glyph_paint: object,
    ) -> cython.void:
        """A capture state's show_text at the text matrix origin (te_value, tf_value)."""
        state = self.state
        decoder = self.decoder
        identity: cython.bint = (
            self.linear_identity and as_double(te_value) == 0.0 and as_double(tf_value) == 0.0
        )
        # Matrix.multiply: the text matrix when the ctm is the identity, the ctm when the
        # text matrix is, multiply_affine's product otherwise.
        A: object
        B: object
        C: object
        D: object
        E: object
        F: object
        if self.ctm_identity:
            A, B, C, D, E, F = self.A, self.B, self.C, self.D, te_value, tf_value
        elif identity:
            A, B, C, D, E, F = self.ca, self.cb, self.cc, self.cd, self.ce, self.cf
        else:
            A, B, C, D = self.A, self.B, self.C, self.D
            E = te_value * self.ca + tf_value * self.cc + self.ce
            F = te_value * self.cb + tf_value * self.cd + self.cf
        identity_value = True if identity else False
        layout = state.text_layout if self.stock_layout else None
        if (
            layout is None
            or layout.given_paint is not glyph_paint
            or layout.decoder is not decoder
            or layout.identity_text_matrix is not identity_value
        ):
            layout = state.current_text_layout(decoder, glyph_paint, A, B, C, D, identity_value)
        visible = False
        if text:
            visible = layout.paints_text
            if visible:
                visible = visible_text(text) == 1
        if layout.pushes_clip_scope:
            state.emit_clip_scope_push()
        seqno = state.sequence
        span = layout.actual_text_span
        clusters: object = ()
        has_geometry: cython.bint = False
        kept_advance: object = None
        kept_ink: object = None
        kept_confidence: object = None
        if span is None:
            if not self.want_glyphs and not self.want_runs:
                state.sequence = seqno + 1
                return
            style = layout.style
            if style is None:
                style = state.new_layout_style(layout, A, B, C, D)
            out_glyphs = self.out_glyphs if self.want_glyphs else []
            basis = (E, F, A, B, C, D)
            kept_count: object = 0
            captured: object = None
            if glyphs and not self.vertical and type(glyphs) is tuple:
                clusters = []
                captured = capture_horizontal_glyphs(
                    text,
                    glyphs,
                    self.glyph_layout,
                    self.observation_layout,
                    layout.width_source,
                    self.glyph_bbox,
                    layout.font_size,
                    self.char_space_value,
                    self.word_space_value,
                    self.horizontal_scale_value,
                    self.render_details,
                    self.want_runs,
                    basis,
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
                    out_glyphs,
                    clusters,
                )
                if captured is not None:
                    kept_count, kept_advance, kept_ink, kept_confidence = captured
                    has_geometry = (
                        self.want_runs and kept_advance is not None and kept_ink is not None
                    )
            if captured is None and glyphs:
                # Vertical text, or a glyph whose text splits: capture_glyphs' own loop.
                result = self.capture_glyphs(
                    text,
                    glyphs,
                    decoder,
                    basis,
                    layout.font_size,
                    layout.font_scale,
                    layout.ascent,
                    layout.descent,
                    layout.advance_scale,
                    self.char_space_value,
                    self.word_space_value,
                    self.horizontal_scale_value,
                    layout.rise,
                    style,
                    self.clip_bbox,
                    self.page_clip,
                    visible,
                    self.current_font,
                    seqno,
                    state.glyph_cluster_count,
                    self.options,
                    layout.width_source,
                    out_glyphs,
                )
                kept_count = result.cluster_count
                clusters = result.clusters
                geometry = result.geometry
                if geometry.started:
                    has_geometry = True
                    kept_advance = geometry.advance
                    kept_ink = geometry.ink
                    kept_confidence = geometry.confidence
            state.glyph_cluster_count += kept_count
            if not self.want_runs:
                state.sequence = seqno + 1
                return

        Ad: cython.double = as_double(A)
        Bd: cython.double = as_double(B)
        Cd: cython.double = as_double(C)
        Dd: cython.double = as_double(D)
        Ed: cython.double = as_double(E)
        Fd: cython.double = as_double(F)
        ascent: cython.double = as_double(layout.ascent)
        descent: cython.double = as_double(layout.descent)
        rise: cython.double = as_double(layout.rise)
        c0_x: cython.double
        c0_y: cython.double
        c1_x: cython.double
        c1_y: cython.double
        c2_x: cython.double
        c2_y: cython.double
        c3_x: cython.double
        c3_y: cython.double
        if self.vertical:
            c0_x = descent * Ad + rise * Cd + Ed
            c0_y = descent * Bd + rise * Dd + Fd
            c1_x = ascent * Ad + rise * Cd + Ed
            c1_y = ascent * Bd + rise * Dd + Fd
            adv_C: cython.double = adv_y * Cd
            adv_D: cython.double = adv_y * Dd
            c2_x = adv_C + c0_x
            c2_y = adv_D + c0_y
            c3_x = adv_C + c1_x
            c3_y = adv_D + c1_y
        else:
            ar: cython.double = ascent + rise
            dr: cython.double = descent + rise
            c0_x = dr * Cd + Ed
            c0_y = dr * Dd + Fd
            c1_x = ar * Cd + Ed
            c1_y = ar * Dd + Fd
            adv_A: cython.double = adv_x * Ad
            adv_B: cython.double = adv_x * Bd
            c2_x = adv_A + c0_x
            c2_y = adv_B + c0_y
            c3_x = adv_A + c1_x
            c3_y = adv_B + c1_y
        advance_bbox = (
            min4(c0_x, c1_x, c2_x, c3_x),
            min4(c0_y, c1_y, c2_y, c3_y),
            max4(c0_x, c1_x, c2_x, c3_x),
            max4(c0_y, c1_y, c2_y, c3_y),
        )
        run_text: str = text if text.isascii() else self.normalize(text)
        cluster_tuple = tuple(clusters)
        accumulator: RunAccumulator = self.accumulator if span is None else None
        if accumulator is not None:
            # update_pending_run: a run clipped away is dropped, and one that continues the
            # pending run is merged into it, so it needs no TextRun of its own.
            if state.is_clipped_away(
                advance_bbox[0], advance_bbox[1], advance_bbox[2], advance_bbox[3]
            ):
                state.sequence = seqno + 1
                return
            if accumulator.run is not None and accumulator.try_merge(
                run_text,
                advance_bbox[0],
                advance_bbox[1],
                advance_bbox[2],
                advance_bbox[3],
                layout.effective_font_size,
                layout.rotation,
                visible,
                self.pending_line_break,
                self.current_font,
                layout.fill_color,
                kept_ink if has_geometry else advance_bbox,
                cluster_tuple,
            ):
                state.sequence = seqno + 1
                return
        baseline = (E, F, Ed + adv_x * Ad + adv_y * Cd, Fd + adv_x * Bd + adv_y * Dd)
        provenance = (("source", self.capture_source), ("seqno", seqno), *layout.run_provenance)
        new_run = self.text_run(
            run_text,
            advance_bbox[0],
            advance_bbox[1],
            advance_bbox[2],
            advance_bbox[3],
            te_value,
            tf_value,
            layout.effective_font_size,
            as_double(layout.space_width) * as_double(layout.scale_factor),
            seqno,
            self.stream_order,
            self.xobject_depth,
            self.current_font,
            self.vertical_value,
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
        if span is not None:
            new_run.confidence = 1.0
            span.add_run(
                new_run, font_decoder=decoder, effective_font_height=layout.effective_font_height
            )
        else:
            new_run.glyph_clusters = cluster_tuple
            if has_geometry:
                new_run.advance_bbox = kept_advance
                new_run.ink_bbox = kept_ink
                new_run.confidence = kept_confidence
            if accumulator is not None:
                accumulator.start(new_run)
            else:
                state.update_pending_run(new_run)
        state.sequence = seqno + 1


def show_text_array(state, array, decoder, spec_context) -> None:
    """The interpreter's append_tj_array for a non-empty list or tuple: each run of strings
    shown as append_text shows it, each number moving the text matrix back by its
    thousandths of the font size, and anything else read with state.tj_array_extra_bytes.

    spec_context is (the string type, moved_to).
    """
    show = TextShow()
    show.setup(state, decoder, spec_context)
    string_type = show.string_type
    te_value: object = show.matrix[4]
    tf_value: object = show.matrix[5]
    te: cython.double
    tf: cython.double
    pending = bytearray()
    item: object
    kind: object
    number: cython.double
    try:
        for item in array:
            kind = type(item)
            if kind is string_type:
                pending.extend(item.data)
            elif kind is bytes:
                pending.extend(item)
            elif kind is int or kind is float:
                if pending:
                    show.commit(te_value, tf_value)
                    te_value, tf_value = show.piece(bytes(pending), te_value, tf_value)
                    pending.clear()
                # text_adjustment_vector: an int is negated as an int, so 0 gives 0.0.
                if kind is int:
                    number = as_double(-item)
                else:
                    number = -as_double(item)
                displacement: cython.double = number * show.font_size / 1000.0
                advance_x: cython.double = 0.0
                advance_y: cython.double = 0.0
                if show.vertical:
                    advance_y = displacement
                else:
                    advance_x = displacement * show.horizontal_scale / 100.0
                te = as_double(te_value)
                tf = as_double(tf_value)
                te += advance_x * show.ta + advance_y * show.tc
                tf += advance_x * show.tb + advance_y * show.td
                te_value = te
                tf_value = tf
            else:
                pending.extend(state.tj_array_extra_bytes(item))
        if pending:
            show.commit(te_value, tf_value)
            te_value, tf_value = show.piece(bytes(pending), te_value, tf_value)
    except BaseException:
        show.hand_over()
        raise
    show.committed = True
    show.commit_e = te_value
    show.commit_f = tf_value
    show.hand_over()


def show_string(state, data: bytes, decoder, spec_context) -> None:
    """The interpreter's append_text: data's glyphs shown at the text matrix, which moves
    past them."""
    show = TextShow()
    show.setup(state, decoder, spec_context)
    try:
        show.piece(data, show.matrix[4], show.matrix[5])
    except BaseException:
        show.hand_over()
        raise
    show.hand_over()


def show_decoded(state, text: str, data, glyphs, decoder, spec_context) -> None:
    """The interpreter's append_decoded_text, for glyphs already decoded."""
    show = TextShow()
    show.setup(state, decoder, spec_context)
    try:
        show.decoded(text, data, glyphs, show.matrix[4], show.matrix[5])
    except BaseException:
        show.hand_over()
        raise
    show.hand_over()


def show_text(state, text: str, glyphs, decoder, adv_x, adv_y, glyph_paint) -> None:
    """A capture state's show_text: the string's glyphs captured and its run merged or
    added, at the text matrix, whose position it leaves."""
    show = TextShow()
    show.setup(state, decoder, None)
    show.show(
        text,
        glyphs,
        show.matrix[4],
        show.matrix[5],
        as_double(adv_x),
        as_double(adv_y),
        glyph_paint,
    )

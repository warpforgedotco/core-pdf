# SPDX-License-Identifier: AGPL-3.0-only

"""Check the dictionary keys core reads for abbreviations, field quadding, annotation
replies and pop-ups, a parameter dictionary's font and overprint, and the validate
package's profiles, on PDFs written here.

- E (ISO 32000-2 14.9.5): a Span property list's E, and a structure element's E reached
  through its MCID, give the extracted span its expansion; JSON carries it and HTML marks
  the span as an abbreviation. A styled span's end spaces stay outside its markers.
  StructureElement.expansion reads it.
- Q (12.7.4.3): a text or choice field's alignment comes from its own Q, else the nearest
  ancestor's down the Kids (with or without a Parent entry) or up the Parent entries of a
  field Fields names below the root (which also inherits its name, FT and V), else the
  AcroForm dictionary's, else left; a button has none.
- IRT, RT and Popup (12.5.6.2, 12.5.6.14): an annotation records the one it replies to
  (RT R, the default) or is grouped with (RT Group), a markup annotation its pop-up, and a
  pop-up its markup annotation, as indexes in the page's annotations and ids in JSON.
  An annotation dropped because its record cannot be built shifts none of them, and a
  reply whose target is not among them has no reply type. A dictionary Annots repeats is
  linked to at its first position.
- Font in a parameter dictionary (8.4.5) selects the font and size as Tf does: text shown
  after gs with no Tf is extracted at that size, a later Tf loads its own font even
  under the gs font's BaseFont, and the pdfminer facade, like pdfminer.six, ignores it.
- OP, op and OPM (8.6.7) are carried in the graphics state, op following OP when absent
  or invalid, and restored by Q. Font and the overprint entries apply or fail on their
  own, whatever the dictionary's other entries do, and a gs font that cannot be loaded
  leaves the text state as it was.
- core-pdf-validate's veraPDF profiles take their editions from STANDARD_PROFILES and
  refuse an identifier that table lacks.

uv run --all-packages python scripts/check_spec_keys.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_pdf import PdfDocument  # noqa: E402
from core_pdf.impl.output.serialize import document_to_html, document_to_json_dict  # noqa: E402
from core_pdf.impl.spec.s_07_content.operations import ContentOperands  # noqa: E402
from scripts.check_patterns import document, stream  # noqa: E402

HELVETICA = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"


def page_pdf(
    content: bytes,
    *,
    catalog: bytes = b"",
    page: bytes = b"",
    resources: bytes = b"/Font << /F1 5 0 R >>",
    extra: tuple[bytes, ...] = (),
) -> bytes:
    """A one-page PDF: catalog 1, pages 2, page 3, content 4, Helvetica 5, then extra from
    object 6."""
    return document(
        [
            b"<< /Type /Catalog /Pages 2 0 R " + catalog + b">>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << " + resources + b" >> " + page + b">>",
            stream(b"", content),
            HELVETICA,
            *extra,
        ]
    )


def expectation(failures: list[str], name: str, got: object, expected: object) -> None:
    if got != expected:
        failures.append(f"{name}: got {got!r}, expected {expected!r}")


def expanded_spans(data: bytes) -> tuple[list[tuple[str, str | None]], Any, str]:
    with PdfDocument(data) as pdf:
        extracted = pdf.extract()
    spans = [
        (span.text, span.expansion)
        for page in extracted.pages
        for block in page.blocks
        for line in block.lines
        for span in line.styled_spans()
    ]
    return spans, document_to_json_dict(extracted), document_to_html(extracted)


SPAN_CONTENT = b"BT /F1 12 Tf 72 700 Td (See ) Tj %s (Dr. ) Tj EMC (Smith) Tj ET"


def check_span_expansion(failures: list[str]) -> None:
    spans, payload, html = expanded_spans(page_pdf(SPAN_CONTENT % b"/Span << /E (Doctor) >> BDC"))
    expectation(
        failures, "Span E spans", spans, [("See", None), (" Dr.", "Doctor"), (" Smith", None)]
    )
    json_spans = [span for line in payload["lines"] for span in line["spans"]]
    expectation(
        failures,
        "Span E in JSON",
        [span.get("expansion") for span in json_spans],
        [None, "Doctor", None],
    )
    if ' <abbr title="Doctor">Dr.</abbr> ' not in html:
        failures.append(f"Span E in HTML: no abbr element in {html!r}")
    # ActualText replaces the content; E does not.
    spans, _, _ = expanded_spans(page_pdf(SPAN_CONTENT % b"/Span << /Alt (x) >> BDC"))
    expectation(
        failures, "Span without E", spans, [("See", None), (" Dr.", None), (" Smith", None)]
    )


def check_structure_expansion(failures: list[str]) -> None:
    data = page_pdf(
        SPAN_CONTENT % b"/Span << /MCID 0 >> BDC",
        catalog=b"/MarkInfo << /Marked true >> /StructTreeRoot 6 0 R ",
        page=b"/StructParents 0 ",
        extra=(
            b"<< /Type /StructTreeRoot /K [7 0 R] /ParentTree << /Nums [0 [8 0 R]] >> >>",
            b"<< /Type /StructElem /S /P /P 6 0 R /Pg 3 0 R /K [8 0 R] >>",
            b"<< /Type /StructElem /S /Span /P 7 0 R /Pg 3 0 R /K 0 /E (Doctor) >>",
        ),
    )
    with PdfDocument(data) as pdf:
        structure = pdf.structure
        elements = list(structure.find_all()) if structure is not None else []
        expectation(
            failures,
            "StructureElement.expansion",
            [(element.type, element.expansion) for element in elements],
            [("P", None), ("Span", "Doctor")],
        )
    spans, _, _ = expanded_spans(data)
    expectation(
        failures, "structure E spans", spans, [("See", None), (" Dr.", "Doctor"), (" Smith", None)]
    )


def field(name: bytes, rect: bytes, entries: bytes = b"") -> bytes:
    return (
        b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (" + name + b") /Rect [" + rect + b"] "
        b"/P 3 0 R " + entries + b">>"
    )


def check_quadding(failures: list[str]) -> None:
    objects = (
        field(b"own", b"72 700 272 720", b"/Q 2 /V (a)"),
        b"<< /T (group) /FT /Tx /Q 1 /Kids [8 0 R] >>",
        b"<< /Type /Annot /Subtype /Widget /Parent 7 0 R /T (kid) /Rect [72 650 272 670] "
        b"/P 3 0 R >>",
        field(b"form", b"72 600 272 620"),
        b"<< /Type /Annot /Subtype /Widget /FT /Btn /T (box) /Rect [72 550 92 570] /Q 2 "
        b"/P 3 0 R >>",
        field(b"bad", b"72 500 272 520", b"/Q 7"),
    )
    for form_q, form_default in ((b"", "left"), (b"/Q 2 ", "right")):
        data = page_pdf(
            b"",
            catalog=b"/AcroForm << /Fields [6 0 R 7 0 R 9 0 R 10 0 R 11 0 R] " + form_q + b">> ",
            page=b"/Annots [6 0 R 8 0 R 9 0 R 10 0 R 11 0 R] ",
            extra=objects,
        )
        with PdfDocument(data) as pdf:
            extracted = pdf.extract()
        fields = {
            item.name: item.alignment for page in extracted.pages for item in page.form_fields
        }
        expectation(
            failures,
            f"field alignment, AcroForm {form_q!r}",
            fields,
            {
                "own": "right",
                "group": "center",
                "group.kid": "center",
                "form": form_default,
                "box": None,
                "bad": "left",
            },
        )
        payload: Any = document_to_json_dict(extracted)
        expectation(
            failures,
            f"field alignment in JSON, AcroForm {form_q!r}",
            {item["name"]: item.get("alignment") for item in payload["form_fields"]},
            fields,
        )


def check_quadding_down_kids(failures: list[str]) -> None:
    """Q comes down Kids as FT does, to a widget with no Parent entry."""
    data = page_pdf(
        b"",
        catalog=b"/AcroForm << /Fields [6 0 R] >> ",
        page=b"/Annots [7 0 R] ",
        extra=(
            b"<< /T (top) /FT /Tx /Q 2 /Kids [7 0 R] >>",
            b"<< /Type /Annot /Subtype /Widget /Rect [72 700 272 720] /P 3 0 R >>",
        ),
    )
    with PdfDocument(data) as pdf:
        extracted = pdf.extract()
    expectation(
        failures,
        "field alignment down Kids",
        [(item.field_type, item.alignment) for item in extracted.pages[0].form_fields],
        [("Tx", "right"), ("Tx", "right")],
    )


def annotation(entries: bytes, rect: bytes = b"72 700 92 720") -> bytes:
    return b"<< /Type /Annot /Rect [" + rect + b"] /P 3 0 R " + entries + b">>"


def check_replies(failures: list[str]) -> None:
    data = page_pdf(
        b"",
        page=b"/Annots [6 0 R 7 0 R 8 0 R 9 0 R 10 0 R] ",
        extra=(
            annotation(b"/Subtype /Text /Contents (note) /Popup 7 0 R"),
            annotation(b"/Subtype /Popup /Parent 6 0 R", b"100 600 300 700"),
            annotation(b"/Subtype /Text /Contents (reply) /IRT 6 0 R"),
            annotation(b"/Subtype /Text /Contents (grouped) /IRT 6 0 R /RT /Group"),
            annotation(b"/Subtype /Text /Contents (other) /IRT 8 0 R /RT /Unknown"),
        ),
    )
    with PdfDocument(data) as pdf:
        extracted = pdf.extract()
    annotations = extracted.pages[0].annotations
    expectation(
        failures,
        "annotation links",
        [
            (item.in_reply_to, item.reply_type, item.popup, item.popup_parent)
            for item in annotations
        ],
        [
            (None, None, 1, None),
            (None, None, None, 0),
            (0, "R", None, None),
            (0, "Group", None, None),
            (2, "R", None, None),
        ],
    )
    payload: Any = document_to_json_dict(extracted)
    expectation(
        failures,
        "annotation links in JSON",
        [
            tuple(item.get(key) for key in ("in_reply_to", "reply_type", "popup", "popup_parent"))
            for item in payload["annotations"]
        ],
        [
            (None, None, "p1:annotation:1", None),
            (None, None, None, "p1:annotation:0"),
            ("p1:annotation:0", "R", None, None),
            ("p1:annotation:0", "Group", None, None),
            ("p1:annotation:2", "R", None, None),
        ],
    )


def check_links_past_dropped(failures: list[str]) -> None:
    """An annotation whose record cannot be built (a Dest of broken UTF-16) is dropped,
    and the links of the others still name the right records; a reply whose target is
    not among them has no reply type."""
    data = page_pdf(
        b"",
        page=b"/Annots [6 0 R 7 0 R 8 0 R 9 0 R 10 0 R] ",
        extra=(
            annotation(b"/Subtype /Text /Contents (a) /Popup 9 0 R"),
            annotation(b"/Subtype /Text /Contents (bad) /Dest <FEFF00>"),
            annotation(b"/Subtype /Text /Contents (c) /IRT 6 0 R /Popup 9 0 R"),
            annotation(b"/Subtype /Popup /Parent 6 0 R", b"100 600 300 700"),
            annotation(b"/Subtype /Text /Contents (e) /IRT 7 0 R /RT /Group"),
        ),
    )
    with PdfDocument(data) as pdf:
        extracted = pdf.extract()
    expectation(
        failures,
        "annotation links past a dropped record",
        [
            (item.contents, item.in_reply_to, item.reply_type, item.popup, item.popup_parent)
            for item in extracted.pages[0].annotations
        ],
        [
            ("a", None, None, 2, None),
            ("c", 0, "R", 2, None),
            ("", None, None, None, 0),
            ("e", None, None, None, None),
        ],
    )


def check_abbreviation_markup(failures: list[str]) -> None:
    """The space before a bold abbreviation stays outside its abbr and bold markers."""
    from core_pdf.impl.output.model import TextLine, TextSpan
    from core_pdf.impl.output.serialize import markdown_line

    line = TextLine(
        "See CIA", spans=(TextSpan("See"), TextSpan(" CIA", bold=True, expansion="Central Agency"))
    )
    expectation(
        failures,
        "abbreviation markup",
        markdown_line(line),
        'See <abbr title="Central Agency">**CIA**</abbr>',
    )


def check_fields_listed_below_root(failures: list[str]) -> None:
    """A field AcroForm's Fields names below the root inherits through its Parent entries:
    the qualified name, FT, V and Q."""
    data = page_pdf(
        b"",
        catalog=b"/AcroForm << /Fields [7 0 R] >> ",
        page=b"/Annots [7 0 R] ",
        extra=(
            b"<< /T (top) /FT /Tx /V (value) /Q 2 /Kids [7 0 R] >>",
            b"<< /Type /Annot /Subtype /Widget /Parent 6 0 R /T (kid) "
            b"/Rect [72 700 272 720] /P 3 0 R >>",
        ),
    )
    with PdfDocument(data) as pdf:
        extracted = pdf.extract()
    expectation(
        failures,
        "field listed below the root",
        [
            (item.name, item.field_type, item.value_text, item.alignment)
            for item in extracted.pages[0].form_fields
        ],
        [("top.kid", "Tx", "value", "right")],
    )


def check_repeated_annotation(failures: list[str]) -> None:
    """A dictionary Annots lists twice is linked to at its first position, and the
    annotation JSON without page ids gives indexes."""
    from core_pdf.impl.output.serialize import annotation_to_json_dict

    data = page_pdf(
        b"",
        page=b"/Annots [6 0 R 7 0 R 6 0 R] ",
        extra=(
            annotation(b"/Subtype /Text /Contents (a) /Popup 7 0 R"),
            annotation(b"/Subtype /Popup /Parent 6 0 R", b"100 600 300 700"),
        ),
    )
    with PdfDocument(data) as pdf:
        extracted = pdf.extract()
    annotations = extracted.pages[0].annotations
    expectation(
        failures,
        "repeated annotation links",
        [(item.popup, item.popup_parent) for item in annotations],
        [(1, None), (None, 0), (1, None)],
    )
    try:
        payload = annotation_to_json_dict(annotations[1])
    except TypeError as error:
        failures.append(f"annotation JSON without ids: {error}")
    else:
        expectation(failures, "annotation JSON without ids", payload["popup_parent"], 0)
    # A repeated dictionary that names itself links nowhere, whichever copy.
    data = page_pdf(
        b"",
        page=b"/Annots [6 0 R 7 0 R 6 0 R] ",
        extra=(
            annotation(b"/Subtype /Text /Contents (a) /IRT 6 0 R /Popup 6 0 R"),
            annotation(b"/Subtype /Text /Contents (b)"),
        ),
    )
    with PdfDocument(data) as pdf:
        extracted = pdf.extract()
    expectation(
        failures,
        "repeated annotation naming itself",
        [
            (item.in_reply_to, item.reply_type, item.popup)
            for item in extracted.pages[0].annotations
        ],
        [(None, None, None)] * 3,
    )


def check_span_markup_spaces(failures: list[str]) -> None:
    """Spaces at a styled span's ends stay outside its Markdown and HTML markers."""
    from core_pdf.impl.output.model import TextLine, TextSpan
    from core_pdf.impl.output.serialize import html_line, markdown_line

    line = TextLine(
        "a word b", spans=(TextSpan("a "), TextSpan("word ", bold=True), TextSpan("b", italic=True))
    )
    expectation(failures, "Markdown span spaces", markdown_line(line), "a **word** *b*")
    expectation(failures, "HTML span spaces", html_line(line), "a <strong>word</strong> <em>b</em>")
    # A styled span of spaces only gets no markers.
    spaces = TextLine(
        "word next", spans=(TextSpan("word"), TextSpan(" ", bold=True), TextSpan("next"))
    )
    expectation(failures, "Markdown span of spaces", markdown_line(spaces), "word next")
    expectation(failures, "HTML span of spaces", html_line(spaces), "word<strong> </strong>next")
    # An underlined blank keeps its tag, and a blank abbreviation its abbr, in both.
    blank = TextLine(
        "Name:    ", spans=(TextSpan("Name:"), TextSpan("    ", underline=True, bold=True))
    )
    expectation(failures, "Markdown underlined blank", markdown_line(blank), "Name:<u>    </u>")
    expectation(
        failures, "HTML underlined blank", html_line(blank), "Name:<strong><u>    </u></strong>"
    )
    abbreviated = TextLine(
        "a  b", spans=(TextSpan("a"), TextSpan("  ", expansion="x"), TextSpan("b"))
    )
    expectation(
        failures,
        "Markdown blank abbreviation",
        markdown_line(abbreviated),
        'a<abbr title="x">  </abbr>b',
    )


def check_extgstate_font(failures: list[str]) -> None:
    data = page_pdf(
        b"BT /GS1 gs 72 700 Td (Hello) Tj ET BT /F1 10 Tf /GS1 gs 72 600 Td (World) Tj ET",
        resources=b"/Font << /F1 5 0 R >> /ExtGState << /GS1 6 0 R >>",
        extra=(
            b"<< /Type /ExtGState /Font [7 0 R 24] >>",
            HELVETICA.replace(b"Helvetica", b"Courier"),
        ),
    )
    with PdfDocument(data) as pdf:
        runs = pdf.pages[0].get_page_program().body.runs
        shown = [(run.text, round(run.font_size, 3), run.font_name) for run in runs]
        expectation(
            failures,
            "gs Font runs",
            shown,
            [("Hello", 24.0, "Courier"), ("World", 24.0, "Courier")],
        )
    # Tf after a gs font loads its own font, even under a name that is that font's BaseFont.
    reused = page_pdf(
        b"BT /GS1 gs 72 700 Td (AAA) Tj /Helvetica 12 Tf 0 -20 Td (AAA) Tj ET",
        resources=b"/Font << /Helvetica 6 0 R >> /ExtGState << /GS1 7 0 R >>",
        extra=(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            b"/Encoding << /Differences [65 /Z] >> >>",
            b"<< /Type /ExtGState /Font [5 0 R 12] >>",
        ),
    )
    with PdfDocument(reused) as pdf:
        runs = pdf.pages[0].get_page_program().body.runs
        expectation(failures, "Tf after gs Font", [run.text for run in runs], ["AAA", "ZZZ"])
    from core_pdf_compat import pdfplumber

    with pdfplumber.open(_bytes_path(data)) as plumbed:
        world = [
            (char["size"], char["fontname"])
            for char in plumbed.pages[0].chars
            if char["text"] in "World" and char["y0"] < 650
        ]
    # pdfminer.six ignores gs, so the second text keeps the font Tf set.
    expectation(failures, "pdfplumber facade ignores gs Font", world, [(10.0, "Helvetica")] * 5)


def _bytes_path(data: bytes) -> Any:
    from io import BytesIO

    return BytesIO(data)


def check_overprint(failures: list[str]) -> None:
    from core_pdf.impl.capture.recording import TextState

    class Probe(TextState):
        """Records the overprint state at each BMC."""

        seen: list[tuple[object, ...]]
        opacities: list[tuple[float, float]]

        def op_BMC(self, operands: ContentOperands, depth: int) -> None:
            graphics = self.graphics
            self.seen.append(
                tuple(
                    getattr(graphics, name, None)
                    for name in ("stroke_overprint", "fill_overprint", "overprint_mode")
                )
            )
            self.opacities.append((graphics.stroke_opacity, graphics.fill_opacity))
            super().op_BMC(operands, depth)

    cases = (
        (b"/OP true", (True, True, 0)),
        (b"/OP true /op false /OPM 1", (True, False, 1)),
        (b"/op true", (False, True, 0)),
        (b"/OPM 2", (False, False, 0)),
        (b"/OP true /op 5", (True, True, 0)),
    )
    for entries, expected in cases:
        data = page_pdf(
            b"q /GS1 gs /P BMC EMC Q /P BMC EMC",
            resources=b"/ExtGState << /GS1 6 0 R >>",
            extra=(b"<< /Type /ExtGState " + entries + b" >>",),
        )
        with PdfDocument(data) as pdf:
            state = Probe(pdf)
            state.seen = []
            state.opacities = []
            try:
                pdf.pages[0].consume_contents(state)
            finally:
                state.release()
        expectation(
            failures,
            f"overprint with {entries!r}, then after Q",
            state.seen,
            [expected, (False, False, 0)],
        )

    # A Font or overprint entry whose handling fails, as a font that cannot be loaded
    # would, is declined alone: the opacities beside it still apply.
    class Failing(Probe):
        def apply_extgstate_font(self, value: object) -> None:
            raise ValueError("unloadable font")

        def apply_extgstate_overprint(self, extgstate: dict[str, Any]) -> None:
            raise ValueError("unreadable overprint")

    for entries in (b"/Font [7 0 R 12]", b"/OP true /OPM 1"):
        data = page_pdf(
            b"/GS1 gs /P BMC EMC",
            resources=b"/ExtGState << /GS1 6 0 R >>",
            extra=(b"<< /Type /ExtGState /CA 0.5 /ca 0.25 " + entries + b" >>", HELVETICA),
        )
        with PdfDocument(data) as pdf:
            state = Failing(pdf)
            state.seen = []
            state.opacities = []
            try:
                pdf.pages[0].consume_contents(state)
            finally:
                state.release()
        expectation(
            failures, f"opacity beside a failing {entries!r}", state.opacities, [(0.5, 0.25)]
        )
    # A failing earlier entry (ca) does not keep Font and OP from applying.
    data = page_pdf(
        b"/GS1 gs /P BMC EMC BT 72 700 Td (Hi) Tj ET",
        resources=b"/ExtGState << /GS1 6 0 R >>",
        extra=(b"<< /Type /ExtGState /Font [7 0 R 20] /OP true /ca (bad) >>", HELVETICA),
    )
    with PdfDocument(data) as pdf:
        state = Probe(pdf)
        state.seen = []
        state.opacities = []
        try:
            pdf.pages[0].consume_contents(state)
            state.run_accumulator.flush()
            runs = [(run.text, run.font_size) for run in state.runs]
        finally:
            state.release()
    expectation(
        failures,
        "Font and OP beside a failing ca",
        (state.seen, runs),
        ([(True, True, 0)], [("Hi", 20.0)]),
    )

    # A gs font that cannot be loaded leaves the state whole: the Tf after it loads its own
    # font rather than keep F1's under the gs font's BaseFont.
    class Unloadable(TextState):
        def decoder_for(self, font_reference: object, font: object, resources: Any) -> Any:
            if isinstance(font, dict) and font.get("Unloadable") is not None:
                raise ValueError("unloadable font")
            return super().decoder_for(font_reference, font, resources)

    data = page_pdf(
        b"BT /F1 12 Tf 72 700 Td (AAA) Tj /GS1 gs /Helvetica 12 Tf 0 -20 Td (AAA) Tj ET",
        resources=b"/Font << /F1 6 0 R /Helvetica 5 0 R >> /ExtGState << /GS1 7 0 R >>",
        extra=(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            b"/Encoding << /Differences [65 /Z] >> >>",
            b"<< /Type /ExtGState /Font [8 0 R 12] >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Unloadable true >>",
        ),
    )
    with PdfDocument(data) as pdf:
        unloadable = Unloadable(pdf)
        try:
            pdf.pages[0].consume_contents(unloadable)
            unloadable.run_accumulator.flush()
            texts = [run.text for run in unloadable.runs]
        finally:
            unloadable.release()
    expectation(failures, "Tf after an unloadable gs Font", texts, ["ZZZ", "AAA"])
    check_extgstate_parse_errors(failures, Probe)


def check_extgstate_parse_errors(failures: list[str], probe: Any) -> None:
    """A PdfParseError in one group of a parameter dictionary's entries, tolerant or strict."""
    # Tolerant: a gs font whose loading raises PdfParseError is declined alone; the page's
    # content goes on, and OP and the opacities beside it apply.
    from core_pdf.impl.spec.exceptions import PdfParseError

    class Broken(probe):  # type: ignore[misc, valid-type]
        def decoder_for(self, font_reference: object, font: object, resources: Any) -> Any:
            if isinstance(font, dict) and font.get("Broken") is not None:
                raise PdfParseError("broken font program")
            return super().decoder_for(font_reference, font, resources)

    data = page_pdf(
        b"/GS1 gs /P BMC EMC BT /F1 12 Tf 72 700 Td (After) Tj ET",
        resources=b"/Font << /F1 5 0 R >> /ExtGState << /GS1 6 0 R >>",
        extra=(
            b"<< /Type /ExtGState /Font [7 0 R 20] /OP true /ca 0.25 >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Broken true >>",
        ),
    )
    with PdfDocument(data) as pdf:
        state = Broken(pdf)
        state.seen = []
        state.opacities = []
        try:
            pdf.pages[0].consume_contents(state)
            state.run_accumulator.flush()
            texts = [run.text for run in state.runs]
        except PdfParseError as error:
            texts = [f"raised {error}"]
        finally:
            state.release()
    expectation(
        failures,
        "tolerant gs Font raising PdfParseError",
        (state.seen, state.opacities, texts),
        ([(True, True, 0)], [(1.0, 0.25)], ["After"]),
    )

    # Tolerant: a PdfParseError from the older entries (a soft mask that raises it) is
    # declined by op_gs; the entries before it and the page's content go on.
    class BadMask(probe):  # type: ignore[misc, valid-type]
        def resolve_soft_mask(self, value: object) -> Any:
            raise PdfParseError("broken soft mask")

    data = page_pdf(
        b"/GS1 gs /P BMC EMC BT /F1 12 Tf 72 700 Td (After) Tj ET",
        resources=b"/Font << /F1 5 0 R >> /ExtGState << /GS1 6 0 R >>",
        extra=(b"<< /Type /ExtGState /ca 0.25 /SMask << /S /Luminosity /G 7 0 R >> /OP true >>",),
    )
    with PdfDocument(data) as pdf:
        state = BadMask(pdf)
        state.seen = []
        state.opacities = []
        try:
            pdf.pages[0].consume_contents(state)
            state.run_accumulator.flush()
            texts = [run.text for run in state.runs]
        except PdfParseError as error:
            texts = [f"raised {error}"]
        finally:
            state.release()
    expectation(
        failures,
        "tolerant soft mask raising PdfParseError",
        (state.seen, state.opacities, texts),
        ([(True, True, 0)], [(1.0, 0.25)], ["After"]),
    )

    # Strict: every group runs, then the first group's error is raised.
    class Strict(probe):  # type: ignore[misc, valid-type]
        def reject(self, error: Exception, context: str, fallback: Any) -> Any:
            raise error

    data = page_pdf(
        b"",
        resources=b"/ExtGState << /GS1 6 0 R /GS2 7 0 R >>",
        extra=(
            b"<< /Type /ExtGState /D 5 /Font [8 0 R 20] /OP true >>",
            b"<< /Type /ExtGState /ca 0.25 /Font 3 /OP true /OPM 1 >>",
            HELVETICA,
        ),
    )
    with PdfDocument(data) as pdf:
        for name, expected in (
            ("GS1", ("Helvetica", 20.0, True, 0, "D must hold")),
            ("GS2", (None, 0.0, True, 1, "Font must hold")),
        ):
            state = Strict(pdf)
            try:
                resources = pdf.pages[0].resources
                state.resources = resources
                extgstates: Any = pdf.resolver.resolve(resources["ExtGState"])
                extgstate = pdf.resolver.resolve(extgstates[name])
                raised = ""
                try:
                    state.apply_extgstate(extgstate)
                except PdfParseError as error:
                    raised = str(error)
                graphics = state.graphics
                got = (
                    graphics.current_font if graphics.current_font != "F1" else None,
                    graphics.font_size if graphics.current_font else 0.0,
                    graphics.stroke_overprint,
                    graphics.overprint_mode,
                    raised[: len(expected[4])],
                )
            finally:
                state.release()
            expectation(failures, f"strict {name}", got, expected)


def check_validate_profiles(failures: list[str]) -> None:
    from core_pdf.impl.spec.standards import STANDARD_PROFILES, get_standard_profile
    from core_pdf_validate import verapdf

    for identifier, (_, _, edition) in verapdf.PROFILES.items():
        profile = get_standard_profile(identifier)
        expectation(
            failures,
            f"validate profile {identifier}",
            edition,
            None if profile is None else profile.edition,
        )
    builder = getattr(verapdf, "verapdf_profiles", None)
    if builder is None:
        failures.append("validate profiles: not built from STANDARD_PROFILES")
        return
    try:
        builder({"pdfa-9z": ("9z", "PDF/A-9z validation profile")}, STANDARD_PROFILES)
    except ValueError:
        pass
    else:
        failures.append("validate profiles: an identifier STANDARD_PROFILES lacks was accepted")


CHECKS: tuple[tuple[str, Callable[[list[str]], None]], ...] = (
    ("Span E", check_span_expansion),
    ("StructElem E", check_structure_expansion),
    ("field Q", check_quadding),
    ("field Q down Kids", check_quadding_down_kids),
    ("field listed below the root", check_fields_listed_below_root),
    ("annotation IRT, RT, Popup", check_replies),
    ("annotation links past a dropped record", check_links_past_dropped),
    ("abbreviation markup", check_abbreviation_markup),
    ("repeated annotation", check_repeated_annotation),
    ("span markup spaces", check_span_markup_spaces),
    ("ExtGState Font", check_extgstate_font),
    ("ExtGState OP, op, OPM", check_overprint),
    ("validate profiles", check_validate_profiles),
)


def main() -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    failed = 0
    for name, check in CHECKS:
        failures: list[str] = []
        try:
            check(failures)
        except Exception as error:  # noqa: BLE001
            failures.append(f"raised {type(error).__name__}: {error}")
        if failures:
            failed += 1
            print(f"FAIL {name}")
            for failure in failures:
                print(f"  {failure}")
        else:
            print(f"ok   {name}")
    print(f"{len(CHECKS) - failed} of {len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Learned Unicode overlays must remain aligned with captured source glyphs."""

from dataclasses import replace

import pytest

from core_pdf.impl._impl.extract.contracts import PageAnalysis as NativePageAnalysis
from core_pdf.impl._impl.model.glyphs import GlyphCluster, GlyphObservation
from core_pdf.impl._impl.model.runs import TextRun
from core_pdf_ocr.impl.extract import capture


def internal_glyph(text: str, font: object, *, x: float = 0, seqno: int = 0) -> GlyphObservation:
    return GlyphObservation(
        text,
        (x, 0, x + 5, 10),
        (x, 0, x + 5, 10),
        seqno,
        code_bytes=b"a",
        font_decoder=font,
        unicode_source="to_unicode",
        confidence=0.9,
    )


def internal_cluster(text: str, *glyphs: GlyphObservation) -> GlyphCluster:
    box = glyphs[0].advance_bbox if glyphs else (0, 0, 5, 10)
    return GlyphCluster(0, text, glyphs, box, box, None, 0.9)


def internal_run(text: str, *clusters: GlyphCluster) -> TextRun:
    return TextRun(text, 0, 0, 50, 10, 0, 10, 10, 5, 0, 0, 0, glyph_clusters=clusters)


def test_learned_overlay_preserves_spacing_geometry_and_original_clusters() -> None:
    font = object()
    first = internal_glyph("a", font)
    second = replace(internal_glyph("b", font, x=10), code_bytes=b"b")
    run = internal_run(" a  b ", internal_cluster("a", first), internal_cluster("b", second))
    replacements: dict[int, str] = {}
    result = capture.internal_apply_learned_unicode_to_run(
        run, {font: {b"a": "A"}}, glyph_replacements=replacements
    )
    assert result.text == " A  b "
    assert run.text == " a  b "
    assert result.advance_bbox == run.advance_bbox
    assert result.glyph_clusters is run.glyph_clusters
    assert result.provenance[-1] == ("unicode_source", "learned_ocr")
    assert replacements == {id(first): "A"}


@pytest.mark.parametrize("replacement", ["", "AB", " ", "\n", "\x00"])
def test_unusable_learned_values_cannot_replace_a_glyph(replacement: str) -> None:
    font = object()
    run = internal_run("a", internal_cluster("a", internal_glyph("a", font)))
    applied: dict[int, str] = {}
    assert (
        capture.internal_apply_learned_unicode_to_run(
            run, {font: {b"a": replacement}}, glyph_replacements=applied
        )
        is run
    )
    assert applied == {}


@pytest.mark.parametrize("source", ["actual_text", "structure_actual_text"])
def test_actual_text_takes_precedence_over_learned_unicode(source: str) -> None:
    font = object()
    run = internal_run("a", internal_cluster("a", internal_glyph("a", font))).replace(
        provenance=(("unicode_source", source),)
    )
    assert capture.internal_apply_learned_unicode_to_run(run, {font: {b"a": "X"}}) is run


@pytest.mark.parametrize("text", ["missing", "prefix a", "a suffix"])
def test_unaligned_clusters_leave_text_and_replacement_map_untouched(text: str) -> None:
    font = object()
    run = internal_run(text, internal_cluster("a", internal_glyph("a", font)))
    applied = {999: "preserved"}
    assert (
        capture.internal_apply_learned_unicode_to_run(
            run, {font: {b"a": "X"}}, glyph_replacements=applied
        )
        is run
    )
    assert applied == {999: "preserved"}


@pytest.mark.parametrize(
    ("rotation", "first_x", "second_x", "allowed"),
    [
        (0, 10, 0, False),
        (0, 0, 10, True),
        (90, 10, 0, True),
        (180, 0, 10, False),
        (180, 10, 0, True),
        (270, 0, 10, False),
    ],
)
def test_overlay_respects_text_show_order_and_rotation(
    rotation: int, first_x: float, second_x: float, allowed: bool
) -> None:
    font = object()
    first = internal_glyph("a", font, x=first_x)
    second = internal_glyph("a", font, x=second_x, seqno=1)
    run = replace(
        internal_run("aa", internal_cluster("a", first), internal_cluster("a", second)),
        rotation_angle=rotation,
    )
    result = capture.internal_apply_learned_unicode_to_run(run, {font: {b"a": "X"}})
    assert result.text == ("XX" if allowed else "aa")
    if not allowed:
        assert result is run


def test_ligature_replacement_is_applied_once_and_updates_glyph_counts() -> None:
    font = object()
    first = internal_glyph("f", font)
    second = internal_glyph("i", font)
    run = internal_run("fi", internal_cluster("fi", first, second))
    applied: dict[int, str] = {}
    result = capture.internal_apply_learned_unicode_to_run(
        run, {font: {b"a": "A"}}, glyph_replacements=applied
    )
    assert result.text == "A"
    assert applied == {id(first): "A", id(second): ""}
    evidence = capture.internal_glyph_evidence_fields((first, second), (result,), applied)
    assert evidence.glyph_count == 1
    assert evidence.semantic_characters == 1
    assert evidence.authoritative_glyphs == 0
    assert evidence.heuristic_glyphs == 1


@pytest.mark.parametrize("different_font", [True, False])
def test_mixed_source_clusters_cannot_receive_one_learned_replacement(different_font: bool) -> None:
    font = object()
    first = internal_glyph("f", font)
    second = internal_glyph("i", object() if different_font else font)
    if not different_font:
        second = replace(second, code_bytes=b"other")
    run = internal_run("fi", internal_cluster("fi", first, second))
    assert capture.internal_apply_learned_unicode_to_run(run, {font: {b"a": "A"}}) is run


def test_empty_unmapped_and_unchanged_clusters_preserve_identity() -> None:
    font = object()
    glyph = internal_glyph("a", font)
    run = internal_run("a", internal_cluster(""), internal_cluster("a", glyph))
    assert capture.internal_apply_learned_unicode_to_run(run) is run
    assert capture.internal_apply_learned_unicode_to_run(run, {object(): {b"a": "X"}}) is run
    applied: dict[int, str] = {}
    assert (
        capture.internal_apply_learned_unicode_to_run(
            run, {font: {b"a": "a"}}, glyph_replacements=applied
        )
        is run
    )
    assert applied == {id(glyph): "a"}
    assert (
        capture.internal_apply_learned_unicode_to_run(internal_run("a"), {font: {b"a": "X"}}).text
        == "a"
    )


@pytest.mark.parametrize(
    ("source", "text", "confidence"),
    [("identity", "A", None), ("to_unicode", "\ufffd", 0.2), ("glyph_name", "A", 0.7)],
)
def test_repaired_glyph_evidence_reclassifies_only_applied_substitutions(
    source: str, text: str, confidence: float | None
) -> None:
    glyph = replace(internal_glyph(text, object()), unicode_source=source, confidence=confidence)
    untouched = internal_glyph("Z", object())
    evidence = capture.internal_glyph_evidence_fields((glyph, untouched), (), {id(glyph): "X"})
    assert evidence.glyph_count == 2
    assert evidence.semantic_characters == 2
    assert evidence.authoritative_glyphs == 1
    assert evidence.heuristic_glyphs == 1
    assert (
        evidence.unknown_glyphs
        == evidence.unsupported_glyphs
        == evidence.low_confidence_glyphs
        == 0
    )
    native = capture.internal_glyph_evidence_fields((glyph,), (), {})
    assert native.glyph_count == 1


@pytest.mark.parametrize("learned", [False, True])
def test_program_capture_applies_unicode_before_observations_and_evidence(learned: bool) -> None:
    from types import SimpleNamespace

    from core_pdf.impl._impl.capture.program import CapturedProgram, PageProgram
    from core_pdf.impl._impl.capture.records import CapturedDrawing
    from core_pdf_spec.types import PdfName

    font = object()
    glyph = internal_glyph("a", font)
    run = internal_run("a", internal_cluster("a", glyph))
    image = CapturedDrawing(
        1,
        None,
        None,
        kind="image",
        bbox=(0, 0, 100, 200),
        dictionary={"Filter": PdfName(b"FlateDecode")},
    )
    program = PageProgram(CapturedProgram(runs=(run,), glyphs=(glyph,), drawings=(image,)))
    page = SimpleNamespace(width=100, height=200, rotation=0)
    result = capture.internal_capture_from_program(
        page,
        program,
        structure=None,
        learned_unicode={font: {b"a": "X"}} if learned else None,
        fields=("field",),
        annotations=("annotation",),
    )
    assert result.observations.text == (("X",) if learned else ("a",))
    assert result.evidence.glyphs.heuristic_glyphs == int(learned)
    assert result.evidence.glyphs.authoritative_glyphs == int(not learned)
    assert result.evidence.image_filters == ("FlateDecode",)
    assert result.evidence.full_page_image
    assert result.fields == ("field",)
    assert result.annotations == ("annotation",)
    assert result.program is program
    assert program.runs[0].text == "a"


@pytest.mark.parametrize("trusted", [False, True])
def test_capture_enrichment_promotes_only_trusted_template_results(
    monkeypatch: pytest.MonkeyPatch,
    trusted: bool,
) -> None:
    from types import SimpleNamespace

    from core_pdf.impl._impl.capture.program import CapturedProgram, PageProgram
    from core_pdf.impl._impl.capture.records import CapturedDrawing, CapturedLine
    from core_pdf_ocr.impl.extract.ocr.newstroke import NewstrokeDecode

    drawing = CapturedDrawing(0, None, None, kind="stroke", bbox=(0, 0, 2, 2))
    program = PageProgram(
        CapturedProgram(drawings=(drawing,) * 10000, lines=(CapturedLine(0, 0, 2, 2),) * 70000)
    )
    decoded = NewstrokeDecode(
        runs=(internal_run("label"),),
        candidate_segments=10000,
        matched_segments=10000 if trusted else 0,
        characters=1000,
        sequences=100,
    )
    calls = []

    def decode(drawings: tuple[CapturedDrawing, ...]) -> NewstrokeDecode:
        calls.append(drawings)
        return decoded

    monkeypatch.setattr(capture, "decode_newstroke_drawings", decode)
    result = capture.internal_capture_from_program(
        SimpleNamespace(width=100, height=200, rotation=0), program, structure=None
    )
    assert calls == [program.drawings]
    assert result.evidence.vector_text_trusted is trusted
    assert result.observations.text == (("label",) if trusted else ())


def test_public_capture_entry_point_passes_selection_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from core_pdf.impl._impl.capture.program import PageProgram

    page = SimpleNamespace(width=100, height=200, rotation=0)
    native = capture.native_capture_from_program(page, PageProgram(), structure=None)
    supplied: dict[str, object] = {}

    def native_capture(target: object, **options: object) -> NativePageAnalysis:
        assert target is page
        supplied.update(options)
        return native

    monkeypatch.setattr(capture, "native_capture_page", native_capture)
    result = capture.capture_page(
        page, structure=None, hidden_layers=frozenset({"layer"}), fields=(), annotations=()
    )
    assert result.page is page
    assert supplied == {
        "structure": None,
        "hidden_layers": frozenset({"layer"}),
        "fields": (),
        "annotations": (),
    }


def test_cluster_without_observations_preserves_text_beside_a_learned_cluster() -> None:
    font = object()
    glyph = internal_glyph("b", font)
    run = internal_run("ab", internal_cluster("a"), internal_cluster("b", glyph))
    applied: dict[int, str] = {}
    result = capture.internal_apply_learned_unicode_to_run(
        run, {font: {b"a": "B"}}, glyph_replacements=applied
    )
    assert result.text == "aB"
    assert applied == {id(glyph): "B"}
    assert run.text == "ab"
    assert result.glyph_clusters is run.glyph_clusters


@pytest.mark.parametrize("leading", ["", " ", "\t"])
def test_learned_cluster_attributes_replacement_to_first_nonblank_observation(leading: str) -> None:
    font = object()
    blank = internal_glyph(leading, font)
    visible = internal_glyph("f", font)
    trailing = internal_glyph("i", font)
    run = internal_run("fi", internal_cluster("fi", blank, visible, trailing))
    applied: dict[int, str] = {}
    result = capture.internal_apply_learned_unicode_to_run(
        run, {font: {b"a": "A"}}, glyph_replacements=applied
    )
    assert result.text == "A"
    assert applied == {id(blank): "", id(visible): "A", id(trailing): ""}
    assert (blank.text, visible.text, trailing.text) == (leading, "f", "i")


def test_blank_observations_do_not_receive_invented_text_attribution() -> None:
    # The record model permits source text separate from observation text.
    # Learning may replace that source text without selecting a blank observation.
    font = object()
    glyphs = (internal_glyph("", font), internal_glyph(" ", font))
    run = internal_run("a", internal_cluster("a", *glyphs))
    applied: dict[int, str] = {}
    result = capture.internal_apply_learned_unicode_to_run(
        run, {font: {b"a": "A"}}, glyph_replacements=applied
    )
    assert result.text == "A"
    assert applied == {id(glyph): "" for glyph in glyphs}
    assert run.text == "a"

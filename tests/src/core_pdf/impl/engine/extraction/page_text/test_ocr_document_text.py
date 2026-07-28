from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from core_ocr.impl import coordinator, schematic, text_analysis
from core_ocr.impl.candidates import OcrCandidate, OcrPageTextResult
from core_ocr.impl.types import OcrTextChoice, OcrTextResult

from core_pdf import PdfDocument

SCORE_BENCH_SRC = Path(__file__).parents[7] / "tests" / "fixtures" / "SCORE-Bench" / "src"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            "2023-gmi-lab-call_p4-15-p003.pdf",
            "Topic Area 1",
        ),
        (
            "EPA_AirQualityLetter_Table-p001.pdf",
            "UNITED STATES ENVIRONMENTAL PROTECTION AGENCY",
        ),
    ],
)
def test_document_extract_preserves_ocr_text_without_resolved_geometry(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    expected: str,
) -> None:
    monkeypatch.setenv("CORE_PDF_OCR", "1")

    with PdfDocument.open(SCORE_BENCH_SRC / filename) as document:
        page = cast(Any, document.pages[0])
        page_text = page.extract_text()
        resolved_lines = page.extract_resolved_lines()
        document_text = document.extract().text

    assert expected in page_text
    assert expected in document_text
    assert resolved_lines


def test_dense_sparse_text_schematic_uses_tiled_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_PDF_OCR", "1")

    fixture = SCORE_BENCH_SRC / "VCAs_REV2_SCHEMATIC-p002.pdf"
    with PdfDocument.open(fixture) as document:
        page = cast(Any, document.pages[0])
        text = page.extract_text()
        diagnostics = page.extraction_cache["ocr_candidate_diagnostics"]

    assert text
    assert any(candidate["name"].endswith("_tiled") for candidate in diagnostics)


def test_candidate_text_diagnostics_are_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    page = cast(Any, SimpleNamespace(extraction_cache={}))
    candidate = OcrCandidate("full_page", OcrTextResult("alpha beta", 82))

    coordinator.record_ocr_candidate_diagnostics(
        page,
        [candidate],
        selected_candidate=candidate,
    )

    assert "text" not in page.extraction_cache["ocr_candidate_diagnostics"][0]
    assert "ocr_candidate_analysis" not in page.extraction_cache

    monkeypatch.setenv("CORE_PDF_CANDIDATE_ANALYSIS", "1")
    coordinator.record_ocr_candidate_diagnostics(
        page,
        [candidate],
        selected_candidate=candidate,
    )

    assert page.extraction_cache["ocr_candidate_analysis"][0]["text"] == "alpha beta"


@pytest.mark.parametrize(
    ("token_type", "evidence_count", "confidence", "expected"),
    [
        ("rail", 2, 60, True),
        ("reference", 1, 90, True),
        ("rail", 1, 89, False),
        (None, 3, 95, False),
    ],
)
def test_rendered_schematic_supplement_requires_typed_evidence(
    token_type: str | None,
    evidence_count: int,
    confidence: int,
    expected: bool,
) -> None:
    entry = schematic.SchematicSupplementEntry(
        token="gnd",
        key="gnd",
        token_type=token_type,
        evidence_count=evidence_count,
        confidence=confidence,
    )

    assert schematic.rendered_schematic_addition_is_safe(entry) is expected


def test_schematic_pin_choice_confidence_recovers_tiny_labels() -> None:
    row = {
        "text": "2",
        "conf": 19,
        "choices": (OcrTextChoice("2", 96),),
    }

    assert schematic.schematic_pin_choice_confidence(row, "2") == 96


def test_repeated_schematic_rails_are_capped_by_corroborated_population() -> None:
    candidates = (
        OcrCandidate("rendered_page_300dpi", OcrTextResult("GND " * 5 + "VCC " * 2, 80)),
        OcrCandidate("rendered_page_400dpi", OcrTextResult("GND " * 4 + "VCC " * 2, 82)),
        OcrCandidate("rendered_page_475dpi", OcrTextResult("GND " * 5 + "VCC " * 2, 84)),
    )

    cleaned = schematic.cap_overrepresented_schematic_rail_tokens(
        "header\n" + "GND\n" * 9 + "VCC\n" * 5,
        candidates,
    )

    assert cleaned.split().count("GND") == 5
    assert cleaned.split().count("VCC") == 5


def test_dense_table_cleanup_removes_form_mark_artifacts() -> None:
    text = "APPEARANCES....... 2\nEXHIBITS....... 6\nCERTIFICATE....... 115\nLabel ~ | *"

    assert coordinator.remove_dense_table_ocr_artifact_tokens(text) == (
        "APPEARANCES 2\nEXHIBITS 6\nCERTIFICATE 115\nLabel"
    )


def test_dense_table_cleanup_removes_systematic_embedded_scan_artifacts() -> None:
    text = "field~value " + "SS*«d ™¢ " * 4

    assert (
        coordinator.remove_dense_table_ocr_artifact_tokens(text)
        == ("fieldvalue " + "SSd " * 4).strip()
    )


def test_dense_table_decimal_repair_requires_systematic_mixed_separators() -> None:
    corrupted = " ".join(("933,9", "5·0", "4•7", "193,6", "42·3", "25•0"))

    assert coordinator.repair_dense_table_decimal_separators(corrupted) == (
        "933.9 5.0 4.7 193.6 42.3 25.0"
    )
    assert coordinator.repair_dense_table_decimal_separators("value 1,000 and 2,000") == (
        "value 1,000 and 2,000"
    )


def test_final_schematic_cleanup_removes_lowercase_ocr_fragments() -> None:
    valid = " ".join(["U2", "D1", "1"] * 8)
    text = f"{valid}\nenn of ca ae\ngnd u1 100k\nVCAs CHRIS MCDOWELL"

    assert schematic.remove_final_schematic_lowercase_fragments(text) == (
        f"{valid}\ngnd u1 100k\nVCAs CHRIS MCDOWELL"
    )
    assert "enn" in schematic.remove_final_schematic_lowercase_fragments(text, "enn")
    assert "use use" in schematic.remove_final_schematic_lowercase_fragments(f"{text}\nuse use")
    assert "in" in schematic.remove_final_schematic_lowercase_fragments(f"{text}\nin")


def test_schematic_artifact_cleanup_removes_standalone_speckles() -> None:
    text = "GPIO1 [ GPIO2 } GPIO3 . GPIO4 GPIO5} GPIO6. " + "[ ] { } . " * 20

    assert schematic.remove_schematic_ocr_artifact_tokens(text) == (
        "GPIO1 GPIO2 GPIO3 GPIO4 GPIO5 GPIO6"
    )


def test_fragmented_schematic_value_repair_rejoins_zero_glyphs_and_units() -> None:
    text = "R1 10 @ k C1 10 O p R2 4.7 k C2 100 nF R3 10@k pin 2 VCA"

    assert schematic.repair_fragmented_schematic_value_tokens(text) == (
        "R1 100k C1 100p R2 4.7k C2 100nF R3 100k pin 2 VCA"
    )


def test_schematic_spaced_date_repair_restores_missing_separators() -> None:
    text = "Board title 4 28 2022 revision 2"

    assert schematic.repair_schematic_spaced_dates(text) == "Board title 4/28/2022 revision 2"


def test_schematic_spaced_date_repair_preserves_duplicate_when_one_date_is_intact() -> None:
    text = "Board title 4/28/2022\nAlternate OCR 4 28 2022"

    assert schematic.repair_schematic_spaced_dates(text) == text


def test_schematic_slash_repair_requires_candidate_consensus_and_matching_components() -> None:
    candidates = (
        OcrCandidate(
            "rendered_page_300dpi",
            OcrTextResult("GPIO19/USB_D-/ADC2_CH8 R1/R2", 72),
        ),
        OcrCandidate(
            "rendered_page_400dpi",
            OcrTextResult("GPIO19/USB_D-/ADC2_CH8\nR1/R2 extra", 78),
        ),
    )
    text = "GPIO19 USB_D- ADC2_CH8 and R1 R2 plus ADC2_CH7 DAC_1"

    assert schematic.restore_consensus_schematic_slash_tokens(text, candidates) == (
        "GPIO19/USB_D-/ADC2_CH8 and R1/R2 plus ADC2_CH7 DAC_1"
    )


def test_schematic_slash_repair_does_not_invent_missing_components() -> None:
    candidate = OcrCandidate("rendered_page_300dpi", OcrTextResult("A/B/C", 80))

    assert schematic.restore_consensus_schematic_slash_tokens("A B", (candidate,)) == "A B"


def test_schematic_slash_repair_can_restore_supported_suffix_without_missing_prefix() -> None:
    candidates = (
        OcrCandidate("rendered_page_300dpi", OcrTextResult("GPIO18/ADC2_CH7/DAC_1", 72)),
        OcrCandidate("rendered_page_400dpi", OcrTextResult("GPIO18/ADC2_CH7/DAC_1 extra", 78)),
    )

    assert (
        schematic.restore_consensus_schematic_slash_tokens(
            "ADC2_CH7 DAC_1",
            candidates,
        )
        == "ADC2_CH7/DAC_1"
    )


def test_schematic_no_connect_marker_is_a_typed_canonical_token() -> None:
    context = schematic.SchematicOcrRepairContext(False, frozenset(), frozenset(), {})

    assert (
        schematic.schematic_row_supplement_display_token(
            "x",
            context,
            frozenset(),
            {},
        )
        == "X"
    )
    assert schematic.classify_schematic_token_type("X") == "no_connect"
    assert schematic.schematic_supplement_token_max_per_token("no_connect") == 48


def test_schematic_no_connect_markers_require_repeated_cross_source_evidence() -> None:
    entry = schematic.SchematicSupplementEntry("X", "x", token_type="no_connect", source="a")
    supported = [
        entry,
        replace(entry, source="b"),
        replace(entry, source="a"),
        replace(entry, source="b"),
    ]

    assert schematic.filter_unsupported_schematic_no_connect_entries([entry]) == []
    assert schematic.filter_unsupported_schematic_no_connect_entries(supported) == supported


def test_clean_tiled_candidate_becomes_sparse_schematic_fusion_base() -> None:
    page = cast(
        Any,
        SimpleNamespace(
            chars=(),
            get_page_profile=lambda: SimpleNamespace(
                recommended_strategy="vector_or_table",
                has_path_ops=True,
            ),
        ),
    )
    current = " ".join(f"GPIO{index % 40}" for index in range(180))
    tiled = " ".join(f"GPIO{index % 80}" for index in range(400))
    candidate = OcrCandidate("rendered_page_400dpi_tiled", OcrTextResult(tiled, 67))
    result = OcrPageTextResult(tiled, candidate, (candidate,))
    classification = SimpleNamespace(kind="schematic")

    assert coordinator.sparse_schematic_tiled_candidate_should_be_fusion_base(
        page,
        current,
        result,
        classification,
    )


def test_structured_application_form_preserves_material_raw_ocr() -> None:
    fields = " ".join(f"{index}. Field value" for index in range(1, 25))
    raw_text = fields + " field value " * 190
    rendered_text = fields + " field value " * 120

    assert coordinator.structured_application_form_should_preserve_raw_ocr(
        raw_text,
        rendered_text,
        confidence=77,
    )
    assert not coordinator.structured_application_form_should_preserve_raw_ocr(
        raw_text,
        rendered_text,
        confidence=60,
    )


def test_dense_table_cleanup_preserves_dot_runs_outside_transcript_indexes() -> None:
    text = "MASS PROPERTIES\n....................\n0.001 0.002"

    assert coordinator.remove_dense_table_ocr_artifact_tokens(text) == text


def test_dense_table_cleanup_removes_ocr_dot_leaders_from_table_of_contents() -> None:
    chapters = [f"CHAPTER {index}: WEN HO LEE ee ee . . . {index}" for index in range(1, 5)]
    chapters[-1] = "CHAPTER 4: SYLVIA LEB AND WEN HO LBS ee ee . . . 4"
    text = "OVERVIEW OF TABLE OF CONTENTS\n" + "\n".join(chapters)

    assert coordinator.remove_dense_table_ocr_artifact_tokens(text).splitlines() == [
        "OVERVIEW OF TABLE OF CONTENTS",
        "CHAPTER 1: WEN HO LEE 1",
        "CHAPTER 2: WEN HO LEE 2",
        "CHAPTER 3: WEN HO LEE 3",
        "CHAPTER 4: SYLVIA LEE AND WEN HO LBS 4",
    ]


def test_toc_token_repair_uses_document_local_frequency() -> None:
    text = "CODE CODE CODE CODB OTHER"

    assert coordinator.repair_repeated_toc_token_variants(text) == "CODE CODE CODE CODE OTHER"


def test_dense_table_cleanup_prunes_systematic_contained_fragments() -> None:
    complete = [f"entry {index} code {index}:4 value" for index in range(10)]
    fragments = [f"code {index}:4" for index in range(10)]

    cleaned = coordinator.precision_prune_redundant_dense_table_text(
        "\n".join([*complete, *fragments])
    )

    assert cleaned.splitlines() == complete


def test_dense_table_cleanup_preserves_isolated_contained_line() -> None:
    lines = [f"entry {index} code {index}:4 value" for index in range(12)]
    lines.append("code 1:4")

    assert coordinator.precision_prune_redundant_dense_table_text("\n".join(lines)) == "\n".join(
        lines
    )


def test_dense_table_cleanup_preserves_repeated_numeric_measurement_rows() -> None:
    complete = [f"station {index} 100 200 300" for index in range(10)]
    fragments = [f"100 200 {index}" for index in range(10)]

    text = "\n".join([*complete, *fragments])

    assert coordinator.precision_prune_redundant_dense_table_text(text) == text


def test_dense_table_currency_repair_normalizes_symbol_shaped_prefixes() -> None:
    text = "Clerk #1620\nAttorney G2600\nWatchman Hl440\nReference G260\nPlain 2600"

    assert coordinator.repair_dense_table_currency_signs(text) == (
        "Clerk $1620\nAttorney $2600\nWatchman $1440\nReference G260\nPlain 2600"
    )


def test_collapsed_numeric_table_repair_normalizes_systematic_decimal_commas() -> None:
    numeric_rows = " ".join(
        f"{index},1 {index},2 12'195 ' )6237 58)o6 25.o 3oO )oS 12·8 I l z.5 : "
        "CSH LH HA55 C5M .47 .4 I 934 II 356"
        for index in range(80)
    )

    repaired = coordinator.repair_dense_table_decimal_separators(numeric_rows)

    assert "," not in repaired
    assert "'" not in repaired
    assert "79.1 79.2" in repaired
    assert "12495" in repaired
    assert "36237 583.6 25.0" in repaired
    assert "3.0 3.5" in repaired
    assert "12.8" in repaired
    assert "1 1 2.5" in repaired
    assert "2.5" in repaired
    assert ":" not in repaired
    assert "CSM LM MASS CSM" in repaired
    assert "-47 .4" in repaired
    assert "1934 11356" in repaired


def test_decimal_repair_preserves_commas_in_multiline_prose() -> None:
    prose = "\n".join(f"Item {index}, with description, owner, and status" for index in range(20))

    assert coordinator.repair_dense_table_decimal_separators(prose) == prose


def test_collapsed_numeric_scan_noise_prunes_systematic_singletons_and_speckles() -> None:
    ascii_table = "DESCRIPTION " + "FIELD " * 30 + " ".join(str(index) for index in range(80))
    noisy = ascii_table + " ⠾ 䥪 [ ] [ ] [ ] . ... r r r r r z z z z z n n n n n t t t t t"

    repaired = coordinator.prune_collapsed_numeric_table_scan_noise(noisy)

    assert "⠾" not in repaired
    assert "䥪" not in repaired
    assert "[" not in repaired
    assert "." not in repaired.split()
    assert "..." in repaired.split()
    assert repaired.split().count("r") == 2
    assert repaired.split().count("z") == 2
    assert repaired.split().count("n") == 2
    assert repaired.split().count("t") == 2


def test_repeated_form_blank_repair_restores_unfilled_fields() -> None:
    lines = [f"{letter}. Field description" for letter in "ABCDEFGHIJKLMNO"]
    lines[0] += " _"
    lines[1] += " ___"
    lines[2] += " 0.034"
    lines.extend(["Maximum", "Minimum"])

    repaired = coordinator.repair_repeated_form_blank_markers("\n".join(lines))

    assert "A. Field description _" in repaired
    assert "B. Field description ___" in repaired
    assert "C. Field description 0.034 ___" not in repaired
    assert repaired.count("___") == 16


def test_repeated_form_blank_repair_ignores_short_lettered_list() -> None:
    text = "A. First _\nB. Second _\nC. Third"

    assert coordinator.repair_repeated_form_blank_markers(text) == text


def test_archival_letter_list_repair_recovers_sequence_markers() -> None:
    text = "\n".join(
        [
            "A letter from Alpha accepting membership.",
            "A letter from Beta accepting membership.",
            "& letter from Gamma accepting membership.",
            "4 letter from Delta accepting membership.",
        ]
    )

    repaired = coordinator.repair_repeated_archival_letter_list_markers(text)

    assert repaired.splitlines() == [
        "(a) A letter from Alpha accepting membership.",
        "(b) A letter from Beta accepting membership.",
        "(c) A letter from Gamma accepting membership.",
        "(d) A letter from Delta accepting membership.",
    ]


def test_archival_letter_list_repair_ignores_isolated_phrase() -> None:
    text = "A letter from Alpha accepting membership."

    assert coordinator.repair_repeated_archival_letter_list_markers(text) == text


def test_archival_letter_list_term_repair_uses_repeated_phrase_consensus() -> None:
    text = "\n".join(
        [
            "(a) A letter from Alpha accopting membership and enclosing executed contracte.",
            "(b) A letter from Beta sccepting membership and enelowing exeeuted contracts.",
            "(c) A letter from Gamma accepting membership and enclosing executed contracts.",
            "(d) A letter from Delta aecepting mewbership and onelosing executed contracts.",
        ]
    )

    repaired = coordinator.repair_repeated_archival_letter_list_terms(text)

    assert "accopting" not in repaired
    assert "enelowing" not in repaired
    assert "exeeuted" not in repaired
    assert "mewbership" not in repaired
    assert repaired.count("accepting") == 4


def test_document_local_word_consensus_repairs_rare_single_character_variants() -> None:
    text = (
        "Their Foundation received their contracts from the Foundation. "
        "Thelr Foundetion retained another copy."
    )

    repaired = coordinator.repair_document_local_repeated_word_variants(text)

    assert repaired == (
        "Their Foundation received their contracts from the Foundation. "
        "Their Foundation retained another copy."
    )


def test_document_local_word_consensus_preserves_common_word_variants() -> None:
    text = "There were three options, and there were three outcomes."

    assert coordinator.repair_document_local_repeated_word_variants(text) == text


def test_native_url_supplement_restores_omitted_well_formed_line() -> None:
    text = "Scanned report body"
    native = "Noisy native body\nSource: https://archive.example.org/docs/record-42"

    assert coordinator.supplement_well_formed_native_url_lines(text, native) == (
        "Scanned report body\nSource: https://archive.example.org/docs/record-42"
    )


def test_native_url_supplement_does_not_duplicate_existing_url() -> None:
    text = "Report\nhttps://example.org/already-present"
    native = "Source: https://archive.example.org/docs/record-42"

    assert coordinator.supplement_well_formed_native_url_lines(text, native) == text


def test_native_url_supplement_rejects_malformed_native_line() -> None:
    text = "Scanned report body"
    native = "Source: https://localhost/document"

    assert coordinator.supplement_well_formed_native_url_lines(text, native) == text


def test_group_insurance_coverage_election_repair_recovers_checkboxes() -> None:
    text = "LifefAD&D Yes No Dependent Life § Yes {1 NoLTO il Yes NoSTD 2: Yes i. No"

    assert coordinator.repair_group_insurance_coverage_election_line(text) == (
        "Life/AD&D [] Yes [] No Dependent Life [] Yes [] No LTD [] Yes [] No STD [] Yes [] No"
    )


def test_chart_row_repair_joins_split_first_numeric_value() -> None:
    text = "2003 3.007 741 60,420\n2004 935 592 12,526\nHeader 3.007 741"

    assert text_analysis.repair_year_prefixed_chart_numeric_rows(text) == (
        "2003 3,007,741 60,420\n2004 935,592 12,526\nHeader 3.007 741"
    )


def test_chart_footnote_repair_normalizes_quote_shaped_stars() -> None:
    rows = [f"{year} 1000" for year in range(2002, 2012)]
    rows[6] = '2008 251079" 9267*"'
    rows[8] = '2010 1725*"*'
    text = "\n".join([*rows, "* note", "*** note"])

    repaired = coordinator.repair_chart_footnote_marker_confusions(text)

    assert "2008 251079* 9267**" in repaired
    assert "2010 1725***" in repaired

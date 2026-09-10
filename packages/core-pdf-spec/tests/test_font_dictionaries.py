"""Font dictionary ownership and optional-entry semantics."""

from __future__ import annotations

import pickle

import pytest

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_09_fonts import dictionaries, widths
from core_pdf_spec.s_09_fonts.dictionaries import prepare_font_program_inputs
from core_pdf_spec.s_09_fonts.widths import parse_font_widths


@pytest.mark.parametrize("font", [{}, {"FontDescriptor": None}, {"FontDescriptor": {}}])
def test_optional_descriptor_has_the_same_absent_and_null_semantics(
    font: dict[str, object],
) -> None:
    # ISO 32000-1, 7.3.9: null dictionary entries are equivalent to omitted entries.
    inputs = prepare_font_program_inputs(font)
    assert (inputs.font_file, inputs.font_file2, inputs.font_file3) == (None, None, None)
    metrics = parse_font_widths(font, "Type1")
    assert metrics.default_width == 0.0
    assert metrics.default_width_explicit is False


@pytest.mark.parametrize("descriptor", [42, [], "descriptor"])
def test_font_descriptor_shape_is_shared_by_widths_and_programs(descriptor: object) -> None:
    font = {"FontDescriptor": descriptor}
    with pytest.raises(ValueError, match="descriptor"):
        prepare_font_program_inputs(font)
    with pytest.raises(ValueError, match="descriptor"):
        parse_font_widths(font, "Type1")


def test_explicit_missing_width_zero_is_retained() -> None:
    metrics = parse_font_widths({"FontDescriptor": {"MissingWidth": 0}}, "Type1")
    assert metrics.default_width == 0.0
    assert metrics.default_width_explicit is True


def test_font_program_stream_selection_remains_lazy_and_uses_its_own_descriptor() -> None:
    def unexpected_decode(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("Selecting font program inputs must not decode streams")

    original = {
        key: PdfStream(decoder=unexpected_decode) for key in ("FontFile", "FontFile2", "FontFile3")
    }
    descendant = {key: PdfStream(decoder=unexpected_decode) for key in original}
    child = {"Subtype": "CIDFontType2", "FontDescriptor": descendant}
    font = {"Subtype": "Type0", "FontDescriptor": original, "DescendantFonts": [child]}
    inputs = prepare_font_program_inputs(font)
    assert inputs.descendant is child
    assert (inputs.original_subtype, inputs.subtype) == ("Type0", "CIDFontType2")
    assert inputs.font_file is original["FontFile"]
    assert inputs.font_file2 is descendant["FontFile2"]
    assert inputs.font_file3 is descendant["FontFile3"]

    simple_inputs = prepare_font_program_inputs({"FontDescriptor": original})
    assert simple_inputs.font_file is original["FontFile"]
    assert simple_inputs.font_file2 is original["FontFile2"]
    assert simple_inputs.font_file3 is original["FontFile3"]


def test_program_validation_checks_original_stream_before_descendant_descriptor() -> None:
    font = {"FontDescriptor": {"FontFile": 42}, "DescendantFonts": [{"FontDescriptor": 42}]}
    with pytest.raises(ValueError, match="stream: FontFile"):
        prepare_font_program_inputs(font)
    font["FontDescriptor"] = None
    with pytest.raises(ValueError, match="descriptor"):
        prepare_font_program_inputs(font)


def test_descendant_export_keeps_the_old_import_and_pickle_path() -> None:
    assert widths.get_descendant is dictionaries.get_descendant
    assert "get_descendant" in widths.__all__
    assert "get_descendant" in dictionaries.__all__
    assert pickle.loads(pickle.dumps(dictionaries.get_descendant)) is dictionaries.get_descendant
    # Protocol 0 GLOBAL reference emitted by the former defining module.
    assert (
        pickle.loads(b"ccore_pdf_spec.s_09_fonts.widths\nget_descendant\n.")
        is dictionaries.get_descendant
    )
    descendant: dict[str, object] = {"Subtype": "CIDFontType2"}
    assert dictionaries.get_descendant({"DescendantFonts": [descendant]}) is descendant

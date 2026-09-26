from types import SimpleNamespace

import pytest

from core_pdf.impl.capture_recording import TextState
from core_pdf.impl.capture_tolerant_state import (
    FontCompanionsCache,
    font_companions,
    font_signature,
)
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.types import PdfReference


@pytest.fixture
def document():
    resolver = ObjectResolver(b"", {})
    resolver.objects[key_for(1)] = {"Type": "Font", "Subtype": "Type1", "BaseFont": "Helvetica"}
    resolver.objects[key_for(2)] = {
        "Type": "Font",
        "Subtype": "Type1",
        "BaseFont": "ABCDEF+Helvetica",
    }
    resolver.objects[key_for(3, 4)] = {
        "Type": "Font",
        "Subtype": "Type1",
        "BaseFont": "GHIJKL+Helvetica",
    }
    instance = SimpleNamespace(resolver=resolver, resolve=resolver.resolve, font_decoders={})
    yield instance
    resolver.close()


def select(state, resources, name="F"):
    state.resources = resources
    state.execute_operation("Tf", (name, 12), 0)
    return state.get_decoder()


@pytest.mark.parametrize("reverse", [False, True])
def test_companion_groups_ignore_subset_tags_and_sort_indirect_identities(document, reverse):
    entries = [
        ("third", PdfReference(3, 4)),
        ("second", PdfReference(2)),
        ("first", PdfReference(1)),
    ]
    fonts: dict[str, object] = dict(reversed(entries) if reverse else entries)
    cache: FontCompanionsCache = {}
    grouped = font_companions(fonts, document.resolve, cache)
    assert grouped == {"Helvetica": ((1, 0), (2, 0), (3, 4))}
    assert cache[id(fonts)][0] is fonts
    assert font_companions(fonts, document.resolve, cache) is grouped


@pytest.mark.parametrize("font", [None, {}, {"BaseFont": None}, {"BaseFont": ""}])
def test_unresolved_or_unnamed_siblings_do_not_create_companion_groups(document, font):
    document.resolver.objects[key_for(7)] = font
    assert font_companions({"F": PdfReference(7)}, document.resolve, {}) == {}


@pytest.mark.parametrize(
    ("resources", "font"),
    [
        (None, {}),
        ({}, None),
        ({}, {}),
        ({"Font": []}, {}),
        ({"Font": {"F": {}}}, {}),
        ({"Font": {"F": None}}, {}),
    ],
)
def test_nonindirect_or_malformed_resources_disable_document_sharing(document, resources, font):
    assert font_signature(PdfReference(1), font, resources, document.resolve, {}) is None


def test_unnamed_font_signature_retains_reference_without_companions(document):
    signature = font_signature(PdfReference(8, 2), {}, {"Font": {}}, document.resolve, {})
    assert signature == (8, 2, ())


@pytest.mark.parametrize("shared_dictionary", [False, True])
def test_separate_captures_share_font_decoders_for_equivalent_sibling_sets(
    document, shared_dictionary
):
    fonts = {"F": PdfReference(1), "Sibling": PdfReference(2)}
    first_resources = {"Font": fonts}
    second_resources = {"Font": fonts if shared_dictionary else dict(reversed(list(fonts.items())))}
    first, second = TextState(document), TextState(document)
    decoder = select(first, first_resources)
    assert select(second, second_resources) is decoder
    assert second.graphics.decoder_resources is second_resources
    assert len(document.font_decoders) == 1


@pytest.mark.parametrize("sibling", [PdfReference(3, 4), {"BaseFont": "Helvetica"}])
def test_changed_or_direct_siblings_prevent_decoder_reuse_across_captures(document, sibling):
    first, second = TextState(document), TextState(document)
    original = select(first, {"Font": {"F": PdfReference(1), "Sibling": PdfReference(2)}})
    changed = select(second, {"Font": {"F": PdfReference(1), "Sibling": sibling}})
    assert changed is not original


def test_alias_selection_reuses_the_capture_owned_decoder(document):
    state = TextState(document)
    resources = {"Font": {"F": PdfReference(1), "Alias": PdfReference(1)}}
    decoder = select(state, resources)
    assert state.get_decoder() is decoder
    assert select(state, resources, "Alias") is decoder
    assert select(state, resources, "F") is decoder
    assert len(state.capture_font_decoders[(1, 0)]) == 1


@pytest.mark.parametrize("wrapped", [False, True])
def test_direct_fonts_reuse_within_scope_but_never_across_captures(document, wrapped):
    font = {"Type": "Font", "Subtype": "Type1", "BaseFont": "Helvetica"}
    value = PdfStream(font) if wrapped else font
    resources = {"Font": {"F": value, "Alias": value}}
    first, second = TextState(document), TextState(document)
    decoder = select(first, resources)
    assert select(first, resources, "Alias") is decoder
    assert select(second, resources) is not decoder
    assert document.font_decoders == {}


@pytest.mark.parametrize("value", [None, 42, "not a font"])
def test_invalid_font_objects_fall_back_to_a_usable_decoder(document, value):
    document.resolver.objects[key_for(9)] = value
    state = TextState(document)
    decoder = select(state, {"Font": {"F": PdfReference(9)}})
    assert decoder is state.graphics.current_decoder
    assert decoder.decode_glyphs(b"A")


@pytest.mark.parametrize("resources", [{}, {"Font": {}}, {"Font": {"Other": {}}}])
def test_missing_font_selection_uses_the_fallback_without_document_cache_entries(
    document, resources
):
    state = TextState(document)
    decoder = select(state, resources)
    assert decoder.decode_glyphs(b"A")
    assert document.font_decoders == {}

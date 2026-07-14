from __future__ import annotations

from collections.abc import Sequence
from math import isclose

from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument
from core_pdf.impl.engine.spec.s_07_document.metadata import resolve_info_metadata
from core_pdf.impl.engine.spec.s_07_document.navigation import NavigationMixin
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_07_syntax.primitives import PdfName
from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder


def test_leading_dot_number_is_passed_to_operator() -> None:
    received: list[object] = []

    def move_to(operands: Sequence[object], _depth: int) -> None:
        received.extend(operands)

    fast_handlers: list[object] = [None] * 65536
    fast_handlers[ord("m") << 8] = move_to
    PdfLexer(b".5 1 m").dispatch_operations({"m": move_to}, fast_handlers, 0)

    assert received == [0.5, 1]


def test_cid_fast_path_applies_word_spacing() -> None:
    decoder = object.__new__(FontDecoder)
    decoder.is_cid_font = True
    decoder.to_unicode = None
    decoder.cmap = None
    decoder.fast_widths_cid = [500.0] * 65536
    decoder.is_vertical = False

    advance = decoder.text_advance_vector(
        b"\x00 \x00A", font_size=10.0, char_space=0.0, word_space=2.0, horizontal_scale=1.0
    )

    assert isclose(advance[0], 0.12)
    assert advance[1] == 0.0


def test_unsupported_operator_does_not_leak_operands() -> None:
    received: list[object] = []

    def move_to(operands: Sequence[object], _depth: int) -> None:
        received.extend(operands)

    fast_handlers: list[object] = [None] * 65536
    fast_handlers[ord("m") << 8] = move_to
    lexer = PdfLexer(b"99 UNKNOWN 1 2 m")

    lexer.dispatch_operations({"m": move_to}, fast_handlers, 0)

    assert received == [1, 2]


def test_trapped_info_value_accepts_pdf_name() -> None:
    info = {"Trapped": PdfName.of("False")}

    result = resolve_info_metadata(_TestResolver(), {"Info": info})

    assert result["Trapped"] == PdfName.of("False")


class _TestResolver:
    def resolve(self, value: object) -> object:
        return value

    def resolve_dict(self, value: object) -> dict[object, object] | None:
        return self._copy(value) if isinstance(value, dict) else None

    def _copy(self, value: object) -> object:
        if isinstance(value, dict):
            return {key: self._copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._copy(item) for item in value]
        return value

    def resolve_name(self, value: object) -> str | None:
        if isinstance(value, PdfName):
            return value.value
        return value if isinstance(value, str) else None

    def resolve_str(self, value: object) -> str | None:
        return value if isinstance(value, str) else None

    def resolve_int(self, value: object, default: int | None = None) -> int | None:
        return value if isinstance(value, int) else default


class _NavigationDocument(NavigationMixin):
    def __init__(self, page: dict[object, object]) -> None:
        self.page = page
        self.resolver = _TestResolver()

    def page_index_for(self, page_obj: object) -> int | None:
        return 0 if page_obj is self.page else None


def test_outline_links_are_resolved_shallowly() -> None:
    page: dict[object, object] = {}
    sibling = {"Title": "Sibling"}
    child = {"Title": "Child", "Dest": [page, PdfName.of("Fit")], "Next": sibling}
    first = {"Title": "First", "First": child}

    result = NavigationMixin.walk_outlines(_NavigationDocument(page), first, 0)

    assert [item.title for item in result] == ["First", "Child", "Sibling"]
    assert result[1].page_index == 0


def test_structure_root_keeps_catalog_object_identity() -> None:
    root: dict[str, object] = {}
    document = object.__new__(PdfDocument)
    document.resolver = _TestResolver()
    document.catalog_cache = {"StructTreeRoot": root}
    document.structure_cache = None
    document.structure_root_cache = None

    structure = document.structure

    assert structure is not None
    assert structure.props is root

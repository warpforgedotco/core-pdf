from __future__ import annotations

import mmap
import threading
from collections.abc import Sequence
from io import BytesIO
from math import isclose
from operator import eq
from pathlib import Path
from typing import Any, cast

import pytest

from core_pdf import PdfDocument as PublicPdfDocument
from core_pdf import PdfSourceError
from core_pdf.impl.engine.spec.s_07_content.operations import dispatch_operations
from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument
from core_pdf.impl.engine.spec.s_07_document.document_labels import (
    format_alpha,
    format_page_label,
)
from core_pdf.impl.engine.spec.s_07_document.metadata import (
    resolve_info_metadata,
)
from core_pdf.impl.engine.spec.s_07_objects.resolver_values import PdfValueResolver
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.primitives import MISSING, PdfName
from core_pdf.impl.types import PdfDict


def test_leading_dot_number_is_passed_to_operator() -> None:
    received: list[object] = []

    def move_to(operands: Sequence[object], internal_depth: int) -> None:
        received.extend(operands)

    fast_handlers: list[object] = [None] * 65536
    fast_handlers[ord("m") << 8] = move_to
    cast(Any, dispatch_operations)(
        PdfLexer(b".5 1 m"), {"m": move_to}, None, fast_handlers, {}, None, 0
    )

    assert received == [0.5, 1]


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        (b"0.123", 0.123),
        (b"-0.123", -0.123),
        (b"3.728", 3.728),
        (b"7.6993", 7.6993),
        (b"-12.345", -12.345),
        (b"123.45", 123.45),
    ],
)
def test_three_decimal_number_is_passed_to_operator(token: bytes, expected: float) -> None:
    received: list[object] = []

    def move_to(operands: Sequence[object], internal_depth: int) -> None:
        received.extend(operands)

    fast_handlers: list[object] = [None] * 65536
    fast_handlers[ord("m") << 8] = move_to
    cast(Any, dispatch_operations)(
        PdfLexer(token + b" 1 m"), {"m": move_to}, None, fast_handlers, {}, None, 0
    )

    assert received == [expected, 1]


def test_multibyte_cid_code_does_not_apply_word_spacing() -> None:
    decoder = FontDecoder(
        {
            "Subtype": "Type0",
            "Encoding": "Identity-H",
            "DescendantFonts": [{"Subtype": "CIDFontType0", "DW": 500}],
        }
    )

    advance = decoder.text_advance_vector(
        b"\x00 \x00A", font_size=10.0, char_space=0.0, word_space=2.0, horizontal_scale=1.0
    )

    assert isclose(advance[0], 0.1)
    assert advance[1] == 0.0


def test_unsupported_operator_does_not_leak_operands() -> None:
    received: list[object] = []

    def move_to(operands: Sequence[object], internal_depth: int) -> None:
        received.extend(operands)

    fast_handlers: list[object] = [None] * 65536
    fast_handlers[ord("m") << 8] = move_to
    lexer = PdfLexer(b"99 UNKNOWN 1 2 m")

    cast(Any, dispatch_operations)(lexer, {"m": move_to}, None, fast_handlers, {}, None, 0)

    assert received == [1, 2]


@pytest.mark.parametrize("invalid_operator", [b"12foo", b".x", b"+bad", b".", b"+", b"-"])
def test_number_shaped_unsupported_operator_does_not_leak_operands(
    invalid_operator: bytes,
) -> None:
    received: list[object] = []

    def move_to(operands: Sequence[object], internal_depth: int) -> None:
        received.extend(operands)

    fast_handlers: list[object] = [None] * 65536
    fast_handlers[ord("m") << 8] = move_to
    lexer = PdfLexer(b"99 " + invalid_operator + b" 1 2 m")

    cast(Any, dispatch_operations)(lexer, {"m": move_to}, None, fast_handlers, {}, None, 0)

    assert received == [1, 2]


def test_pdf_name_bytes_equality_supports_mapping_lookup() -> None:
    name = PdfName.of(b"N\xe1me")
    mapping = {name: "value"}
    cross_type_mapping = cast(dict[Any, str], mapping)

    assert name == b"N\xe1me"
    assert eq(b"N\xe1me", name)
    assert hash(name) == hash(b"N\xe1me")
    assert cross_type_mapping[b"N\xe1me"] == "value"


@pytest.mark.parametrize(
    ("number", "expected"),
    [(1, "a"), (26, "z"), (27, "aa"), (28, "bb"), (52, "zz"), (53, "aaa")],
)
def test_page_label_alphabetic_sequence(number: int, expected: str) -> None:
    assert format_alpha(number) == expected


def test_page_label_without_style_is_prefix_only() -> None:
    assert format_page_label({"P": b"Appendix-"}, 17, lambda value: value) == "Appendix-"


class internal_PageLabelDocument(PdfDocument):
    def __init__(self, *, recovered: bool) -> None:
        self.internal_cache_lock = threading.RLock()
        self.page_labels_cache = None
        self.xref_was_recovered = recovered
        self.page_tree_was_recovered = False
        self.page_dicts_cache = [{}, {}, {}, {}]
        self.pages_cache = cast(Any, [None, None, None, None])
        self.internal_catalog = {"PageLabels": {"Nums": [2, {"S": PdfName.of("D")}]}}

    def catalog(self) -> PdfDict:
        return cast(PdfDict, self.internal_catalog)

    def resolve(self, ref: object) -> object:
        return ref


def test_page_labels_require_page_zero_range() -> None:
    with pytest.raises(ValueError, match="page index 0"):
        internal_PageLabelDocument(recovered=False).build_page_labels()


def test_recovered_page_labels_fill_missing_initial_range() -> None:
    assert internal_PageLabelDocument(recovered=True).build_page_labels() == ["", "", "1", "2"]


def test_explicit_crypt_metadata_stream_uses_document_security_handler() -> None:
    fixture = Path(__file__).parent / "fixtures" / "SCORE-Bench" / "src" / "g-325a.pdf"

    with PublicPdfDocument(fixture) as document:
        xmp = document.get_metadata()["xmp"]

    assert xmp is not None
    assert "parse_error" not in xmp
    assert xmp["tag"].endswith("xmpmeta")


def simple_pdf_fixture() -> Path:
    return Path(__file__).parent / "fixtures" / "SCORE-Bench" / "src" / "g-325a.pdf"


def test_nested_form_rebinds_same_named_font_resource() -> None:
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "korean_power_system_challenges-p003.pdf"
    )

    with PublicPdfDocument(fixture) as document:
        result = cast(Any, document.pages[0]).extract()
        text = "\n".join(line.text for block in result.blocks for line in block.lines)

    assert "This document was prepared as an account of work" in text
    assert "5Iis document was prepared as an account of worL" not in text


def test_document_close_releases_owned_path_resources() -> None:
    document: PdfDocument[Any] = PdfDocument(simple_pdf_fixture())
    mapping = document.raw_data
    file_handle = document.file_handle

    assert isinstance(mapping, mmap.mmap)
    assert file_handle is not None
    document.close()

    assert document.closed
    assert mapping.closed
    assert file_handle.closed
    assert document.raw_data == b""
    document.close()


def test_document_close_preserves_caller_owned_reader() -> None:
    reader = BytesIO(simple_pdf_fixture().read_bytes())
    document: PdfDocument[Any] = PdfDocument(reader)

    document.close()

    assert not reader.closed
    assert document.closed


def test_document_close_defers_unmap_for_external_view() -> None:
    document: PdfDocument[Any] = PdfDocument(simple_pdf_fixture())
    mapping = document.raw_data
    assert isinstance(mapping, mmap.mmap)
    external_view = memoryview(mapping)

    document.close()

    assert document.closed
    assert not mapping.closed
    external_view.release()
    mapping.close()
    assert mapping.closed


def test_empty_path_closes_opened_file(tmp_path: Path) -> None:
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    document = object.__new__(PdfDocument)
    document.file_handle = None

    with pytest.raises(PdfSourceError, match="empty"):
        document.load_data(empty_pdf)

    assert document.file_handle is None


class internal_FailingDocument(PdfDocument):
    mapping_at_failure: mmap.mmap | None = None
    handle_at_failure: Any = None

    def scan_xref(self) -> None:
        type(self).mapping_at_failure = cast(mmap.mmap, self.raw_data)
        type(self).handle_at_failure = self.file_handle
        raise RuntimeError("scan failed")


def test_construction_failure_releases_acquired_resources() -> None:
    with pytest.raises(RuntimeError, match="scan failed"):
        internal_FailingDocument(simple_pdf_fixture())

    mapping = internal_FailingDocument.mapping_at_failure
    handle = internal_FailingDocument.handle_at_failure
    assert mapping is not None
    assert mapping.closed
    assert handle is not None
    assert handle.closed


def test_trapped_info_value_accepts_pdf_name() -> None:
    info = {"Trapped": PdfName.of("False")}

    result = resolve_info_metadata(
        cast(PdfValueResolver, internal_TestResolver()), cast(PdfDict, {"Info": info})
    )

    assert result["Trapped"] == PdfName.of("False")


class internal_TestResolver:
    def resolve(self, value: object) -> object:
        return value

    def resolve_dict(self, value: object) -> dict[object, object] | None:
        return (
            cast(dict[object, object], self.internal_copy(value))
            if isinstance(value, dict)
            else None
        )

    def internal_copy(self, value: object) -> object:
        if isinstance(value, dict):
            return {key: self.internal_copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.internal_copy(item) for item in value]
        return value

    def resolve_name(self, value: object) -> str | None:
        if isinstance(value, PdfName):
            return value.value
        return value if isinstance(value, str) else None

    def resolve_str(self, value: object) -> str | None:
        return value if isinstance(value, str) else None

    def resolve_int(self, value: object, default: int | None = None) -> int | None:
        return value if isinstance(value, int) else default


class internal_NavigationDocument(PdfDocument[Any]):
    def __init__(self, page: dict[object, object]) -> None:
        self.page = page
        self.resolver = cast(Any, internal_TestResolver())
        self.xref_was_recovered = False
        self.page_tree_was_recovered = False
        self.named_destinations_cache = None

    def page_index_for(self, page_obj: object) -> int | None:
        return 0 if page_obj is self.page else None


def test_outline_links_are_resolved_shallowly() -> None:
    page: dict[object, object] = {}
    sibling = {"Title": "Sibling"}
    child = {"Title": "Child", "Dest": [page, PdfName.of("Fit")], "Next": sibling}
    first = {"Title": "First", "First": child}

    result = PdfDocument.walk_outlines(internal_NavigationDocument(page), first, 0)

    assert [item.title for item in result] == ["First", "Child", "Sibling"]
    assert result[1].page_index == 0


def test_structure_root_keeps_catalog_object_identity() -> None:
    root: dict[str, object] = {}
    document: PdfDocument[Any] = object.__new__(PdfDocument)
    document.resolver = cast(Any, internal_TestResolver())
    document.catalog_cache = cast(PdfDict, {"StructTreeRoot": root})
    document.structure_cache = MISSING

    structure = document.structure

    assert structure is not None
    assert structure.props is root

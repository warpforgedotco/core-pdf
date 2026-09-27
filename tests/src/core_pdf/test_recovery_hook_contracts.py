import importlib
import inspect
from pathlib import Path
from types import MemberDescriptorType

import pytest

import core_pdf.impl
from core_pdf.impl.recovery_lexer import PdfLexer
from core_pdf.impl.recovery_policy import LENIENT, STRICT, RecoveryMode
from core_pdf.impl.recovery_xref import StrictXRefScanner, XRefScanner

ADDITIONS = {
    "PdfLexer": {
        "at_endstream",
        "copied_data",
        "find_keyword_candidate",
        "find_object_end",
        "find_stream_end",
        "mode",
        "object_scanner",
        "recover_stream_data",
        "scanner_args",
        "search_buffer",
    },
    "PdfObjectStream": {
        "body_lexer",
        "handle_object_error",
        "lexer_over_body",
        "scan_dictionary_at",
    },
    "ObjectResolver": {
        "adopt_parsed_objects",
        "array_at",
        "as_dict",
        "dict_at",
        "float_at",
        "int_at",
        "name_at",
        "recover_indirect_object",
        "resolve_name_like_value",
        "resolve_name_or_text",
        "resolve_or_none",
    },
    "XRefScanner": {
        "brute_force_scan",
        "find_nearby_sections",
        "lexer",
        "mode",
        "parse_xref_stream_salvage",
        "recover_object_stream_entries",
        "recover_section_at",
        "salvage_xref_stream_span",
    },
    "StrictXRefScanner": {"mode"},
}


def recovery_classes():
    impl = Path(core_pdf.impl.__file__).parent
    for path in sorted(impl.glob("recovery_*.py")):
        module = importlib.import_module(f"core_pdf.impl.{path.stem}")
        for cls in vars(module).values():
            if inspect.isclass(cls) and cls.__module__ == module.__name__:
                spec = next(
                    (
                        base
                        for base in cls.__mro__[1:]
                        if base.__module__.startswith("core_pdf_spec")
                    ),
                    None,
                )
                if spec is not None:
                    yield cls, spec


@pytest.mark.parametrize(
    ("cls", "spec"), list(recovery_classes()), ids=lambda value: value.__qualname__
)
def test_recovery_classes_only_override_spec_hooks(cls, spec):
    own = {
        name
        for name, value in vars(cls).items()
        if not (name.startswith("__") and name.endswith("__"))
        and not isinstance(value, MemberDescriptorType)
    }
    additions = {name for name in own if not hasattr(spec, name)}
    assert additions <= ADDITIONS.get(cls.__name__, set())


def test_recovery_modes_name_their_strictness():
    assert RecoveryMode.for_document(recovery_enabled=True) is RecoveryMode.TOLERANT
    assert RecoveryMode.for_document(recovery_enabled=False) is RecoveryMode.STRICT
    assert RecoveryMode.TOLERANT.malformed is LENIENT
    assert RecoveryMode.STRICT.malformed is STRICT
    assert XRefScanner.mode is RecoveryMode.TOLERANT
    assert not StrictXRefScanner.mode.objects


@pytest.mark.parametrize("objects", [True, False])
@pytest.mark.parametrize("dictionary_structure", [True, False])
def test_lexer_keyword_flags_select_a_mode(objects, dictionary_structure):
    lexer = PdfLexer(
        b"<< /A 1 >>",
        recover_malformed_objects=objects,
        recover_dictionary_structure=dictionary_structure,
    )
    try:
        assert lexer.mode == RecoveryMode.for_objects(objects, dictionary_structure)
        assert lexer.mode.objects is objects
        assert lexer.mode.dictionary_structure is dictionary_structure
    finally:
        lexer.close()

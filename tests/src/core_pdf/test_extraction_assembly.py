import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl._impl.extract.selection import internal_assemble_document
from core_pdf.impl._impl.output.model import Diagnostic, Page
from core_pdf.impl._impl.runtime.execution import ExtractionScope, internal_ExtractionCancelled


@pytest.mark.parametrize("numbers", [(), (3, 1, 3)])
def test_document_assembly_retains_selection_metadata_and_diagnostics(
    numbers: tuple[int, ...],
) -> None:
    pages = tuple(
        Page(number, diagnostics=(Diagnostic("example", str(number)),)) for number in numbers
    )
    extractions = tuple(
        SimpleNamespace(assembled_page=lambda context, page=page: page) for page in pages
    )
    document = SimpleNamespace(get_metadata=lambda: {"Title": "Selection"})
    result = internal_assemble_document(
        cast(Any, document), cast(Any, extractions), ExtractionScope()
    )
    assert result.pages == pages
    assert dict(result.metadata) == {"Title": "Selection"}
    assert result.diagnostics == tuple(d for page in pages for d in page.diagnostics)


def test_document_assembly_checks_cancellation_between_pages() -> None:
    visited: list[int] = []

    def assemble(context: ExtractionScope) -> Page:
        visited.append(1)
        return Page(1)

    extractions = (SimpleNamespace(assembled_page=assemble),) * 2
    context = ExtractionScope(cancelled=lambda: bool(visited))
    with pytest.raises(internal_ExtractionCancelled):
        internal_assemble_document(cast(Any, None), cast(Any, extractions), context)
    assert visited == [1]


def test_native_extraction_keeps_companions_unloaded(text_pdf_bytes: bytes) -> None:
    script = """
import sys
import core_pdf
for name in core_pdf.__all__:
    getattr(core_pdf, name)
with core_pdf.PdfDocument(sys.stdin.buffer.read()) as document:
    assert 'Hello maintenance' in document.extract().text
assert not any(name.split('.')[0] in {'core_pdf_ocr', 'core_pdf_validate'} for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], input=text_pdf_bytes, check=True)

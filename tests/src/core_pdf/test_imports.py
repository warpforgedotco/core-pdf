# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Sequence


def test_pdf_document_import_defers_page_stack() -> None:
    script = textwrap.dedent(
        """
        import sys

        from core_pdf import PdfDocument
        from core_pdf.impl.engine.document import PdfDocument as DirectPdfDocument

        assert PdfDocument is DirectPdfDocument
        deferred_modules = (
            "core_pdf.impl.engine.page",
            "core_pdf.impl.engine.spec.s_07_document.page",
        )
        assert not [name for name in deferred_modules if name in sys.modules]
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)


def test_lazy_public_exports_preserve_the_root_api() -> None:
    from core_pdf import (
        PageSelection,
        PdfDocument,
        PdfError,
        PdfPage,
        PdfParseError,
        PdfRasterTooLargeError,
        PdfSourceError,
        PdfUnsupportedError,
    )

    assert PdfDocument.__name__ == "PdfDocument"
    assert PdfPage.__name__ == "PdfPage"
    assert PageSelection == (int | str | range | Sequence[int])
    assert issubclass(PdfParseError, PdfError)
    assert issubclass(PdfRasterTooLargeError, PdfError)
    assert issubclass(PdfSourceError, PdfError)
    assert issubclass(PdfUnsupportedError, PdfError)

from pathlib import Path

import pytest

from .support import call_pair, differential_pdfs, pdf_id

real_reader = pytest.importorskip("llama_index.readers.file").PDFReader
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("pdf_path", differential_pdfs("llamaindex"), ids=pdf_id)
def test_matches_real_reader_on_fixture_corpus(pdf_path: Path) -> None:
    from core_pdf.api.compat.llamaindex import load_data

    pair = call_pair(
        lambda: real_reader().load_data(file=pdf_path),
        lambda: load_data(pdf_path),
    )
    if pair is not None:
        assert [(document.text, document.metadata) for document in pair[1]] == [
            (document.text, document.metadata) for document in pair[0]
        ]

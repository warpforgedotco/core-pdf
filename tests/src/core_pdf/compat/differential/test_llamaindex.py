from pathlib import Path

import pytest

from .support import ALL_PDFS, call_pair


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_matches_real_reader_on_all_fixture_pdfs(pdf_path: Path) -> None:
    real_reader = pytest.importorskip("llama_index.readers.file").PDFReader
    from core_pdf.api.compat.llamaindex import load_data

    pair = call_pair(
        lambda: real_reader().load_data(file=pdf_path),
        lambda: load_data(pdf_path),
    )
    if pair is not None:
        assert [document.text for document in pair[1]] == [document.text for document in pair[0]]

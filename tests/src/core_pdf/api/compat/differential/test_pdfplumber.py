import builtins
import pickle
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from .support import FIXTURES_ROOT, call_pair, differential_pdfs, pdf_id, words

real_pdfplumber = pytest.importorskip("pdfplumber")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("pdf_path", differential_pdfs("pdfplumber"), ids=pdf_id)
def test_matches_real_library_on_fixture_corpus(pdf_path: Path) -> None:
    from core_pdf.api.compat import pdfplumber as compat_pdfplumber

    def snapshot(open_pdf: Any) -> tuple[tuple[str, list[dict[str, Any]], float, float], ...]:
        with open_pdf(pdf_path) as pdf:
            return tuple(
                (page.extract_text(), page.extract_words(), page.width, page.height)
                for page in pdf.pages
            )

    pair = call_pair(
        lambda: snapshot(real_pdfplumber.open), lambda: snapshot(compat_pdfplumber.open)
    )
    if pair is not None:
        expected_pages, actual_pages = pair
        assert len(actual_pages) == len(expected_pages)
        for actual, expected in zip(actual_pages, expected_pages, strict=True):
            assert actual[0] == expected[0]
            assert actual[1] == words(expected[1])
            assert actual[2:] == expected[2:]


@pytest.mark.parametrize("completion", ["close", "exhaust", "throw"])
def test_pdfminer_selection_preserves_lazy_file_lifetime(
    monkeypatch: pytest.MonkeyPatch, completion: str
) -> None:
    from core_pdf.api import compat
    from core_pdf.api.compat import pdfminer as compat_pdfminer

    real_extract_pages = pytest.importorskip("pdfminer.high_level").extract_pages
    pdf_path = FIXTURES_ROOT / "pdfplumber/tests/pdfs/pdffill-demo.pdf"
    assert compat.extract_pages is compat_pdfminer.extract_pages
    assert compat.LAParams is compat_pdfminer.LAParams
    assert compat_pdfminer.LTTextContainer is compat_pdfminer.LTTextBox

    def snapshot(extract_pages: Any) -> tuple[str, tuple[float, ...]]:
        handles: list[Any] = []
        original_open = builtins.open

        def track_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            handle = original_open(file, *args, **kwargs)
            if isinstance(file, (str, Path)) and Path(file) == pdf_path:
                handles.append(handle)
            return handle

        with monkeypatch.context() as patch:
            patch.setattr(builtins, "open", track_open)
            pages = extract_pages(pdf_path, page_numbers=(1,), maxpages=1)
            assert not handles
            with closing(pages):
                page = next(pages)
                assert handles
                assert all(not handle.closed for handle in handles)
                restored = pickle.loads(pickle.dumps(page))
                assert type(restored) is type(page)
                if extract_pages is compat_pdfminer.extract_pages:
                    assert type(restored).__module__ == compat_pdfminer.__name__
                    assert restored.pageid == 2
                if completion == "close":
                    pages.close()
                elif completion == "exhaust":
                    assert list(pages) == []
                else:
                    with pytest.raises(RuntimeError, match="abort extraction"):
                        pages.throw(RuntimeError("abort extraction"))
            assert all(handle.closed for handle in handles)
        return (
            "".join(item.get_text() for item in restored if hasattr(item, "get_text")),
            tuple(restored.bbox),
        )

    assert snapshot(compat_pdfminer.extract_pages) == snapshot(real_extract_pages)

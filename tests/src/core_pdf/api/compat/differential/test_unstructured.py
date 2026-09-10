from pathlib import Path

import pytest

from core_pdf.api.compat import unstructured as compat_unstructured

from .support import call_pair, differential_pdfs, pdf_id

# Import the facade first: its required model must be available before the
# reference tokenizer can attempt its automatic model installation.
real_partition = pytest.importorskip("unstructured.partition.pdf").partition_pdf
real_tokenize = pytest.importorskip("unstructured.nlp.tokenize")
pytestmark = pytest.mark.compat_differential


@pytest.fixture(scope="module", autouse=True)
def require_nlp_pipelines() -> None:
    # Dependency failures must happen outside call_pair(), which treats a
    # rejection on both sides as equivalent PDF behavior.
    text = "The library preserves sentence boundaries."
    tokens, sentences = compat_unstructured.internal_nlp_features(text)
    assert tokens
    assert sentences
    assert any(tag in compat_unstructured.internal_POS_VERB_TAGS for _token, tag in tokens)
    assert tokens == tuple(real_tokenize.pos_tag(text))
    assert sentences == tuple(real_tokenize.sent_tokenize(text))


@pytest.mark.parametrize("pdf_path", differential_pdfs("unstructured"), ids=pdf_id)
def test_matches_real_library_on_fixture_corpus(pdf_path: Path) -> None:
    pair = call_pair(
        lambda: real_partition(filename=str(pdf_path), strategy="fast"),
        lambda: compat_unstructured.partition_pdf(pdf_path),
    )
    if pair is None:
        return
    expected, actual = pair
    assert [(item.category, item.text) for item in actual] == [
        (item.category, item.text) for item in expected
    ]

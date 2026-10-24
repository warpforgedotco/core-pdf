import pickle
from pathlib import Path

import pytest

from .support import call_pair, differential_pdfs, pdf_id

# Skip before importing the facade: without the reference library its spaCy
# requirement is usually absent too, and the facade's import would fail first.
real_partition = pytest.importorskip("unstructured.partition.pdf").partition_pdf
real_tokenize = pytest.importorskip("unstructured.nlp.tokenize")

import core_pdf_compat.unstructured  # noqa: E402
from core_pdf_compat.unstructured import _classification  # noqa: E402

pytestmark = pytest.mark.compat_differential


@pytest.fixture(scope="module", autouse=True)
def require_nlp_pipelines() -> None:
    text = "The library preserves sentence boundaries."
    tokens, sentences = _classification.nlp_features(text)
    assert tokens
    assert sentences
    assert any(tag in _classification.POS_VERB_TAGS for _token, tag in tokens)
    assert tokens == tuple(real_tokenize.pos_tag(text))
    assert sentences == tuple(real_tokenize.sent_tokenize(text))


@pytest.mark.parametrize("pdf_path", differential_pdfs("unstructured"), ids=pdf_id)
def test_matches_real_library_on_fixture_corpus(pdf_path: Path) -> None:
    pair = call_pair(
        lambda: real_partition(filename=str(pdf_path), strategy="fast"),
        lambda: core_pdf_compat.unstructured.partition_pdf(pdf_path),
    )
    if pair is None:
        return
    expected, actual = pair
    assert [(item.category, item.text) for item in actual] == [
        (item.category, item.text) for item in expected
    ]
    restored = pickle.loads(pickle.dumps(actual))
    assert restored == actual
    for original, copy in zip(actual, restored, strict=True):
        assert type(copy) is type(original)
        assert type(copy).__module__ == core_pdf_compat.unstructured.__name__

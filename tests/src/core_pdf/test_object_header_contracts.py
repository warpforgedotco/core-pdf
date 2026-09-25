"""Xref offsets are checked against object headers as the old expression checked them."""

import gzip
import pickle
from pathlib import Path

from core_pdf.impl.document.document import object_headers_present

GOLDEN_PATH = (
    Path(__file__).parents[3] / "packages/core-pdf-cythonized/tests/xref_headers_golden.pkl.gz"
)


def test_every_golden_case_including_huge_object_numbers() -> None:
    golden = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))
    for case in golden:
        present = object_headers_present(case["data"], [case["key"]], [case["offset"]])
        assert present == [case["expected"]], case


def test_a_batch_answers_as_its_entries_do_alone() -> None:
    data = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n99999999999999999999 0 obj\nnull\nendobj\n"
    entries = [
        (1 << 16, 9),
        (1 << 16, 10),
        (99999999999999999999 << 16, data.index(b"9999")),
        (99999999999999999998 << 16, data.index(b"9999")),
        (2 << 16, 10**30),
    ]
    keys = [key for key, _ in entries]
    offsets = [offset for _, offset in entries]
    assert object_headers_present(data, keys, offsets) == [True, False, True, False, False]
    small = [index for index, key in enumerate(keys) if key < 1 << 63]
    assert object_headers_present(
        data, [keys[index] for index in small], [offsets[index] for index in small]
    ) == [True, False, False]
    assert [object_headers_present(data, [key], [offset])[0] for key, offset in entries] == [
        True,
        False,
        True,
        False,
        False,
    ]

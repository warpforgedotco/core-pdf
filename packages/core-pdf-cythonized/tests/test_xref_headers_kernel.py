# SPDX-License-Identifier: AGPL-3.0-only


import gzip
import pickle
from array import array
from pathlib import Path

import pytest

from core_pdf_cythonized import object_headers_match

GOLDEN_PATH = Path(__file__).parent / "xref_headers_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))
HUGE = 1 << 63


def state(case):
    number = case["key"] >> 16
    return object_headers_match(
        case["data"],
        array("q", (number if number < HUGE else -2,)),
        array("q", (case["key"] & 0xFFFF,)),
        array("q", (case["offset"] if 0 <= case["offset"] < len(case["data"]) else -1,)),
    )[0]


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 16444
    assert sum(case["expected"] for case in GOLDEN) == 5296
    synthetic = [case for case in GOLDEN if case["origin"] == "synthetic"]
    assert len(synthetic) == 39
    assert any(case["key"] >> 16 >= HUGE and case["expected"] for case in synthetic)


def test_each_case_answers_as_the_expression_did():
    for case in GOLDEN:
        got = state(case)
        if case["key"] >> 16 >= HUGE:
            assert got in ((2,) if case["expected"] else (0, 2)), case
        else:
            assert got == int(case["expected"]), case


def test_one_call_answers_every_entry_of_one_buffer():
    data = b"1 0 obj\n<<>>\nendobj\n22 3 obj 5\nendobj"
    numbers = array("q", (1, 22, 1, 22, 1))
    generations = array("q", (0, 3, 0, 3, 0))
    offsets = array("q", (0, 20, 20, 21, -1))
    assert object_headers_match(data, numbers, generations, offsets).tolist() == [1, 1, 0, 0, 0]


def test_the_arrays_must_pair_up():
    with pytest.raises(ValueError, match="differ in length"):
        object_headers_match(b"1 0 obj", array("q", (1,)), array("q", (0,)), array("q", ()))

# SPDX-License-Identifier: AGPL-3.0-only


import gzip
import pickle
from pathlib import Path

import pytest

from core_pdf_cythonized import scan_to_unicode_cmap

GOLDEN_PATH = Path(__file__).parent / "tounicode_scanner_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def scanned(data):
    result = scan_to_unicode_cmap(data)
    if result is None:
        return None
    mappings, codespace_blocks, usecmap_name = result
    return tuple(mappings.items()), codespace_blocks, usecmap_name


def mappings_of(data):
    result = scan_to_unicode_cmap(data)
    assert result is not None
    return result[0]


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 2339
    assert sum(case["expected"] is not None for case in GOLDEN) == 1760
    origins = {case["origin"].split("/")[0] for case in GOLDEN}
    assert origins == {"corpus", "mutation", "synthetic"}
    assert sum(case["origin"].startswith("corpus/") for case in GOLDEN) == 347
    assert any(case["expected"] and case["expected"][2] for case in GOLDEN), "no usecmap case"


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_scanner_reproduces_the_python_parser_or_declines(index):
    case = GOLDEN[index]
    assert scanned(case["data"]) == case["expected"], case["origin"]


def test_bfchar_destinations_decode_like_the_python_parser():
    data = b"4 beginbfchar <01> <FEFF0041> <02> <41> <03> <004100> <04> <D800> endbfchar"
    mappings = mappings_of(data)
    assert mappings == {
        b"\x01": "A",
        b"\x02": "A",
        b"\x03": b"\x00\x00\x41\x00".decode("utf-16-be"),
        b"\x04": "\ufffd",
    }


def test_a_bfrange_increments_only_its_last_scalar():
    mappings = mappings_of(b"1 beginbfrange <10> <12> <00660066> endbfrange")
    assert mappings == {b"\x10": "ff", b"\x11": "fg", b"\x12": "fh"}


def test_later_mappings_replace_earlier_ones_in_place():
    data = b"2 beginbfchar <01> <0041> <02> <0042> endbfchar 1 beginbfchar <01> <0043> endbfchar"
    mappings = mappings_of(data)
    assert list(mappings.items()) == [(b"\x01", "C"), (b"\x02", "B")]


@pytest.mark.parametrize(
    "data",
    [
        b"1 beginbfchar <0 1> <0041> endbfchar",
        b"1 beginbfchar <1> <0041> endbfchar",
        b"1 beginbfchar (a) <0041> endbfchar",
        b"1 begincidrange <00> <FF> 1 endcidrange",
        b"1 beginbfrange <00> <01> [<0041> (b)] endbfrange",
        b"1 beginbfrange <01> <00> <0041> endbfrange",
        b"(unterminated",
        b"<unterminated",
        b"[" * 100 + b"]" * 100,
    ],
)
def test_declines_what_it_does_not_own(data):
    assert scan_to_unicode_cmap(data) is None

# SPDX-License-Identifier: AGPL-3.0-only

import gzip
import pickle
from pathlib import Path

import pytest

from core_pdf.impl import fonts_cmap
from core_pdf.impl.fonts_cmap import CMapProgram, parse_to_unicode_cmap, parse_to_unicode_program

CYTHONIZED_TESTS = Path(__file__).parents[3] / "packages" / "core-pdf-cythonized" / "tests"
GOLDEN = pickle.loads(
    gzip.decompress((CYTHONIZED_TESTS / "tounicode_scanner_golden.pkl.gz").read_bytes())
)
LARGE_RANGES = [
    b"1 beginbfrange <0000> <FFFF> <0041> endbfrange",
    b"1 beginbfrange <0000> <FFFF> <00410042> endbfrange",
    b"1 beginbfrange <000000> <010000> <0041> endbfrange 1 beginbfchar <01> <0042> endbfchar",
    b"2 beginbfrange <00000000> <FFFFFFFF> <0041> <0000> <0001> <0042> endbfrange",
]


def outcome(parse, data):
    try:
        parsed = parse(data)
    except ValueError as error:
        return ("raises", str(error))
    return (tuple(parsed.mappings.items()), parsed.code_space_ranges, parsed.usecmap_name)


def python_parse(data):
    return parse_to_unicode_program(CMapProgram.parse(data))


@pytest.mark.parametrize(
    "data",
    [case["data"] for case in GOLDEN] + LARGE_RANGES,
    ids=[case["origin"] for case in GOLDEN] + [f"large-{i}" for i in range(len(LARGE_RANGES))],
)
def test_composed_parser_matches_the_python_parser(data):
    assert outcome(parse_to_unicode_cmap, data) == outcome(python_parse, data)


def test_the_kernel_parses_the_common_case():
    data = b"begincmap 1 beginbfchar <41> <0041> endbfchar endcmap"
    assert fonts_cmap.scan_to_unicode_cmap(data) is not None


def test_non_bytes_input_is_parsed_as_bytes():
    data = b"1 beginbfchar <41> <0042> endbfchar"
    assert outcome(parse_to_unicode_cmap, bytearray(data)) == outcome(python_parse, data)
    assert outcome(parse_to_unicode_cmap, memoryview(data)) == outcome(python_parse, data)

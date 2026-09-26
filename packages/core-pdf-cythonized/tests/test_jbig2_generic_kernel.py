# SPDX-License-Identifier: AGPL-3.0-only


import gzip
import pickle
import struct
from pathlib import Path

import pytest

from core_pdf_cythonized import decode_arithmetic_generic_template0

GOLDEN_PATH = Path(__file__).parent / "jbig2_generic_golden.pkl.gz"
GOLDEN = pickle.loads(gzip.decompress(GOLDEN_PATH.read_bytes()))


def test_golden_file_covers_the_cases_it_claims_to():
    assert len(GOLDEN) == 532
    sources = {case["source"] for case in GOLDEN}
    assert {"corpus", "corpus-truncated", "edge", "random"} <= sources
    assert any(case["args"][0] == b"" for case in GOLDEN)
    assert any(case["args"][1] % 8 and case["args"][1] > 8 for case in GOLDEN)
    assert max(case["args"][1] * case["args"][2] for case in GOLDEN) == 2550 * 3300


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_kernel_reproduces_python_output_bytewise(index):
    case = GOLDEN[index]
    got = decode_arithmetic_generic_template0(*case["args"])
    assert type(got) is bytearray
    assert got == case["expected"]


def test_accepts_any_byte_buffer():
    data, width, height = GOLDEN[0]["args"]
    expected = GOLDEN[0]["expected"]
    assert decode_arithmetic_generic_template0(bytearray(data), width, height) == expected
    assert decode_arithmetic_generic_template0(memoryview(data), width, height) == expected


def test_template_zero_decodes_a_black_row():
    assert decode_arithmetic_generic_template0(bytes.fromhex("ff ac"), 8, 1) == b"\xff"


@pytest.mark.parametrize(("width", "height"), [(0, 0), (0, 5), (5, 0)])
def test_empty_regions_decode_to_an_empty_bitmap(width, height):
    assert decode_arithmetic_generic_template0(b"\x12\x34", width, height) == bytearray()


@pytest.mark.parametrize(("width", "height"), [(-1, 1), (1, -1)])
def test_negative_sizes_are_rejected(width, height):
    with pytest.raises(ValueError, match="negative JBIG2 generic region size"):
        decode_arithmetic_generic_template0(b"", width, height)


def test_core_decodes_arithmetic_regions_with_the_kernel():
    pytest.importorskip("core_pdf")
    from core_pdf.impl import graphics_stream_decoding as stream_decoding

    assert stream_decoding.decode_arithmetic_generic_template0 is (
        decode_arithmetic_generic_template0
    )
    page = struct.pack(">IBBBI", 1, 48, 0, 1, 19) + struct.pack(">IIIIBH", 8, 1, 0, 0, 0, 0)
    header = struct.pack(">IIiiBB", 8, 1, 0, 0, 0, 0) + bytes.fromhex("03 ff fd ff 02 fe fe fe")
    region = header + bytes.fromhex("ff ac")
    segment = struct.pack(">IBBBI", 2, 38, 0, 1, len(region)) + region
    end = struct.pack(">IBBBI", 3, 49, 0, 1, 0)
    assert stream_decoding.decode_jbig2(page + segment + end, None) == b"\x00"

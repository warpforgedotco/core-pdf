from __future__ import annotations

import pytest

from core_pdf.impl.spec.s_09_fonts.cmap_decoder import CMapDecoder
from core_pdf.impl.spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.spec.s_09_fonts.cmap_widths import parse_cid_widths


def cmap_program() -> bytes:
    return b"""\
1 begincodespacerange
<00> <ff>
endcodespacerange
256 begincidrange
<00> <ff> 0
endcidrange
"""


def tounicode_program() -> bytes:
    mappings = b"".join(f"<{value:02x}> <{0x4100 + value:04x}>\n".encode() for value in range(256))
    return (
        b"1 begincodespacerange <00> <ff> endcodespacerange\n"
        + b"256 beginbfchar\n"
        + mappings
        + b"endbfchar\n"
    )


CMAP_DATA = cmap_program()
TO_UNICODE_DATA = tounicode_program()
DECODE_DATA = bytes(range(256)) * 16
TO_UNICODE = ToUnicodeCMap(TO_UNICODE_DATA)
CMAP = CMapDecoder(CMAP_DATA)
WIDTH_MAP = parse_cid_widths([0, [500, 510, 520, 530]])


@pytest.mark.benchmark_high_impact
def test_tounicode_construction_benchmark(benchmark) -> None:
    result = benchmark(ToUnicodeCMap, TO_UNICODE_DATA)
    assert result.decode(b"\x01")


@pytest.mark.benchmark_high_impact
def test_cmap_construction_benchmark(benchmark) -> None:
    result = benchmark(CMapDecoder, CMAP_DATA)
    assert result.decode_entries(b"\x01") == [(b"\x01", 1)]


def test_tounicode_decode_benchmark(benchmark) -> None:
    result = benchmark(TO_UNICODE.decode, DECODE_DATA)
    assert result


@pytest.mark.benchmark_high_impact
def test_cmap_decode_benchmark(benchmark) -> None:
    result = benchmark(CMAP.decode_entries, DECODE_DATA)
    assert result


@pytest.mark.benchmark_high_impact
def test_width_lookup_benchmark(benchmark) -> None:
    result = benchmark(WIDTH_MAP.fast_256, 1000.0, 250.0)
    assert len(result) == 256

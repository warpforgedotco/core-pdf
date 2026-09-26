import base64
import zlib
from types import SimpleNamespace

import pytest

from core_pdf.impl import graphics_stream_decoding as recovery
from core_pdf.impl.graphics_decode_compat import FilterParams
from core_pdf_spec.s_07_filters.errors import FilterParseError


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        (b"", b""),
        (b">ff", b""),
        (b"41 42>ff", b"AB"),
        (b"41\x00\xff42", b"AB"),
        (b"4g1!4?2", b"AB"),
        (b"a", b"\xa0"),
        (b"abC>", b"\xab\xc0"),
    ],
)
def test_ascii_hex_ignores_invalid_bytes_and_pads_only_the_final_nibble(encoded, expected):
    assert recovery.apply_ascii_hex(encoded, None) == expected


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        (b"", b""),
        (b"\x80ignored", b""),
        (b"\x00A\x01BC\x80", b"ABC"),
        (b"\x7f" + b"A" * 128 + b"\x80", b"A" * 128),
        (b"\x81Z\xffY\x80", b"Z" * 128 + b"YY"),
        (b"\x05AB", b"AB"),
        (b"\x00A\xff", b"A"),
        (b"\x01", b""),
        (b"\xffZ", b"ZZ"),
    ],
)
def test_run_length_retains_available_literals_and_complete_repeats(encoded, expected):
    assert recovery.apply_run_length(encoded, None) == expected


@pytest.mark.parametrize("payload", [b"", b"a", b"ab", b"abc", b"\x00" * 4, b"hello world"])
@pytest.mark.parametrize("markers", ["adobe", "pdf", "missing"])
@pytest.mark.parametrize("view", [False, True])
def test_ascii85_accepts_optional_adobe_start_and_missing_end(payload, markers, view):
    encoded = base64.a85encode(payload)
    if markers == "adobe":
        encoded = b"<~" + encoded + b"~>"
    elif markers == "pdf":
        encoded += b"~>"
    encoded = b"\x00 \t\r\n" + encoded
    assert recovery.apply_ascii85(memoryview(encoded) if view else encoded, None) == payload


def pack_codes(codes):
    bits = "".join(f"{code:09b}" for code in codes)
    bits += "0" * (-len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b""


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("early_change", [0, 1])
@pytest.mark.parametrize("normalized", [False, True])
def test_lzw_backend_routes_decode_literal_and_dictionary_codes(
    monkeypatch, native, early_change, normalized
):
    monkeypatch.setattr(recovery.imagecodecs, "LZW", SimpleNamespace(available=native))
    encoded = pack_codes([256, 65, 258, 259, 256, 66, 257])
    params = (
        FilterParams(early_change=early_change) if normalized else {"EarlyChange": early_change}
    )
    assert recovery.apply_lzw(encoded, params) == b"AAAAAAB"


@pytest.mark.parametrize("early_change", [0, 1])
@pytest.mark.parametrize(
    ("codes", "expected"), [([], b""), ([256], b""), ([256, 65], b"A"), ([256, 65, 300], b"A")]
)
def test_python_lzw_recovery_returns_only_the_decoded_prefix(
    monkeypatch, early_change, codes, expected
):
    monkeypatch.setattr(recovery.imagecodecs, "LZW", SimpleNamespace(available=False))
    assert recovery.apply_lzw(pack_codes(codes), {"EarlyChange": early_change}) == expected


def test_lzw_backend_failure_preserves_cause(monkeypatch):
    failure = RuntimeError("decoder failed")

    def fail(data):
        raise failure

    monkeypatch.setattr(recovery.imagecodecs, "LZW", SimpleNamespace(available=True))
    monkeypatch.setattr(recovery.imagecodecs, "lzw_decode", fail)
    with pytest.raises(ValueError, match="invalid LZW stream") as caught:
        recovery.apply_lzw(b"invalid", None)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("wbits", [-15, 15, 31])
@pytest.mark.parametrize("truncated", [False, True])
def test_flate_recovers_real_raw_zlib_and_gzip_streams(wbits, truncated):
    payload = bytes(range(256)) * 3
    encoder = zlib.compressobj(wbits=wbits)
    encoded = encoder.compress(payload) + encoder.flush()
    if truncated:
        encoded = encoded[:-1]
    assert recovery.apply_flate(encoded, None) == payload


@pytest.mark.parametrize("encoded", [b"", b"\x03", b"\x00"])
def test_short_incomplete_raw_deflate_is_not_accepted_as_an_empty_stream(encoded):
    assert recovery.recover_flate(encoded, -15) is None


def test_empty_flate_input_is_recoverable():
    assert recovery.apply_flate(b"", None) == b""


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        (b"q", True),
        (b"1 0 0 1 0 0 cm", True),
        (b"% q\n(foo (q) bar) <71> /q [q] << /X Q >> BT", True),
        (b"% q\n(foo (q) bar) <71> /q [q] << /X Q >>", False),
        (b"[q [BT] << /X Q >>]", False),
        (b"<< /X [q] /Y << /Z BT >> >>", False),
        (b"/BT", False),
        (b"(BT)", False),
        (b"<4254>", False),
        (b"% BT", False),
        (b"not_an_operator", False),
        (b"] >> } q", True),
        (b"1 " * 63 + b"q", True),
        (b"1 " * 64 + b"q", False),
        (b" " * 1023 + b"q", True),
        (b" " * 1024 + b"q", False),
    ],
)
@pytest.mark.parametrize("view", ["bytes", "whole", "slice"])
def test_content_probe_ignores_nested_operators_and_respects_scan_limits(encoded, expected, view):
    data = encoded
    if view == "whole":
        data = memoryview(encoded)
    elif view == "slice":
        data = memoryview(b"q " + encoded + b" Q")[2:-2]
    assert recovery.looks_like_pdf_content_stream(data) is expected


@pytest.mark.parametrize("encoded", [b"q 1 0 0 1 0 0 cm Q", b"BT (hello) Tj ET"])
def test_mislabeled_flate_preserves_actual_content(encoded):
    assert recovery.apply_flate(encoded, None) == encoded


@pytest.mark.parametrize("encoded", [b"\xff", b"(q)", b"/BT", b"[q]", b"<< /X Q >>"])
def test_mislabeled_flate_rejects_data_with_only_nested_operators(encoded):
    with pytest.raises(FilterParseError, match="invalid FlateDecode stream"):
        recovery.apply_flate(encoded, None)


@pytest.mark.parametrize("wbits", [-15, 15, 47])
def test_incomplete_single_byte_headers_never_count_as_recovered_streams(wbits):
    for byte in range(256):
        assert recovery.recover_flate(bytes([byte]), wbits) is None, (wbits, byte)


@pytest.mark.parametrize("wbits", [-15, 15, 31])
def test_complete_empty_compressed_streams_remain_valid(wbits):
    encoder = zlib.compressobj(wbits=wbits)
    encoded = encoder.flush()
    assert recovery.apply_flate(encoded, None) == b""


def test_flate_can_recover_a_valid_body_with_a_broken_checksum():
    payload = b"BT (recoverable) Tj ET" * 10
    encoded = zlib.compress(payload)[:-4] + bytes(4)
    assert recovery.recover_flate(encoded) == payload


def test_flate_does_not_recover_a_corrupt_body_despite_a_valid_header():
    assert recovery.recover_flate(b"\x78\x9c" + b"\xff" * 8 + bytes(4)) is None


@pytest.mark.parametrize("encoded", [b"!", b"!z", b"v", b"uuuuu"])
def test_ascii85_recovery_does_not_hide_invalid_digits_or_groups(encoded):
    with pytest.raises(ValueError):
        recovery.apply_ascii85(encoded, None)

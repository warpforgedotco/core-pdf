"""ISO 32000-1 7.4 stream codecs, including state and representation boundaries."""

import base64
import binascii
import zlib

import pytest

from core_pdf_spec.s_07_filters import codecs
from core_pdf_spec.s_07_filters.decode_spec import FilterParams
from core_pdf_spec.s_07_filters.errors import FilterParseError


@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 5, 3276, 3280, 4097])
@pytest.mark.parametrize("zeros", [False, True])
@pytest.mark.parametrize("whitespace", [b"", b"\x00\t\n\x0c\r "])
def test_ascii85_matches_independent_encoder(length, zeros, whitespace):
    payload = (b"\0" * 4 if zeros else bytes(range(1, 256))) * (length + 1)
    payload = payload[:length]
    encoded = base64.a85encode(payload, adobe=False) + b"~>"
    encoded = whitespace + whitespace.join(bytes([byte]) for byte in encoded) + b"ignored after EOD"
    assert codecs.apply_ascii85(memoryview(encoded), None) == payload


@pytest.mark.parametrize(
    "encoded",
    [
        b"!",
        b"!~>",
        b"!z~>",
        b"z!~>",
        b"uuuuu~>",
        b"zuuuuu~>",
        b"uu~>",
        b"zuu~>",
        b"!!!!!" * 820 + b"uuuuu~>",
        b"<~z~>",
        b"y~>",
        b"v~>",
        b"\x80~>",
    ],
)
def test_ascii85_rejects_bad_tuple_digits_and_overflow(encoded):
    with pytest.raises(FilterParseError, match="ASCII85Decode"):
        codecs.apply_ascii85(bytes(encoded), None)


@pytest.mark.parametrize("position", range(8))
def test_ascii85_rejects_invalid_digits_at_every_tuple_position(position):
    encoded = bytearray(b"!!!!!!!!~>")
    encoded[position] = 118
    with pytest.raises(FilterParseError):
        codecs.apply_ascii85(bytes(encoded), None)


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [(b">tail", b""), (b"a>", b"\xa0"), (b"00 ff\x00\t\n\x0c\r>", b"\0\xff")],
)
def test_asciihex_eod_whitespace_and_odd_nibble(encoded, expected):
    assert codecs.apply_ascii_hex(encoded, None) == expected


def test_asciihex_missing_eod_and_invalid_digits_remain_errors():
    with pytest.raises(FilterParseError, match="end marker"):
        codecs.apply_ascii_hex(b"00", None)
    with pytest.raises(binascii.Error):
        codecs.apply_ascii_hex(b"gg>", None)


@pytest.mark.parametrize("length", [1, 2, 127, 128])
def test_runlength_literal_and_repeated_runs(length):
    literal = bytes(range(length))
    encoded = bytes([length - 1]) + literal + b"\x81Z\xffY\x80ignored"
    assert codecs.apply_run_length(encoded, None) == literal + b"Z" * 128 + b"YY"


@pytest.mark.parametrize("encoded", [b"", b"\0A", b"\x02AB", b"\x81", b"\xff"])
def test_runlength_requires_complete_runs_and_eod(encoded):
    with pytest.raises(FilterParseError):
        codecs.apply_run_length(encoded, None)


def test_flate_round_trip_and_truncation():
    payload = bytes(range(256)) * 10
    encoded = zlib.compress(payload)
    assert codecs.apply_flate(encoded, None) == payload
    with pytest.raises(FilterParseError, match="FlateDecode"):
        codecs.apply_flate(encoded[:-2], None)


def pack_words(words):
    """Pack explicitly specified code/width pairs, independently of decoder state."""
    bits = "".join(f"{code:0{width}b}" for code, width in words)
    bits += "0" * (-len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b""


@pytest.mark.parametrize("early_change", [0, 1])
def test_lzw_crosses_every_width_boundary_and_saturates_dictionary(early_change):
    # Table 8: sizes increase one code earlier under EarlyChange=1.
    # Repeating literal codes makes the expected plaintext independent of the
    # encoder's dictionary, while explicit boundaries cover 9 -> 10 -> 11 -> 12.
    counts = [255 - early_change, 512, 1024, 2400]
    words = [(256, 9)]
    for width, count in zip((9, 10, 11, 12), counts, strict=True):
        words.extend((65, width) for _ in range(count))
    words.extend([(256, 12), (66, 9), (257, 9)])
    expected = b"A" * sum(counts) + b"B"
    assert codecs.apply_lzw(pack_words(words), {"EarlyChange": early_change}) == expected


def test_lzw_forward_dictionary_reference_and_clear():
    data = pack_words([(256, 9), (65, 9), (258, 9), (259, 9), (256, 9), (66, 9), (257, 9)])
    assert codecs.apply_lzw(data, FilterParams()) == b"AAAAAAB"


@pytest.mark.parametrize(
    ("codes", "decoded"), [([256, 65], b"A"), ([256, 65, 300], b"A"), ([256], b"")]
)
def test_incomplete_lzw_retains_only_the_decoded_prefix(codes, decoded):
    with pytest.raises(codecs.IncompleteLzwError) as caught:
        codecs.apply_lzw(pack_words((code, 9) for code in codes), None)
    assert caught.value.decoded == decoded


def test_initial_invalid_lzw_code_has_no_recovery_prefix():
    with pytest.raises(ValueError, match="invalid LZW code: 300"):
        codecs.apply_lzw(pack_words([(300, 9)]), None)


def test_bit_reader_retains_partial_eof_bits_for_smaller_reads():
    reader = codecs.BitReader(memoryview(b"\xab\xcd"))
    assert reader.read_bits(3) == 5
    assert reader.read_bits(5) == 11
    assert reader.read_bits(9) is None
    assert reader.read_bits(4) == 12
    assert reader.read_bits(4) == 13
    assert reader.read_bits(1) is None
    assert reader.read_bits(0) == 0


def test_ascii85_mixed_zero_shorthand_full_tuple_and_tail():
    payload = b"\0\0\0\0abcdxyz"
    assert codecs.apply_ascii85(base64.a85encode(payload) + b"~>", None) == payload


@pytest.mark.parametrize(
    "name",
    [
        "EarlyChange",
        "Predictor",
        "Columns",
        "Colors",
        "BitsPerComponent",
        "K",
        "Rows",
        "DamagedRowsBeforeError",
    ],
)
@pytest.mark.parametrize("value", [True, 1.5, "1", b"1"])
def test_decode_parameter_integers_do_not_coerce_other_object_types(name, value):
    with pytest.raises(ValueError, match=name):
        FilterParams.from_parms({name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EarlyChange", -1),
        ("EarlyChange", 2),
        ("Predictor", 0),
        ("Predictor", 16),
        ("Columns", 0),
        ("Colors", -1),
        ("BitsPerComponent", 0),
        ("BitsPerComponent", 17),
        ("Rows", -1),
        ("DamagedRowsBeforeError", -1),
        ("BlackIs1", 1),
        ("EncodedByteAlign", "true"),
    ],
)
def test_decode_parameter_ranges_and_booleans_are_strict(name, value):
    with pytest.raises(ValueError, match=name):
        FilterParams.from_parms({name: value})


@pytest.mark.parametrize("enabled", [False, True])
def test_decode_parameter_defaults_and_explicit_values(enabled):
    assert FilterParams.from_parms(None) == FilterParams()
    assert FilterParams.from_parms({"Columns": None, "JBIG2Globals": None}) == FilterParams()
    globals_stream = object()
    params = FilterParams.from_parms(
        {
            "EarlyChange": 0,
            "Predictor": 15,
            "Columns": 7,
            "Colors": 3,
            "BitsPerComponent": 16,
            "K": -1,
            "Rows": 9,
            "DamagedRowsBeforeError": 2,
            "BlackIs1": enabled,
            "EncodedByteAlign": enabled,
            "JBIG2Globals": globals_stream,
        }
    )
    assert params == FilterParams(
        early_change=0,
        predictor=15,
        columns=7,
        colors=3,
        bits_per_component=16,
        k=-1,
        rows=9,
        damaged_rows_before_error=2,
        black_is_1=enabled,
        encoded_byte_align=enabled,
        has_columns=True,
        jbig2_globals=globals_stream,
    )
    assert params.jbig2_globals is globals_stream


def test_decode_parameters_require_a_dictionary_or_null():
    with pytest.raises(ValueError, match="DecodeParms dictionary"):
        FilterParams.from_parms([])

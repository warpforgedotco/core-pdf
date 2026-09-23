import numpy as np
import pytest

from core_pdf.impl.runtime import codec_backends as codecs


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        ("u1", [0, 127, 255], [0, 127, 255]),
        ("u2", [0, 32768, 65535], [0, 128, 255]),
        ("i2", [-1, 256, 32767], [0, 1, 127]),
        ("i1", [-128, 0, 127], [0, 0, 127]),
        ("f4", [-1, 12.6, 300], [0, 13, 255]),
    ],
)
@pytest.mark.parametrize("channels", [False, True])
def test_normalized_samples_use_declared_precision_and_contiguous_layout(
    dtype, values, expected, channels
):
    source = np.asarray([values, values], dtype=dtype)[:, ::-1]
    if channels:
        source = source[:, :, None]
    result = codecs.normalize_imagecodecs_array(source, name="test", allow_float=True)
    wanted = np.asarray([expected[::-1], expected[::-1]], dtype=np.uint8)
    if channels:
        wanted = wanted[:, :, None]
    np.testing.assert_array_equal(result, wanted)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous


def test_uint16_precision_can_be_preserved_without_aliasing_strided_storage():
    source = np.arange(12, dtype=np.uint16).reshape(3, 4)[:, ::2]
    result = codecs.normalize_imagecodecs_array(source, name="test", preserve_uint16=True)
    np.testing.assert_array_equal(result, source)
    assert result.dtype == np.uint16
    assert result.flags.c_contiguous


@pytest.mark.parametrize(
    "source",
    [
        np.zeros(3),
        np.zeros((1, 1, 1, 1)),
        np.zeros((1, 1), dtype=bool),
        np.zeros((1, 1), dtype=complex),
        np.zeros((1, 1, 0), dtype=np.uint8),
        np.zeros((1, 1), dtype=np.float32),
    ],
)
def test_unsupported_decoded_shapes_and_types_fail_clearly(source):
    with pytest.raises(codecs.CodecUnsupportedError, match="test decoder returned"):
        codecs.normalize_imagecodecs_array(source, name="test")


@pytest.mark.parametrize(
    ("data", "valid", "error"),
    [
        (b"", True, codecs.CodecParseError),
        (b"bad", False, codecs.CodecParseError),
        (b"header", True, codecs.CodecUnsupportedError),
    ],
)
def test_codec_failures_distinguish_invalid_and_unsupported_streams(data, valid, error):
    cause = RuntimeError("backend failed")
    with pytest.raises(error) as failure:
        codecs.raise_codec_error(data, cause, check=lambda value: valid, name="test")
    assert failure.value.__cause__ is cause


def test_failed_signature_probe_still_reports_parse_error():
    def probe(data):
        raise RuntimeError("probe failed")

    with pytest.raises(codecs.CodecParseError):
        codecs.raise_codec_error(b"bad", RuntimeError(), check=probe, name="test")


@pytest.mark.parametrize(
    ("setting", "expected"),
    [(None, 3), ("", 3), ("bad", 3), ("0", 1), ("-2", 1), ("2", 2), ("99", 4)],
)
def test_jpx_thread_setting_respects_default_and_hard_bound(monkeypatch, setting, expected):
    monkeypatch.setattr(codecs.os, "cpu_count", lambda: 3)
    if setting is None:
        monkeypatch.delenv("CORE_PDF_JPX_THREADS", raising=False)
    else:
        monkeypatch.setenv("CORE_PDF_JPX_THREADS", setting)
    assert codecs.jpx_thread_count() == expected


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("colors", [1, 3, 4])
def test_png_prediction_backend_preserves_exact_sample_bytes(bits, colors):
    columns = 8
    row_length = columns * colors * bits // 8
    raw = bytes((index * 37 + 19) % 256 for index in range(row_length))
    result = codecs.png_predict_codec(
        b"\0" + raw + b"\2" + bytes(row_length),
        columns=columns,
        colors=colors,
        bits_per_component=bits,
    )
    if bits < 8 and colors != 1:
        assert result is None
    else:
        assert result == raw * 2


@pytest.mark.parametrize(
    ("columns", "colors", "bits", "data"),
    [
        (1, 2, 8, b"\0ab"),
        (3, 1, 1, b"\0x"),
        (0, 1, 8, b"\0x"),
        (1_000_001, 1, 8, b"\0x"),
        (1, 1, 8, b""),
        (1, 1, 8, b"\0"),
    ],
)
def test_png_backend_declines_inapplicable_layouts(columns, colors, bits, data):
    assert (
        codecs.png_predict_codec(data, columns=columns, colors=colors, bits_per_component=bits)
        is None
    )


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("colors", [1, 3, 4])
def test_tiff_prediction_accumulates_per_channel_and_restarts_each_row(bits, colors):
    dtype = np.dtype("u1" if bits == 8 else ">u2")
    maximum = (1 << bits) - 1
    row = [maximum] * colors + [2] * colors + [3] * colors
    encoded = np.asarray(row * 2, dtype=dtype).tobytes()

    def decoder(data, columns, colors):
        return codecs.tiff_predict_codec(
            data, columns=columns, colors=colors, bits_per_component=bits
        )

    actual = decoder(encoded + b"x", 3, colors)
    expected = np.asarray(
        ([maximum] * colors + [1] * colors + [4] * colors) * 2, dtype=dtype
    ).tobytes()
    assert actual == expected


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize(("data", "columns"), [(b"", 3), (b"x", 3), (b"abc", 0)])
def test_tiff_prediction_ignores_incomplete_rows(bits, data, columns):
    assert (
        codecs.tiff_predict_codec(data, columns=columns, colors=1, bits_per_component=bits) == b""
    )


def pack_rows(rows, bits):
    output = bytearray()
    for row in rows:
        binary = "".join(f"{value:0{bits}b}" for value in row)
        binary += "0" * (-len(binary) % 8)
        output.extend(int(binary[index : index + 8], 2) for index in range(0, len(binary), 8))
    return bytes(output)


@pytest.mark.parametrize("bits", [1, 2, 4])
@pytest.mark.parametrize("colors", [1, 3])
@pytest.mark.parametrize("columns", [3, 8])
def test_subbyte_tiff_prediction_preserves_channel_and_padded_row_boundaries(bits, colors, columns):
    mask = (1 << bits) - 1
    row = [mask] * colors + [1] * ((columns - 1) * colors)
    expected = [((mask + column) & mask) for column in range(columns) for _ in range(colors)]
    encoded = pack_rows([row, row], bits)
    assert codecs.tiff_predict_bits(encoded, columns, colors, bits) == pack_rows(
        [expected, expected], bits
    )


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_subbyte_tiff_prediction_returns_empty_for_missing_row(bits):
    assert codecs.tiff_predict_bits(b"", 8, 1, bits) == b""

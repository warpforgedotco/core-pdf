"""Image capture resolves decoder inputs while preserving unrelated objects."""

import zlib

import pytest

from core_pdf_spec.s_07_filters.pipeline import decode_stream_data
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.s_08_graphics.image_spec import image_source_from_stream
from core_pdf_spec.types import PdfName, PdfReference


class SelectedImageResolver(ObjectResolver):
    def missing_object(self, ref: PdfReference) -> object:
        raise ValueError(f"unexpected image reference: {ref.object_number}")


@pytest.mark.parametrize("indirect_mask", [False, True])
@pytest.mark.parametrize(
    ("filter_key", "params_key"),
    [("Filter", "DecodeParms"), ("FFilter", "FDecodeParms"), ("F", "DP")],
)
def test_main_and_soft_mask_share_selected_resolution(
    indirect_mask: bool, filter_key: str, params_key: str
) -> None:
    resolver = SelectedImageResolver(b"", {}, {})
    resolver.objects.update(
        {
            key_for(1): 2,
            key_for(2): 1,
            key_for(3): 8,
            key_for(4): [PdfReference(7), PdfReference(2)],
            key_for(5): PdfName.of("DeviceGray"),
            key_for(6): PdfName.of("FlateDecode"),
            key_for(7): 0,
            key_for(8): {
                "Predictor": PdfReference(1),
                "Columns": PdfReference(1),
                "Colors": PdfReference(2),
                "BitsPerComponent": PdfReference(3),
            },
            key_for(9): False,
        }
    )
    unrelated = {"Metadata": PdfReference(99)}
    dictionary: dict[object, object] = {
        "Width": PdfReference(1),
        "Height": PdfReference(2),
        "BitsPerComponent": PdfReference(3),
        "Decode": PdfReference(4),
        "ColorSpace": PdfReference(5),
        "ImageMask": PdfReference(9),
        filter_key: PdfReference(6),
        params_key: PdfReference(8),
        "Unrelated": unrelated,
    }
    raw = memoryview(zlib.compress(b"\x20\x20"))

    def unexpected_decode(*args: object, **kwargs: object) -> bytes:
        pytest.fail("image capture must leave sample decoding to its consumer")

    mask = PdfStream(dict(dictionary), raw_data=raw, decoder=unexpected_decode)
    resolver.objects[key_for(10)] = mask
    source = PdfStream(
        {**dictionary, "SMask": PdfReference(10) if indirect_mask else mask},
        raw_data=raw,
        decoder=unexpected_decode,
    )
    original_source = dict(source.dictionary)
    try:
        captured = image_source_from_stream(source, resolver)
        assert captured.soft_mask is not None
        assert captured.raw is raw
        assert captured.soft_mask.raw is raw
        for resolved in (captured.dictionary, captured.soft_mask.dictionary):
            assert resolved["Width"] == 2
            assert resolved["Height"] == 1
            assert resolved["BitsPerComponent"] == 8
            assert resolved["Decode"] == [0, 1]
            assert resolved["ColorSpace"] == PdfName.of("DeviceGray")
            assert resolved["ImageMask"] is False
            assert resolved[filter_key] == PdfName.of("FlateDecode")
            assert resolved[params_key] == {
                "Predictor": 2,
                "Columns": 2,
                "Colors": 1,
                "BitsPerComponent": 8,
            }
            assert resolved["Unrelated"] is unrelated
            # Abbreviations are normalized by the inline-image parser. Check
            # these resolved decoder inputs independently of that parser.
            assert (
                decode_stream_data(
                    raw, {"Filter": resolved[filter_key], "DecodeParms": resolved[params_key]}
                )
                == b"\x20\x40"
            )
        assert captured.dictionary is not source.dictionary
        assert captured.soft_mask.dictionary is not mask.dictionary
        assert source.dictionary == original_source
        assert mask.dictionary == dictionary
        assert resolver.objects[key_for(4)] == [PdfReference(7), PdfReference(2)]
        assert source.decoder is unexpected_decode
        assert mask.decoder is unexpected_decode
    finally:
        raw.release()
        resolver.close()


@pytest.mark.parametrize("soft_mask", [False, True])
def test_selected_image_input_resolution_preserves_errors(soft_mask: bool) -> None:
    resolver = SelectedImageResolver(b"", {}, {})
    image = PdfStream({"Width": PdfReference(99)}, raw_data=b"sample")
    source = PdfStream({"SMask": image}) if soft_mask else image
    try:
        with pytest.raises(ValueError, match="unexpected image reference: 99"):
            image_source_from_stream(source, resolver)
    finally:
        resolver.close()


@pytest.mark.parametrize("mask", [None, PdfName.of("None"), {}])
def test_nonstream_soft_mask_remains_absent(mask: object) -> None:
    resolver = SelectedImageResolver(b"", {}, {})
    source = PdfStream({"SMask": mask, "Metadata": PdfReference(99)}, raw_data=b"sample")
    try:
        captured = image_source_from_stream(source, resolver)
        assert captured.raw is source.raw_data
        assert captured.soft_mask is None
        assert captured.dictionary == source.dictionary
        assert captured.dictionary is not source.dictionary
    finally:
        resolver.close()

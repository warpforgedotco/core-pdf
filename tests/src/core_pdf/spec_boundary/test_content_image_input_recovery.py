"""Selected image resolution preserves core's decoding and alpha policies."""

import zlib

import pytest

from core_pdf.impl._impl.capture.images import image_source_from_stream
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf.impl._impl.graphics.images import prepare_image
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.types import PdfName, PdfReference


@pytest.mark.parametrize("numeric_strings", [False, True])
@pytest.mark.parametrize(
    ("filter_key", "params_key"),
    [("Filter", "DecodeParms"), ("FFilter", "FDecodeParms"), ("F", "DP")],
)
def test_reader_prepares_selected_mask_inputs_and_keeps_raw_alpha_policy(
    numeric_strings: bool, filter_key: str, params_key: str
) -> None:
    class Resolver(ObjectResolver):
        def missing_object(self, ref: PdfReference) -> object:
            pytest.fail(f"unrelated image object resolved: {ref}")

    resolver = Resolver(b"", {})
    resolver.objects.update(
        {
            key_for(1): "2" if numeric_strings else 2,
            key_for(2): "1" if numeric_strings else 1,
            key_for(3): "8" if numeric_strings else 8,
            key_for(4): PdfName.of("DeviceGray"),
            key_for(5): PdfName.of("FlateDecode"),
            key_for(6): {"Predictor": 2, "Columns": 2, "Colors": 1, "BitsPerComponent": 8},
            key_for(7): [0, 1],
        }
    )
    dictionary = {
        "Width": PdfReference(1),
        "Height": PdfReference(2),
        "BitsPerComponent": PdfReference(3),
        "ColorSpace": PdfReference(4),
        filter_key: PdfReference(5),
        params_key: PdfReference(6),
        "Decode": PdfReference(7),
        "Metadata": PdfReference(99),
    }
    mask_raw = zlib.compress(b"\x20\x20")
    mask = PdfStream(dictionary, raw_data=mask_raw)
    resolver.objects[key_for(8)] = mask
    stream = PdfStream(dict(dictionary, SMask=PdfReference(8)), raw_data=zlib.compress(b"\x70\x10"))
    try:
        source, mask_alpha = image_source_from_stream(stream, resolver)
        assert mask_alpha == (mask_raw[0] + mask_raw[1]) / (2 * 255)
        prepared = prepare_image(source)
        assert prepared is not None
        assert prepared.soft_mask is not None
        assert prepared.soft_mask.array.tolist() == [[[32], [64]]]
        # Numeric-string sample metadata takes core's RGB recovery path.
        expected = (
            [[[112, 112, 112, 32], [128, 128, 128, 64]]]
            if numeric_strings
            else [[[112, 32], [128, 64]]]
        )
        assert prepared.raster.array.tolist() == expected
        assert mask.dictionary == dictionary
        assert stream.dictionary["SMask"] == PdfReference(8)
    finally:
        resolver.close()

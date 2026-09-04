# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TypeAlias, cast

from core_pdf.impl.spec.s_07_filters import decode_spec as stream_decode_spec

PdfStreamDictionary: TypeAlias = dict[object, object]
PdfStreamDecodeSpec: TypeAlias = stream_decode_spec.StreamDecodeSpec | PdfStreamDictionary | None


__all__ = ("PdfStream",)


class PdfStream:
    """PDF stream object: dictionary plus source bytes."""

    __slots__ = (
        "dictionary",
        "raw_data",
        "spec",
    )

    dictionary: PdfStreamDictionary
    raw_data: bytes | memoryview
    spec: PdfStreamDecodeSpec

    def __init__(
        self,
        dictionary: object | None = None,
        raw_data: bytes | memoryview = b"",
        spec: object | None = None,
        decoded_data: bytes | None = None,
    ) -> None:
        if dictionary is not None and not isinstance(dictionary, dict):
            raise ValueError("invalid stream dictionary")
        if not isinstance(raw_data, (bytes, memoryview)):
            raise ValueError("invalid stream data")
        if spec is not None and not isinstance(spec, (stream_decode_spec.StreamDecodeSpec, dict)):
            raise ValueError("invalid stream decode spec")
        if decoded_data is not None and not isinstance(decoded_data, bytes):
            raise ValueError("invalid stream data")
        self.dictionary = cast(PdfStreamDictionary, dictionary) if dictionary is not None else {}
        self.raw_data = decoded_data if decoded_data is not None else raw_data
        self.spec = None if decoded_data is not None else cast(PdfStreamDecodeSpec, spec)

    def replace(self, **kwargs: object) -> "PdfStream":
        dictionary = kwargs.get("dictionary", self.dictionary)
        spec = kwargs.get("spec", dictionary if dictionary is not self.dictionary else self.spec)
        if spec is not None and not isinstance(spec, (stream_decode_spec.StreamDecodeSpec, dict)):
            raise ValueError("invalid stream decode spec")
        return PdfStream(
            dictionary=cast(PdfStreamDictionary, dictionary),
            raw_data=cast(bytes | memoryview, kwargs.get("raw_data", self.raw_data)),
            spec=cast(PdfStreamDecodeSpec, spec),
            decoded_data=cast(bytes | None, kwargs.get("decoded_data")),
        )

    @property
    def data(self) -> bytes:
        from core_pdf.impl.spec.s_07_filters import pipeline as stream_pipeline

        spec = self.spec
        if isinstance(spec, dict):
            spec = stream_decode_spec.normalize_stream_decode_spec(spec)
        return stream_pipeline.decode_stream_data(
            self.raw_data,
            spec,
            parent_dictionary=self.dictionary,
        )

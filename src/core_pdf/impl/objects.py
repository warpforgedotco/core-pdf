# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TypeAlias, cast

from core_pdf.impl.engine.spec.s_07_filters import decode_spec as stream_decode_spec
from core_pdf.impl.engine.spec.s_07_filters import pipeline as stream_pipeline
from core_pdf.impl.primitives import (
    MISSING,
    MissingObject,
    PdfName,
    PdfReference,
    PdfString,
)

PdfStreamDictionary: TypeAlias = dict[object, object]
PdfStreamDecodeSpec: TypeAlias = stream_decode_spec.StreamDecodeSpec | PdfStreamDictionary | None


__all__ = (
    "MISSING",
    "MissingObject",
    "PdfName",
    "PdfReference",
    "PdfStream",
    "PdfString",
)


class PdfStream:
    """PDF stream object: dictionary plus raw and lazily decoded bytes."""

    __slots__ = ("dictionary", "raw_data", "spec", "decoded_data", "data_view_cache")

    dictionary: PdfStreamDictionary
    raw_data: bytes | memoryview
    spec: PdfStreamDecodeSpec
    decoded_data: bytes | None
    data_view_cache: memoryview | None

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
        self.raw_data = raw_data
        self.spec = cast(PdfStreamDecodeSpec, spec)
        self.decoded_data = decoded_data
        self.data_view_cache = None

    def replace(self, **kwargs: object) -> "PdfStream":
        dictionary = kwargs.get("dictionary", self.dictionary)
        spec = kwargs.get("spec", dictionary if dictionary is not self.dictionary else self.spec)
        if spec is not None and not isinstance(spec, (stream_decode_spec.StreamDecodeSpec, dict)):
            raise ValueError("invalid stream decode spec")
        return PdfStream(
            dictionary=cast(PdfStreamDictionary, dictionary),
            raw_data=cast(bytes | memoryview, kwargs.get("raw_data", self.raw_data)),
            spec=cast(PdfStreamDecodeSpec, spec),
            decoded_data=cast(bytes | None, kwargs.get("decoded_data", self.decoded_data)),
        )

    @property
    def data(self) -> bytes:
        if self.decoded_data is None:
            spec = self.spec
            if isinstance(spec, dict):
                spec = stream_decode_spec.normalize_stream_decode_spec(spec)
                self.spec = spec
            self.decoded_data = stream_pipeline.decode_stream_data(self.raw_data, spec)
        return self.decoded_data

    @property
    def data_view(self) -> memoryview:
        data_view = self.data_view_cache
        if data_view is None:
            data_view = memoryview(self.data)
            self.data_view_cache = data_view
        return data_view

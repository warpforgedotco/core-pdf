# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TypeAlias, cast

from core_pdf_spec.s_07_filters import decode_spec as stream_decode_spec
from core_pdf_spec.types import MISSING, MissingObject

PdfStreamDictionary: TypeAlias = dict[object, object]
PdfStreamDecodeSpec: TypeAlias = stream_decode_spec.StreamDecodeSpec | PdfStreamDictionary | None


__all__ = ("PdfStream",)


class PdfStream:
    """PDF stream object: dictionary plus source bytes."""

    __slots__ = (
        "dictionary",
        "raw_data",
        "spec",
        "decoder",
    )

    dictionary: PdfStreamDictionary
    raw_data: bytes | memoryview
    spec: PdfStreamDecodeSpec
    decoder: stream_decode_spec.StreamDecoder

    def __init__(
        self,
        dictionary: object | None = None,
        raw_data: bytes | memoryview = b"",
        spec: object | None = None,
        *,
        decoder: stream_decode_spec.StreamDecoder | None = None,
    ) -> None:
        if dictionary is not None and not isinstance(dictionary, dict):
            raise ValueError("invalid stream dictionary")
        if not isinstance(raw_data, (bytes, memoryview)):
            raise ValueError("invalid stream data")
        if spec is not None and not isinstance(spec, (stream_decode_spec.StreamDecodeSpec, dict)):
            raise ValueError("invalid stream decode spec")
        self.dictionary = cast(PdfStreamDictionary, dictionary) if dictionary is not None else {}
        self.raw_data = raw_data
        self.spec = cast(PdfStreamDecodeSpec, spec)
        if decoder is None:
            from core_pdf_spec.s_07_filters.pipeline import decode_stream_data

            decoder = decode_stream_data
        self.decoder = decoder

    def replace(
        self,
        *,
        dictionary: PdfStreamDictionary | None | MissingObject = MISSING,
        raw_data: bytes | memoryview | MissingObject = MISSING,
        spec: PdfStreamDecodeSpec | MissingObject = MISSING,
        decoder: stream_decode_spec.StreamDecoder | None | MissingObject = MISSING,
    ) -> PdfStream:
        """Copy source state without activating decoding or evaluating the stream."""
        next_dictionary = self.dictionary if isinstance(dictionary, MissingObject) else dictionary
        if isinstance(spec, MissingObject):
            spec = next_dictionary if self.spec is self.dictionary else self.spec
        return PdfStream(
            dictionary=next_dictionary,
            raw_data=self.raw_data if isinstance(raw_data, MissingObject) else raw_data,
            spec=spec,
            decoder=self.decoder if isinstance(decoder, MissingObject) else decoder,
        )

    @property
    def data(self) -> bytes:
        return self.decoder(
            self.raw_data,
            self.spec,
            parent_dictionary=self.dictionary,
        )

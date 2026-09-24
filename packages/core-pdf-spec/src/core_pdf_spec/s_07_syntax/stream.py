# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from core_pdf_spec.s_07_filters import decode_spec as stream_decode_spec
from core_pdf_spec.types import MISSING, MissingObject

if TYPE_CHECKING:
    from core_pdf_spec.s_07_syntax.types import PdfDict

# Forward references rather than imports: PdfDict names PdfObject, which names
# PdfStream, so the module defining the object types has to import this one.
# Only the runtime import is circular -- a stream dictionary is an ordinary PDF
# dictionary, and saying so keeps every reader of stream.dictionary typed.
# Both aliases appear only in annotations, which are not evaluated at runtime.
PdfStreamDictionary: TypeAlias = "PdfDict"
PdfStreamDecodeSpec: TypeAlias = "stream_decode_spec.StreamDecodeSpec | PdfDict | None"


__all__ = ("PdfStream",)


class PdfStream:
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
        self.dictionary = dictionary if dictionary is not None else {}
        self.raw_data = raw_data
        self.spec = spec
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

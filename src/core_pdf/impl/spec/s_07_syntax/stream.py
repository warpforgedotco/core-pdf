# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import threading
from typing import TypeAlias, cast

from core_pdf.impl.spec.s_07_filters import decode_spec as stream_decode_spec

PdfStreamDictionary: TypeAlias = dict[object, object]
PdfStreamDecodeSpec: TypeAlias = stream_decode_spec.StreamDecodeSpec | PdfStreamDictionary | None


__all__ = ("PdfStream",)


class PdfStream:
    """PDF stream object: dictionary plus raw and lazily decoded bytes."""

    __slots__ = (
        "dictionary",
        "raw_data",
        "spec",
        "decoded_data",
        "internal_lock",
    )

    dictionary: PdfStreamDictionary
    raw_data: bytes | memoryview
    spec: PdfStreamDecodeSpec
    decoded_data: bytes | None

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
        self.internal_lock = threading.RLock()

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
        decoded_data = self.decoded_data
        if decoded_data is None:
            with self.internal_lock:
                decoded_data = self.decoded_data
                if decoded_data is None:
                    from core_pdf.impl.spec.s_07_filters import (
                        pipeline as stream_pipeline,
                    )

                    spec = self.spec
                    if isinstance(spec, dict):
                        spec = stream_decode_spec.normalize_stream_decode_spec(spec)
                        self.spec = spec
                    decoded_data = stream_pipeline.decode_stream_data(
                        self.raw_data,
                        spec,
                        parent_dictionary=self.dictionary,
                    )
                    self.decoded_data = decoded_data
        return decoded_data

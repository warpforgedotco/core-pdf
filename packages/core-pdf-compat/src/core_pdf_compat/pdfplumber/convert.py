# Ported from pdfplumber's convert.py and container.py, Copyright (c) 2015,
# Jeremy Singer-Vine, under the MIT License.
from __future__ import annotations

import base64
import csv
import json
from collections.abc import Callable
from io import StringIO
from typing import Any, TextIO

ENCODINGS_TO_TRY = ("utf-8", "latin-1", "utf-16", "utf-16le")
CSV_COLS_REQUIRED = ["object_type"]
CSV_COLS_TO_PREPEND = [
    "page_number",
    "x0",
    "x1",
    "y0",
    "y1",
    "doctop",
    "top",
    "bottom",
    "width",
    "height",
]


def get_attr_filter(
    include_attrs: list[str] | None = None, exclude_attrs: list[str] | None = None
) -> Callable[[str], bool]:
    if include_attrs is not None and exclude_attrs is not None:
        raise ValueError("Cannot specify `include_attrs` and `exclude_attrs` at the same time.")
    if include_attrs is not None:
        included = {*CSV_COLS_REQUIRED, *include_attrs}
        return lambda attr: attr in included
    if exclude_attrs is not None:
        nonexcludable = set(exclude_attrs).intersection(CSV_COLS_REQUIRED)
        if nonexcludable:
            raise ValueError(f"Cannot exclude these required properties: {list(nonexcludable)}")
        excluded = set(exclude_attrs)
        return lambda attr: attr not in excluded
    return lambda _: True


class Serializer:
    def __init__(
        self,
        precision: int | None = None,
        include_attrs: list[str] | None = None,
        exclude_attrs: list[str] | None = None,
    ) -> None:
        self.precision = precision
        self.attr_filter = get_attr_filter(include_attrs, exclude_attrs)
        self.active: set[int] = set()

    def serialize(self, obj: Any) -> Any:
        if obj is None or type(obj) in {int, str}:
            return obj
        if isinstance(obj, bool):
            return int(obj)
        if isinstance(obj, float):
            return float(obj) if self.precision is None else round(float(obj), self.precision)
        if isinstance(obj, (list, tuple, dict)):
            # Annotation data resolves references, and page trees point back at themselves.
            if id(obj) in self.active:
                return str(type(obj).__name__)
            self.active.add(id(obj))
            try:
                return self.serialize_container(obj)
            finally:
                self.active.discard(id(obj))
        if isinstance(obj, (bytes, bytearray)):
            return self.serialize_bytes(bytes(obj))
        rawdata = getattr(obj, "rawdata", None)
        if rawdata is not None:
            return {"rawdata": base64.b64encode(rawdata).decode("ascii")}
        return str(obj)

    def serialize_container(self, obj: list[Any] | tuple[Any, ...] | dict[Any, Any]) -> Any:
        if isinstance(obj, list):
            return [self.serialize(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self.serialize(item) for item in obj)
        keep = self.attr_filter if "object_type" in obj else lambda _: True
        return {str(key): self.serialize(value) for key, value in obj.items() if keep(key)}

    @staticmethod
    def serialize_bytes(value: bytes) -> str | None:
        try:
            return value.decode(ENCODINGS_TO_TRY[0])
        except UnicodeDecodeError:
            return None


class Container:
    @property
    def objects(self) -> dict[str, list[dict[str, Any]]]:
        raise NotImplementedError

    def to_dict(self, object_types: list[str] | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def to_json(
        self,
        stream: TextIO | None = None,
        object_types: list[str] | None = None,
        include_attrs: list[str] | None = None,
        exclude_attrs: list[str] | None = None,
        precision: int | None = None,
        indent: int | None = None,
    ) -> str | None:
        serialized = Serializer(precision, include_attrs, exclude_attrs).serialize(
            self.to_dict(object_types)
        )
        if stream is None:
            return json.dumps(serialized, indent=indent)
        json.dump(serialized, stream, indent=indent)
        return None

    def to_csv(
        self,
        stream: TextIO | None = None,
        object_types: list[str] | None = None,
        precision: int | None = None,
        include_attrs: list[str] | None = None,
        exclude_attrs: list[str] | None = None,
    ) -> str | None:
        buffer = StringIO()
        output = buffer if stream is None else stream
        if object_types is None:
            object_types = [*self.objects, "annot"]
        serializer = Serializer(precision, include_attrs, exclude_attrs)
        rows: list[dict[str, Any]] = []
        fields: set[str] = set()
        pages: list[Any] | None = getattr(self, "pages", None)
        for page in [self] if pages is None else pages:
            for object_type in object_types:
                objects = getattr(page, f"{object_type}s")
                if objects:
                    rows += serializer.serialize(objects)
                    fields |= {key for key, value in objects[0].items() if type(value) is not dict}
        optional = CSV_COLS_TO_PREPEND + sorted(fields - {*CSV_COLS_REQUIRED, *CSV_COLS_TO_PREPEND})
        writer = csv.DictWriter(
            output,
            fieldnames=CSV_COLS_REQUIRED + [col for col in optional if serializer.attr_filter(col)],
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
        )
        writer.writeheader()
        writer.writerows(rows)
        if stream is None:
            return buffer.getvalue()
        return None


__all__ = ("Container", "Serializer", "get_attr_filter")

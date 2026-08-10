"""High-level pikepdf-shaped facade backed by core-pdf."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, MutableMapping, MutableSequence
from os import PathLike
from typing import Any, BinaryIO, cast, overload

from core_pdf.api.v0.compat._common import Document, PdfUnsupportedError
from core_pdf.api.v0.compat.pypdf import PdfPageObject, PdfReader, PdfWriter
from core_pdf.api.v0.compat.state import StructuredState


class Attachments(MutableMapping[str, bytes]):
    def __init__(self, values: dict[str, bytes] | None = None) -> None:
        self._values = dict(values or {})

    def __getitem__(self, key: str) -> bytes:
        return self._values[key]

    def __setitem__(self, key: str, value: bytes) -> None:
        self._values[str(key)] = bytes(value)

    def __delitem__(self, key: str) -> None:
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class DocumentInfo(MutableMapping[str, Any]):
    def __init__(self, owner: "Pdf", values: dict[str, Any]) -> None:
        self._owner = owner
        self._values = dict(values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._values[str(key)] = value
        self._owner.metadata[str(key).lstrip("/")] = value

    def __delitem__(self, key: str) -> None:
        del self._values[key]
        self._owner.metadata.pop(key.lstrip("/"), None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class Pages(MutableSequence[PdfPageObject]):
    def __init__(self, owner: "Pdf", values: tuple[PdfPageObject, ...] = ()) -> None:
        self._owner = owner
        self._values = list(values)

    @overload
    def __getitem__(self, index: int) -> PdfPageObject: ...

    @overload
    def __getitem__(self, index: slice) -> list[PdfPageObject]: ...

    def __getitem__(self, index: int | slice) -> Any:
        return self._values[index]

    @overload
    def __setitem__(self, index: int, value: PdfPageObject) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[PdfPageObject]) -> None: ...

    def __setitem__(self, index: int | slice, value: Any) -> None:
        self._values[index] = value
        self._sync()

    def __delitem__(self, index: int | slice) -> None:
        del self._values[index]
        self._sync()

    def __len__(self) -> int:
        return len(self._values)

    def insert(self, index: int, value: PdfPageObject) -> None:
        self._values.insert(index, value)
        self._sync()

    def _sync(self) -> None:
        pages = tuple(item._page for item in self._values)
        self._owner._document = StructuredState.from_structured(
            Document(pages=pages, metadata=self._owner._document.snapshot.metadata)
        )
        self._values = [PdfPageObject(self._owner._document, page) for page in pages]


class Pdf(PdfReader):
    pages: Any

    @classmethod
    def new(cls) -> Pdf:
        instance = cls.__new__(cls)
        instance._document = StructuredState.from_structured(Document())
        instance.__dict__["pages"] = Pages(instance)
        instance.metadata = {}
        instance.trailer = {}
        instance._attachments = Attachments()
        instance._outlines = []
        instance._source_name = None
        return instance

    @classmethod
    def open(cls, stream: object, *args: object, **kwargs: object) -> Pdf:
        del args
        password = kwargs.pop("password", None)
        if kwargs:
            raise TypeError(f"unsupported open options: {', '.join(kwargs)}")
        return cls(cast(str | PathLike[str] | bytes, stream), cast(str | None, password))

    def __init__(self, stream: object, password: str | None = None) -> None:
        self._source_name = str(stream) if isinstance(stream, (str, PathLike)) else None
        super().__init__(cast(str | PathLike[str] | bytes, stream), password)
        if self._document is None:
            raise PdfUnsupportedError("password required for encrypted PDF")
        self.__dict__["pages"] = Pages(self, tuple(self.pages))
        self._outlines = [
            [item.level + 1, item.title, item.page + 1]
            for item in self.outline
            if item.page is not None
        ]
        self._attachments = Attachments(
            {embedded.filename: embedded.data for embedded in self._document.source_pdf.attachments}
        )

    @property
    def docinfo(self) -> DocumentInfo:
        return DocumentInfo(self, dict(self.metadata))

    @property
    def filename(self) -> str | None:
        return self._source_name

    @property
    def is_linearized(self) -> bool:
        return False

    def save(self, filename: object, **kwargs: object) -> bytes:
        del kwargs
        writer = PdfWriter()
        for page in self.pages:
            writer.add_page(page)
        writer.add_metadata(self.metadata)
        parents: dict[int, object] = {}
        for row in self._outlines:
            if len(row) < 3 or not isinstance(row[0], int) or not isinstance(row[2], int):
                continue
            level = max(1, row[0])
            parent = parents.get(level - 1)
            destination = writer.add_outline_item(
                str(row[1]), row[2] - 1, parent=cast(Any, parent) if parent is not None else None
            )
            parents[level] = destination
            for deeper in tuple(parents):
                if deeper > level:
                    del parents[deeper]
        for attachment_name, data in self.attachments.items():
            writer.add_attachment(attachment_name, data)
        return writer.write(cast(str | PathLike[str] | BinaryIO, filename))

    @property
    def attachments(self) -> Attachments:
        return self._attachments


__all__ = ("Attachments", "DocumentInfo", "Pdf")

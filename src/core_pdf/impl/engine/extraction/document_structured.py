# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, TypeAlias, cast

from core_layout.impl.layout.geometry import RectBox

from core_pdf.impl.engine.extraction.common.page_content import PageContentRecord
from core_pdf.impl.engine.spec.s_09_fonts.encoding import decode_pdf_text_string
from core_pdf.impl.exceptions import PdfError
from core_pdf.impl.objects import PdfName, PdfReference, PdfStream, PdfString
from core_pdf.impl.types import PageSelection

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class DocumentStructuredMixin:
    PDF_SPEC_PATH_FRAGMENT = str(Path("core_pdf/impl/engine/spec"))

    def extract_structured(
        self: Any,
        *,
        pages: PageSelection | None = None,
        include_content: bool = True,
        use_resolved_text: bool = False,
        include_tables: bool = True,
        include_images: bool = True,
        include_annotations: bool = True,
        include_fields: bool = True,
        include_lines: bool = False,
        include_words: bool = False,
        include_text_runs: bool = False,
        include_drawings: bool = True,
        include_inline_images: bool = True,
        include_xobject_images: bool = True,
        skip_errors: bool = True,
        table_options: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        selected_pages = self.selected_page_indexes(pages)
        selected_page_items = list(self.iter_selected_pages(pages))
        selected_page_numbers = [page_index + 1 for page_index in selected_pages]
        page_count = self.page_count()
        table_options_dict = dict(table_options or {})

        metadata = self.get_metadata()
        document_result = self.extract()
        if pages is None:
            canonical_document = document_result
        else:
            document_type = type(document_result)
            canonical_document = document_type(
                pages=tuple(document_result.pages[index] for index in selected_pages),
                metadata=document_result.metadata,
                diagnostics=document_result.diagnostics,
                schema_version=document_result.schema_version,
            )
        warnings = [
            {
                "code": diagnostic.code,
                "message": diagnostic.message,
                "severity": diagnostic.severity,
                "page_number": diagnostic.page_number,
            }
            for diagnostic in document_result.diagnostics
            if diagnostic.page_number is None or diagnostic.page_number in selected_page_numbers
        ]
        result: dict[str, object] = {
            "schema_version": document_result.schema_version,
            "document": canonical_document.to_json_dict(),
            "metadata": metadata,
            "page_count": page_count,
        }
        if warnings:
            result["warnings"] = warnings
        errors: list[PageContentRecord] = []

        def append_error(
            name: str,
            exc: Exception,
            page_index: int | None = None,
            page: Any | None = None,
        ) -> None:
            error: dict[str, object] = {
                "section": name,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            if page_index is None:
                error["selected_pages"] = selected_page_numbers
            else:
                error["page_index"] = page_index
                error["page_number"] = getattr(page, "page_number", page_index + 1)
                error["page_label"] = getattr(page, "label", None)
            errors.append(error)

        def collect_section(name: str, fn: Any) -> Any:
            try:
                return fn()
            except Exception as exc:
                if not self.should_skip_error(exc, skip_errors):
                    raise
                append_error(name, exc)
                return []

        def collect_page_section(name: str, fn: Any) -> list[PageContentRecord]:
            records: list[PageContentRecord] = []
            for page_index, page in selected_page_items:
                try:
                    records.extend(fn(page_index, page))
                except Exception as exc:
                    if not self.should_skip_error(exc, skip_errors):
                        raise
                    append_error(name, exc, page_index, page)
            return records

        def page_content(page_index: int, page: Any) -> list[PageContentRecord]:
            return [
                self.with_page_metadata(page_index, page, item)
                for item in page.extract_content(
                    include_words=include_words,
                    include_drawings=include_drawings,
                    include_images=include_images,
                    include_annotations=include_annotations,
                    use_resolved_text=use_resolved_text,
                )
            ]

        def page_tables(page_index: int, page: Any) -> list[PageContentRecord]:
            return self.build_page_table_records(
                page_index,
                page,
                table_options=table_options_dict,
            )

        field_index_counter = 0

        def page_fields(page_index: int, page: Any) -> list[PageContentRecord]:
            nonlocal field_index_counter

            field_records: list[PageContentRecord] = []
            for page_field_index, field in enumerate(page.get_fields()):
                field_records.append(
                    self.with_page_metadata(
                        page_index,
                        page,
                        {
                            "field_index": field_index_counter,
                            "page_field_index": page_field_index,
                            "kind": "field",
                            "name": field.name,
                            "type": field.type,
                            "value": field.value,
                            "value_text": field.value_text,
                            "value_words": field.value_words(),
                            "bbox": field.rect,
                            "rect": field.rect,
                            "dict": field.dict,
                            "has_widget": field.widget is not None,
                            "kid_count": len(field.kids),
                        },
                    )
                )
                field_index_counter += 1
            return field_records

        if include_content:
            result["content"] = collect_section(
                "content",
                lambda: collect_page_section("content", page_content),
            )
        if include_tables:
            result["tables"] = collect_section(
                "tables",
                lambda: collect_page_section("tables", page_tables),
            )
        if include_images:
            result["images"] = collect_section(
                "images",
                lambda: collect_page_section(
                    "images",
                    lambda page_index, page: [
                        self.with_page_metadata(page_index, page, image)
                        for image in page.extract_images(
                            include_inline=include_inline_images,
                            include_xobjects=include_xobject_images,
                        )
                    ],
                ),
            )
        if include_annotations:
            result["annotations"] = collect_section(
                "annotations",
                lambda: collect_page_section(
                    "annotations",
                    lambda page_index, page: [
                        self.with_page_metadata(
                            page_index,
                            page,
                            {
                                "annotation_index": annotation_index,
                                "kind": "annotation",
                                "subtype": annotation.subtype,
                                "bbox": annotation.rect,
                                "contents": annotation.contents,
                                "dest": annotation.dest,
                                "action": annotation.action,
                                "dict": annotation.dict,
                            },
                        )
                        for annotation_index, annotation in enumerate(page.get_annotations())
                    ],
                ),
            )
        if include_fields:
            result["fields"] = collect_section(
                "fields",
                lambda: collect_page_section("fields", page_fields),
            )
        if include_lines:
            result["lines"] = collect_section(
                "lines",
                lambda: collect_page_section(
                    "lines",
                    lambda page_index, page: [
                        self.with_page_metadata(page_index, page, line)
                        for line in page.extract_lines(include_words=include_words)
                    ],
                ),
            )
        if include_words:
            result["words"] = collect_section(
                "words",
                lambda: collect_page_section(
                    "words",
                    lambda page_index, page: [
                        self.with_page_metadata(page_index, page, word)
                        for word in page.extract_words()
                    ],
                ),
            )
        if include_text_runs:
            result["text_runs"] = collect_section(
                "text_runs",
                lambda: collect_page_section(
                    "text_runs",
                    lambda page_index, page: [
                        self.with_page_metadata(page_index, page, run)
                        for run in page.extract_text_runs()
                    ],
                ),
            )

        def result_count(key: str) -> int:
            value = result.get(key)
            return len(value) if isinstance(value, (list, tuple, dict)) else 0

        result["summary"] = {
            "page_count": page_count,
            "selected_page_count": len(selected_pages),
            "selected_pages": selected_page_numbers,
            "content_items": result_count("content"),
            "tables": result_count("tables"),
            "images": result_count("images"),
            "annotations": result_count("annotations"),
            "fields": result_count("fields"),
            "lines": result_count("lines"),
            "words": result_count("words"),
            "text_runs": result_count("text_runs"),
            "warnings": len(warnings),
            "errors": len(errors),
        }
        if errors:
            result["errors"] = errors
        return result

    def extract_structured_json(self: Any, **kwargs: Any) -> dict[str, JsonValue]:
        return cast(
            dict[str, JsonValue],
            self.json_safe(self.extract_structured(**kwargs)),
        )

    def to_structured_json_string(
        self: Any,
        *,
        indent: int | None = 2,
        sort_keys: bool = True,
        **kwargs: Any,
    ) -> str:
        return json.dumps(
            self.extract_structured_json(**kwargs),
            indent=indent,
            sort_keys=sort_keys,
        )

    def to_json(
        self: Any,
        *,
        indent: int | None = 2,
        sort_keys: bool = True,
    ) -> str:
        """Serialize the canonical extraction IR as JSON."""
        return self.extract().to_json(indent=indent, sort_keys=sort_keys)

    def to_html(self: Any) -> str:
        """Render the canonical extraction IR as semantic HTML."""
        return self.extract().to_html()

    def extract_structured_text(
        self: Any,
        *,
        pages: PageSelection | None = None,
        include_text_runs: bool = False,
        skip_errors: bool = True,
    ) -> dict[str, object]:
        return self.extract_structured(
            pages=pages,
            include_content=True,
            include_tables=False,
            include_images=False,
            include_annotations=False,
            include_fields=False,
            include_lines=True,
            include_words=True,
            include_text_runs=include_text_runs,
            include_drawings=False,
            skip_errors=skip_errors,
        )

    def extract_structured_text_json(
        self: Any,
        *,
        pages: PageSelection | None = None,
        include_text_runs: bool = False,
        skip_errors: bool = True,
    ) -> dict[str, JsonValue]:
        return cast(
            dict[str, JsonValue],
            self.json_safe(
                self.extract_structured_text(
                    pages=pages,
                    include_text_runs=include_text_runs,
                    skip_errors=skip_errors,
                )
            ),
        )

    def to_structured_text_json_string(
        self: Any,
        *,
        pages: PageSelection | None = None,
        include_text_runs: bool = False,
        skip_errors: bool = True,
        indent: int | None = 2,
        sort_keys: bool = True,
    ) -> str:
        return json.dumps(
            self.extract_structured_text_json(
                pages=pages,
                include_text_runs=include_text_runs,
                skip_errors=skip_errors,
            ),
            indent=indent,
            sort_keys=sort_keys,
        )

    @classmethod
    def json_safe(cls, value: object) -> JsonValue:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, PdfName):
            return value.value
        if isinstance(value, PdfString):
            return decode_pdf_text_string(value.data)
        if isinstance(value, PdfReference):
            return {
                "object_number": value.object_number,
                "generation_number": value.generation_number,
            }
        if isinstance(value, PdfStream):
            return {
                "dictionary": cls.json_safe(value.dictionary),
                "raw_bytes": len(value.raw_data),
            }
        if isinstance(value, RectBox):
            return [value.x0, value.y0, value.x1, value.y1]
        if isinstance(value, bytes):
            return value.decode("latin-1")
        if isinstance(value, memoryview):
            return bytes(value).decode("latin-1")
        if isinstance(value, (list, tuple)):
            return [cls.json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(cls.json_safe(key)): cls.json_safe(item) for key, item in value.items()}
        return str(value)

    @staticmethod
    def should_skip_error(exc: Exception, skip_errors: bool) -> bool:
        if not skip_errors:
            return False
        if isinstance(exc, PdfError):
            return True
        if not isinstance(exc, ValueError):
            return False
        traceback = exc.__traceback__
        while traceback is not None:
            if (
                DocumentStructuredMixin.PDF_SPEC_PATH_FRAGMENT
                in traceback.tb_frame.f_code.co_filename
            ):
                return True
            traceback = traceback.tb_next
        return False


__all__ = ("DocumentStructuredMixin", "JsonScalar", "JsonValue")

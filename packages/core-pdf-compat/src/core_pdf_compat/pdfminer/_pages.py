from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any, cast

from core_pdf import PdfDocument, PdfPage
from core_pdf.impl.document.recovery.lexer import PdfLexer
from core_pdf.impl.document.recovery.xref import XRefScanner
from core_pdf.impl.exceptions import PdfError
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.types import PdfReference
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import iter_xref_revisions, merge_xref_sections


def internal_pdfminer_resolvable_pages(  # noqa: C901
    document: PdfDocument,
) -> Iterator[tuple[int, PdfPage]]:
    data = bytes(document.raw_data)

    def fallback_pages(object_keys: Iterable[tuple[int, int]]) -> Iterator[tuple[int, PdfPage]]:
        found = 0
        seen: set[int] = set()
        for object_number, generation_number in object_keys:
            if object_number in seen:
                continue
            seen.add(object_number)
            entry = strict_xref.get((object_number << 16) | generation_number)
            value: object = None
            if entry is not None and entry.object_stream is None:
                lexer = PdfLexer(data, recover_malformed_objects=False)
                lexer.rewind(entry.offset)
                try:
                    parsed = lexer.parse_indirect_object()
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    value = parsed
            elif entry is not None and entry.object_stream is not None:
                try:
                    value = document.resolver.resolve(
                        PdfReference(object_number, generation_number)
                    )
                except Exception:
                    value = None
            if not isinstance(value, dict):
                continue
            if recover_pdf_name(value.get("Type")) != "Page":
                continue
            if found >= len(document.pages):
                raise PdfError("fallback page is unavailable in the native page list")
            try:
                page = document.pages[found]
            except IndexError as exc:
                raise PdfError("fallback page is not resolvable") from exc
            yield found, page
            found += 1

    def fallback_projection() -> Iterator[tuple[int, PdfPage]]:
        trailer_match = re.search(rb"(?m)^trailer\b", data)
        if trailer_match is None:
            raise PdfError("No /Root object")
        trailer_data = re.split(
            rb"(?m)^trailer\b|startxref|%%EOF",
            data[trailer_match.end() :],
            maxsplit=1,
        )[0]
        if re.search(rb"/Root\b", trailer_data) is None:
            raise PdfError("No /Root object")
        malformed_root = re.search(rb"/Root\s+\d+\s+\d+\s+R\b", trailer_data) is None
        recovered: dict[int, tuple[int, int]] = {}
        for match in re.finditer(rb"(?m)^(\d+)\s+(\d+)\s+obj\b", data[: trailer_match.start()]):
            object_number = int(match.group(1))
            generation_number = int(match.group(2))
            if generation_number <= 65535:
                recovered[object_number] = (generation_number, match.start())
        root_match = re.search(rb"/Root\s+(\d+)\s+(\d+)\s+R\b", trailer_data)
        fallback_catalog: dict[Any, Any] | None = None
        if root_match is not None:
            root_number = int(root_match.group(1))
            root_generation = int(root_match.group(2))
            recovered_root = recovered.get(root_number)
            if recovered_root is not None and recovered_root[0] == root_generation:
                root_lexer = PdfLexer(data, recover_malformed_objects=True)
                root_lexer.rewind(recovered_root[1])
                try:
                    root_value = root_lexer.parse_indirect_object()
                except Exception:
                    root_value = None
                if (
                    isinstance(root_value, dict)
                    and recover_pdf_name(root_value.get("Type")) == "Catalog"
                ):
                    fallback_catalog = root_value
        if fallback_catalog is not None:
            try:
                catalog_pages = document.catalog().get("Pages")
            except Exception:
                catalog_pages = fallback_catalog.get("Pages")
        else:
            catalog_pages = None

        reachable_page_ids: set[int] = set()
        visited_tree_nodes: set[tuple[int, int]] = set()

        def collect_reachable_pages(node: Any) -> None:
            node_reference = node if isinstance(node, PdfReference) else None
            if node_reference is not None:
                key = (node_reference.object_number, node_reference.generation_number)
                if key in visited_tree_nodes:
                    return
                visited_tree_nodes.add(key)
                try:
                    node = document.resolver.resolve(node_reference)
                except Exception:
                    return
            if not isinstance(node, dict):
                return
            node_type = recover_pdf_name(node.get("Type"))
            if node_type == "Page":
                if node_reference is not None:
                    reachable_page_ids.add(node_reference.object_number)
                return
            try:
                kids = document.resolver.resolve(node.get("Kids"))
            except Exception:
                return
            if isinstance(kids, (tuple, list)):
                for child in kids:
                    collect_reachable_pages(child)

        collect_reachable_pages(catalog_pages)

        def belongs_to_catalog_tree(value: dict[Any, Any]) -> bool:
            if not isinstance(catalog_pages, PdfReference):
                return True
            parent = value.get("Parent")
            seen_parents: set[tuple[int, int]] = set()
            while isinstance(parent, PdfReference):
                key = (parent.object_number, parent.generation_number)
                if key == (catalog_pages.object_number, catalog_pages.generation_number):
                    return True
                if key in seen_parents:
                    return False
                seen_parents.add(key)
                try:
                    parent_value = document.resolver.resolve(parent)
                except Exception:
                    return False
                if not isinstance(parent_value, dict):
                    return False
                parent = parent_value.get("Parent")
            return False

        found = 0
        for object_number, (generation_number, offset) in recovered.items():
            header_end = re.match(rb"\d+\s+\d+\s+obj\b", data[offset:])
            if header_end is None:
                continue
            value_start = offset + header_end.end()
            value_start += len(data[value_start:]) - len(data[value_start:].lstrip())
            if data[value_start : value_start + 2] != b"<<" and not malformed_root:
                continue
            lexer = PdfLexer(data, recover_malformed_objects=True)
            lexer.rewind(offset)
            try:
                value = lexer.parse_indirect_object()
            except Exception:
                continue
            if not isinstance(value, dict):
                continue
            if recover_pdf_name(value.get("Type")) != "Page":
                continue
            try:
                resolved_value = document.resolver.resolve(
                    PdfReference(object_number, generation_number)
                )
            except Exception:
                resolved_value = None
            if isinstance(resolved_value, dict):
                value = resolved_value
            if reachable_page_ids and object_number not in reachable_page_ids:
                continue
            if not belongs_to_catalog_tree(value):
                continue
            try:
                document.resolver.resolve(PdfReference(object_number, generation_number))
                page = document.pages[found]
            except Exception:
                page = PdfPage(document, cast(PdfDict, value), found + 1)
            yield found, page
            found += 1

    previous_line = b""
    start: int | None = None
    for raw_line in reversed(data.splitlines()):
        line = raw_line.strip()
        if line == b"startxref":
            if previous_line.isdigit():
                candidate = int(previous_line)
                if candidate < 2**31:
                    start = candidate
            break
        if line:
            previous_line = line
    if start is None:
        yield from fallback_projection()
        return
    section_start = start
    section_pos = XRefScanner.skip_ws(data, section_start)
    section_is_direct = data[section_pos : section_pos + 4] == b"xref"
    section_is_stream = re.match(rb"\d+\s+\d+\s+obj\b", data[section_pos:]) is not None
    if not section_is_direct and not section_is_stream:
        preceding = data[max(0, section_pos - 3) : section_pos + 4]
        relative = preceding.find(b"xref")
        candidate = max(0, section_pos - 3) + relative if relative >= 0 else -1
        if candidate >= 0 and candidate <= section_pos < candidate + 4:
            section_start = candidate
        else:
            yield from fallback_projection()
            return
    try:
        read_section = partial(
            XRefScanner.recover_section_at,
            document.raw_data,
            recover_malformed_objects=False,
        )
        revisions = list(iter_xref_revisions(section_start, read_section))
        strict_xref = merge_xref_sections(revision.entries for revision in revisions)
        strict_trailer = revisions[0].trailer
    except Exception:
        yield from fallback_projection()
        return

    xref_sections: list[dict[int, Any]] = []
    section_seen: set[int] = set()
    try:
        while section_start not in section_seen:
            section_seen.add(section_start)
            section = read_section(section_start)
            section_seen.add(section.offset)
            entries = section.entries
            previous = cast(int | None, section.trailer.get("Prev"))
            xref_stream = (
                cast(int | None, section.trailer.get("XRefStm"))
                if section.kind == "table"
                else None
            )
            if xref_stream is not None:
                supplemental = read_section(xref_stream, stream_only=True)
                entries = dict(entries)
                entries.update(supplemental.entries)
            xref_sections.append(entries)
            if previous is None:
                break
            section_start = previous
    except Exception:
        xref_sections = [strict_xref]

    info_reference = strict_trailer.get("Info")
    if isinstance(info_reference, PdfReference):
        info_key = (info_reference.object_number << 16) | info_reference.generation_number
        info_entry = strict_xref.get(info_key)
        if info_entry is not None and info_entry.in_use and info_entry.object_stream is None:
            expected_header = re.compile(
                rb"\s*"
                + str(info_reference.object_number).encode("ascii")
                + rb"\s+"
                + str(info_reference.generation_number).encode("ascii")
                + rb"\s+obj\b"
            )
            if expected_header.match(data, info_entry.offset):
                info_lexer = PdfLexer(data, recover_malformed_objects=False)
                info_lexer.rewind(info_entry.offset)
                info_lexer.parse_indirect_object()

    def reference_is_resolvable(value: object) -> bool:
        if not isinstance(value, PdfReference):
            return True
        key = (value.object_number << 16) | value.generation_number
        candidates = [section[key] for section in xref_sections if key in section]
        if not candidates:
            return False
        for entry in candidates:
            if not entry.in_use:
                continue
            if entry.object_stream is not None:
                return True
            search_end = min(len(data), entry.offset + 1_048_576)
            header_pattern = re.compile(rb"(?<!\d)(\d+)\s+(\d+)\s+obj\b")
            first_header = header_pattern.search(data, entry.offset, search_end)
            expected_pattern = re.compile(
                rb"(?<!\d)"
                + str(value.object_number).encode("ascii")
                + rb"\s+"
                + str(value.generation_number).encode("ascii")
                + rb"\s+obj\b"
            )
            expected_header = expected_pattern.search(data, entry.offset, search_end)
            if expected_header is None:
                continue
            offset_start = XRefScanner.skip_ws(data, entry.offset)
            if first_header is not None and first_header.start() == offset_start:
                found_number, found_generation = (int(item) for item in first_header.groups())
                if (
                    found_number != value.object_number
                    or found_generation != value.generation_number
                ):
                    continue
            return True
        return False

    root_reference = strict_trailer.get("Root")
    if root_reference is None:
        raise PdfError("No /Root object")
    if not reference_is_resolvable(root_reference):
        hard_mismatch = False
        if isinstance(root_reference, PdfReference):
            root_key = (root_reference.object_number << 16) | root_reference.generation_number
            root_entry = strict_xref.get(root_key)
            if root_entry is not None and root_entry.object_stream is None:
                offset = XRefScanner.skip_ws(data, root_entry.offset)
                header = re.match(rb"(\d+)\s+(\d+)\s+obj\b", data[offset:])
                if header is not None:
                    hard_mismatch = (int(header.group(1)), int(header.group(2))) != (
                        root_reference.object_number,
                        root_reference.generation_number,
                    )
        if not hard_mismatch:
            yield from fallback_pages(
                ((key >> 16, key & 0xFFFF) for key, entry in strict_xref.items() if entry.in_use)
            )
        return
    try:
        catalog = document.resolver.resolve(root_reference)
    except Exception:
        return
    if not isinstance(catalog, dict):
        raise PdfError("invalid /Root object")
    pages_reference = catalog.get("Pages")
    page_index = 0
    visited: set[tuple[str, int, int] | tuple[str, int]] = set()

    def traverse(value: object, depth: int = 0) -> Iterator[tuple[int, PdfPage]]:
        nonlocal page_index
        if depth > 100:
            return
        valid_reference = reference_is_resolvable(value)
        try:
            node = document.resolver.resolve(value)
        except Exception:
            return
        if not isinstance(node, dict):
            return
        marker: tuple[str, int, int] | tuple[str, int] = (
            ("ref", value.object_number, value.generation_number)
            if isinstance(value, PdfReference)
            else ("dict", id(node))
        )
        duplicate = marker in visited
        visited.add(marker)
        node_type = recover_pdf_name(node.get("Type"))
        if node_type == "Pages":
            if not valid_reference or duplicate:
                return
            kids = document.resolver.resolve(node.get("Kids"))
            if not isinstance(kids, list):
                return
            for kid in kids:
                yield from traverse(kid, depth + 1)
            return
        if node_type != "Page":
            return
        current_index = page_index
        page_index += 1
        if valid_reference and not duplicate and current_index < len(document.pages):
            yield current_index, document.pages[current_index]

    declared_pages = tuple(traverse(pages_reference))
    if declared_pages:
        yield from declared_pages
        return
    yield from fallback_pages(
        ((key >> 16, key & 0xFFFF) for key, entry in strict_xref.items() if entry.in_use)
    )

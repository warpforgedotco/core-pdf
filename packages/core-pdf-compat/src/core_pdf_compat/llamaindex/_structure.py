"""The checks pypdf's reader would fail on, for a reader that goes straight to core-pdf."""

from __future__ import annotations

import re
from collections.abc import Iterator
from mmap import mmap

from core_pdf import PdfDocument
from core_pdf.impl.types import PdfReference
from core_pdf_compat.pypdf._reader_scan import STARTXREF_RECOVERY_LINES, previous_line
from core_pdf_compat.pypdf.errors import PdfStreamError
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import key_for

# pypdf Configuration.page_tree_maximum_depth and page_tree_maximum_entries.
PAGE_TREE_MAXIMUM_DEPTH = 100
PAGE_TREE_MAXIMUM_ENTRIES = 100_000


TRUNCATED_EOF_MARKERS = (b"%%EO", b"%%E", b"%%", b"%")


def is_startxref_value(value: bytes) -> bool:
    return re.fullmatch(rb"[+-]?[0-9]+(?:_[0-9]+)*", value) is not None


def finds_startxref(data: bytes | mmap) -> bool:
    """Whether pypdf (non-strict) locates %%EOF and the startxref offset above it.

    Mirrors PdfReader._find_eof_marker and _find_startxref_pos: the last line
    may carry the marker inline (py-pdf/pypdf#4127) or truncated, duplicate
    markers above it are skipped (#4008), and a corrupt trailing pointer falls
    back to the previous intact startxref (#3238).
    """
    try:
        position = end_position = len(data)
        line = b""
        last = True
        while True:
            marker = line.find(b"%%EOF")
            if marker == 0:
                break
            if marker > 0 and last:
                if line[:marker].lstrip(b" \t\n\r\x00\x0c"):
                    position = end_position - len(line) + marker
                break
            if line and last:
                if line.strip().endswith(TRUNCATED_EOF_MARKERS):
                    break
                last = False
            end_position = position
            line, position = previous_line(data, position)

        line, position = previous_line(data, position)
        for _ in range(STARTXREF_RECOVERY_LINES):
            if not line.startswith(b"%%EOF") or position == 0:
                break
            line, position = previous_line(data, position)
        try:
            int(line)
        except ValueError:
            return line.startswith(b"startxref") and is_startxref_value(line[9:].strip())
        line, position = previous_line(data, position)
        if line.startswith(b"startxref"):
            return True

        offset_found = False
        for _ in range(STARTXREF_RECOVERY_LINES):
            if position <= 0:
                break
            line, position = previous_line(data, position)
            if not line.startswith(b"startxref"):
                try:
                    int(line)
                except ValueError:
                    offset_found = False
                else:
                    offset_found = True
                continue
            if len(line) > 9:
                if is_startxref_value(line[9:].strip()):
                    return True
                continue
            return offset_found
        return False
    except PdfStreamError, ValueError:
        return False


def validate_pypdf_page_tree(pdf: PdfDocument) -> None:
    literal_trailers = tuple(pdf.iter_literal_trailer_dictionaries())
    if literal_trailers:
        latest_root = literal_trailers[-1].get("Root")
        if isinstance(latest_root, PdfReference):
            root_key = key_for(latest_root.object_number, latest_root.generation_number)
            if root_key not in pdf.xref:
                raise ValueError("catalog root references a missing object generation")
        elif latest_root is not None and not isinstance(latest_root, dict):
            raise ValueError("invalid catalog root")

    if "Root" not in pdf.trailer_dict:
        return  # pypdf reports this when the pages are read.
    root = pdf.resolver.as_dict(pdf.catalog().get("Pages"))
    if root is None:
        raise ValueError("invalid object in /Pages")
    # pypdf walks the tree iteratively. A cycle is a /Pages node reached again
    # while it is still an ancestor; a page shared by two parents is valid.
    # Entries that are not dictionaries, and empty ones, are skipped.
    entries = 0
    ancestors: set[int] = set()
    stack: list[tuple[PdfDict, Iterator[object], int]] = []

    def enter(node: PdfDict, depth: int) -> None:
        if depth > PAGE_TREE_MAXIMUM_DEPTH:
            raise ValueError(f"maximum page tree depth reached: {depth}")
        declared_type = node.get("Type")
        if declared_type is not None:
            node_type = str(declared_type)
        else:
            node_type = "Page" if "Kids" not in node else "Pages"
        if node_type != "Pages":
            return
        kids = pdf.resolver.resolve(node.get("Kids"))
        if kids is None:
            kids = []
        elif not isinstance(kids, (list, tuple)):
            raise ValueError("expected /Kids to be an array")
        ancestors.add(id(node))
        stack.append((node, iter(kids), depth))

    enter(root, 0)
    while stack:
        parent, kids, depth = stack[-1]
        for kid in kids:
            child = pdf.resolver.as_dict(kid)
            if not child:
                continue
            if id(child) in ancestors:
                raise ValueError("detected cyclic page references")
            entries += 1
            if entries > PAGE_TREE_MAXIMUM_ENTRIES:
                raise ValueError(f"maximum page tree entry limit reached: {entries}")
            enter(child, depth + 1)
            break
        else:
            stack.pop()
            ancestors.discard(id(parent))

"""Deterministic content, structure, and citation analysis operations."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from core_pdf.impl.engine.layout.geometry import rect_tuple
from core_pdf.impl.text import collapse_ws, search_key

from ..models import EvidenceLayer, EvidenceRecord, Severity, SourceRef, TextSpan
from ..types import ExecutionContext

if TYPE_CHECKING:
    from ..document import PdfDocument
from .base import AnalysisOperation, FindingCollector, OperationOptions
from .checks import AUTHOR_YEAR_PATTERN, DOI_PATTERN, REFERENCE_SECTION_HEADING_PATTERN


class LayoutAnalysisOperation(AnalysisOperation):
    """Detect repeated page furniture and multi-column layouts deterministically."""

    operation_id = "analysis.layout"

    def _analyze(
        self,
        document: PdfDocument,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        edge_fraction = options.get_float("edge_fraction", 0.15)
        pages = tuple(self._pages(document, context, options))
        edge_occurrences: dict[str, list[tuple[str, int, TextSpan | Any]]] = {}
        column_pages = 0

        for page in pages:
            blocks = tuple(page.text_blocks())
            width = page.info.width
            height = page.info.height
            centers: list[float] = []
            for block in blocks:
                if block.bbox is None:
                    continue
                normalized = search_key(block.text)
                if not normalized:
                    continue
                if block.bbox.y0 <= height * edge_fraction:
                    edge_occurrences.setdefault(normalized, []).append(
                        ("header", page.info.number, block)
                    )
                elif block.bbox.y1 >= height * (1.0 - edge_fraction):
                    edge_occurrences.setdefault(normalized, []).append(
                        ("footer", page.info.number, block)
                    )
                centers.append((block.bbox.x0 + block.bbox.x1) / 2.0)
            centers.sort()
            clusters: list[float] = []
            for center in centers:
                if not clusters or center - clusters[-1] > width * 0.2:
                    clusters.append(center)
            if len(clusters) >= 2:
                column_pages += 1
                out.add(
                    "layout.columns",
                    Severity.INFO,
                    f"Page {page.info.number} has {len(clusters)} text columns.",
                    page=page.info.number,
                    remediation="Use column-aware reading order when creating retrieval chunks.",
                )

        repeated_count = 0
        for text, occurrences in edge_occurrences.items():
            page_numbers = tuple(dict.fromkeys(item[1] for item in occurrences))
            if len(page_numbers) < 2:
                continue
            repeated_count += 1
            kind = occurrences[0][0]
            evidence = tuple(
                EvidenceRecord(
                    layer=EvidenceLayer.STRUCTURED,
                    value=text,
                    bbox=block.bbox,
                    source=block.source,
                    attributes={"role": kind},
                )
                for _, _, block in occurrences
            )
            out.add(
                f"layout.repeated-{kind}",
                Severity.INFO,
                f"Repeated {kind} text appears on {len(page_numbers)} pages.",
                evidence=evidence,
                remediation=(
                    f"Treat this {kind} as page furniture during chunking if appropriate."
                ),
            )
        out.set_metric("page_count", len(pages))
        out.set_metric("column_page_count", column_pages)
        out.set_metric("repeated_edge_text_count", repeated_count)


class StructureAnalysisOperation(AnalysisOperation):
    """Summarize classical document structure already inferred by the parser."""

    operation_id = "analysis.structure"

    def _analyze(
        self,
        document: PdfDocument,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        counts: dict[str, int] = {}
        examples: dict[str, tuple[int, object]] = {}
        for page in self._pages(document, context, options):
            for element in page.structured_view.elements:
                kind = getattr(getattr(element, "kind", None), "value", None)
                if kind is None:
                    kind = type(element).__name__.casefold()
                kind = str(kind)
                counts[kind] = counts.get(kind, 0) + 1
                examples.setdefault(kind, (page.info.number, element))

        for kind, count in sorted(counts.items()):
            page_number, example = examples[kind]
            source = getattr(example, "source", SourceRef(page_number=page_number))
            evidence = EvidenceRecord(
                layer=EvidenceLayer.STRUCTURED,
                value=str(getattr(example, "text", kind)),
                source=(
                    source if isinstance(source, SourceRef) else SourceRef(page_number=page_number)
                ),
                attributes={"kind": kind, "count": count},
            )
            out.add(
                f"structure.{kind}",
                Severity.INFO,
                f"Detected {count} structured {kind} element(s).",
                evidence=(evidence,),
                page=page_number,
                remediation="Retain this element type when building structure-aware outputs.",
            )
        out.set_metric("element_counts", counts)
        out.set_metric("element_count", sum(counts.values()))


class CitationAnalysisOperation(AnalysisOperation):
    """Extract common citation and bibliography signals with page provenance."""

    operation_id = "analysis.citations"
    _numeric = re.compile(r"\[(\d+(?:\s*[,;–-]\s*\d+)*)\]")
    _author_year = AUTHOR_YEAR_PATTERN
    _reference_heading = REFERENCE_SECTION_HEADING_PATTERN

    def _analyze(
        self,
        document: PdfDocument,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        numeric_count = 0
        author_year_count = 0
        reference_sections = 0
        for page in self._pages(document, context, options):
            for block in page.text_blocks():
                text = collapse_ws(block.text)
                if not text:
                    continue
                if self._reference_heading.fullmatch(text):
                    reference_sections += 1
                    out.add(
                        "citation.reference-section",
                        Severity.INFO,
                        "Detected a bibliography or reference-section heading.",
                        evidence=(
                            EvidenceRecord(
                                layer=EvidenceLayer.STRUCTURED,
                                value=text,
                                bbox=block.bbox,
                                source=block.source,
                            ),
                        ),
                        page=page.info.number,
                        bbox=block.bbox,
                        remediation="Associate subsequent reference entries with this section.",
                    )
                numeric_matches = tuple(self._numeric.finditer(text))
                author_matches = tuple(self._author_year.finditer(text))
                numeric_count += len(numeric_matches)
                author_year_count += len(author_matches)
                if numeric_matches or author_matches:
                    labels = tuple(match.group(0) for match in (*numeric_matches, *author_matches))
                    out.add(
                        "citation.inline",
                        Severity.INFO,
                        f"Detected {len(labels)} inline citation signal(s).",
                        evidence=(
                            EvidenceRecord(
                                layer=EvidenceLayer.STRUCTURED,
                                value=" ".join(labels),
                                bbox=block.bbox,
                                source=block.source,
                                attributes={
                                    "numeric": len(numeric_matches),
                                    "author_year": len(author_matches),
                                },
                            ),
                        ),
                        page=page.info.number,
                        bbox=block.bbox,
                        remediation=(
                            "Link the citation signal to a normalized reference entry "
                            "when available."
                        ),
                    )
        out.set_metric("numeric_citation_count", numeric_count)
        out.set_metric("author_year_citation_count", author_year_count)
        out.set_metric("reference_section_count", reference_sections)


class FigureCaptionAnalysisOperation(AnalysisOperation):
    """Associate nearby caption blocks with figures using page geometry."""

    operation_id = "analysis.figure-captions"

    @staticmethod
    def _box(value: object) -> tuple[float, float, float, float] | None:
        return rect_tuple(value)

    def _analyze(
        self,
        document: PdfDocument,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        figure_count = associated_count = 0
        for page in self._pages(document, context, options):
            figures = tuple(page.structured_view.figures)
            captions = tuple(
                element
                for element in page.structured_view.elements
                if str(getattr(getattr(element, "kind", None), "value", "")).casefold() == "caption"
            )
            figure_count += len(figures)
            for figure in figures:
                figure_box = self._box(getattr(figure, "bbox", None))
                if figure_box is None:
                    continue
                candidates: list[tuple[float, Any]] = []
                for caption in captions:
                    caption_box = self._box(getattr(caption, "bbox", None))
                    if caption_box is None:
                        continue
                    horizontal_overlap = min(figure_box[2], caption_box[2]) - max(
                        figure_box[0], caption_box[0]
                    )
                    if horizontal_overlap <= 0:
                        continue
                    distance = min(
                        abs(caption_box[1] - figure_box[3]),
                        abs(figure_box[1] - caption_box[3]),
                    )
                    candidates.append((distance, caption))
                if candidates:
                    _, caption = min(candidates, key=lambda item: item[0])
                    associated_count += 1
                    text = str(getattr(caption, "text", "")).strip()
                    out.add(
                        "figure.caption-associated",
                        Severity.INFO,
                        "Associated a nearby caption with a figure.",
                        evidence=(
                            EvidenceRecord(
                                layer=EvidenceLayer.STRUCTURED,
                                value=text,
                                source=SourceRef(
                                    page_number=page.info.number, stage="figure-caption"
                                ),
                                attributes={"figure_order": getattr(figure, "order", None)},
                            ),
                        ),
                        page=page.info.number,
                        remediation=(
                            "Preserve this association in structured and retrieval outputs."
                        ),
                    )
                else:
                    out.add(
                        "figure.caption-missing",
                        Severity.WARNING,
                        "The figure has no nearby compatible caption.",
                        page=page.info.number,
                        remediation=(
                            "Review the figure for a missing caption or accessibility description."
                        ),
                    )
        out.set_metric("figure_count", figure_count)
        out.set_metric("associated_figure_count", associated_count)
        out.set_metric("uncaptioned_figure_count", figure_count - associated_count)


class SectionHierarchyAnalysisOperation(AnalysisOperation):
    """Build deterministic heading/list hierarchy signals from structured blocks."""

    operation_id = "analysis.section-hierarchy"

    def _analyze(
        self,
        document: PdfDocument,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        heading_count = list_count = 0
        max_depth = 0
        previous_depth = 0
        skipped_depth_count = 0
        for page in self._pages(document, context, options):
            for block in page.structured_view.blocks:
                kind = str(getattr(getattr(block, "kind", None), "value", "")).casefold()
                if kind == "list":
                    list_count += 1
                    continue
                if kind != "heading":
                    continue
                heading_count += 1
                raw_level = getattr(block, "level", None)
                depth = int(raw_level) if isinstance(raw_level, int) and raw_level > 0 else 1
                max_depth = max(max_depth, depth)
                if previous_depth and depth > previous_depth + 1:
                    skipped_depth_count += 1
                previous_depth = depth
                out.add(
                    "structure.heading",
                    Severity.INFO,
                    f"Detected a level-{depth} heading.",
                    evidence=(
                        EvidenceRecord(
                            layer=EvidenceLayer.STRUCTURED,
                            value=str(getattr(block, "text", "")),
                            source=SourceRef(
                                page_number=page.info.number,
                                stage="section-hierarchy",
                            ),
                            attributes={"depth": depth, "order": getattr(block, "order", None)},
                        ),
                    ),
                    page=page.info.number,
                    remediation=("Preserve heading depth when constructing section-aware outputs."),
                )
        if skipped_depth_count:
            out.add(
                "structure.heading-depth-gap",
                Severity.WARNING,
                f"Detected {skipped_depth_count} heading depth jump(s).",
                remediation="Review heading nesting for a coherent document hierarchy.",
            )
        out.set_metric("heading_count", heading_count)
        out.set_metric("list_count", list_count)
        out.set_metric("max_heading_depth", max_depth)
        out.set_metric("heading_depth_gap_count", skipped_depth_count)


class IdentifierAnalysisOperation(AnalysisOperation):
    """Extract deterministic identifiers and contact-like entities from text."""

    operation_id = "analysis.identifiers"
    _patterns = {
        "doi": DOI_PATTERN,
        "isbn": re.compile(r"\b(?:ISBN(?:-1[03])?:?\s*)?(?:\d[ -]?){9,16}\d\b", re.I),
        "url": re.compile(r"https?://[^\s<>]+", re.I),
        "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
        "date": re.compile(
            r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
            r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
            r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b",
            re.I,
        ),
    }

    def _analyze(
        self,
        document: PdfDocument,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        counts = dict.fromkeys(self._patterns, 0)
        for page in self._pages(document, context, options):
            for block in page.text_blocks():
                text = block.text
                for kind, pattern in self._patterns.items():
                    matches = tuple(
                        match.group(0).rstrip(".,;)") for match in pattern.finditer(text)
                    )
                    if not matches:
                        continue
                    counts[kind] += len(matches)
                    out.add(
                        f"identifier.{kind}",
                        Severity.INFO,
                        f"Detected {len(matches)} {kind} identifier(s).",
                        evidence=(
                            EvidenceRecord(
                                layer=EvidenceLayer.STRUCTURED,
                                value=" ".join(matches),
                                bbox=block.bbox,
                                source=block.source,
                                attributes={"identifier_type": kind},
                            ),
                        ),
                        page=page.info.number,
                        bbox=block.bbox,
                        remediation=(
                            "Normalize this identifier while retaining the exact source spelling."
                        ),
                    )
        for kind, count in counts.items():
            out.set_metric(kind, count)
        out.set_metric("identifier_count", sum(counts.values()))


class ReferenceEntryAnalysisOperation(AnalysisOperation):
    """Normalize common numbered bibliography entries without discarding source text."""

    operation_id = "analysis.reference-entries"
    _heading = REFERENCE_SECTION_HEADING_PATTERN
    _number = re.compile(r"^(?:\[(\d+)\]|(\d+)[.)])\s+(.*)$")
    _author_year = AUTHOR_YEAR_PATTERN
    _doi = DOI_PATTERN

    def _analyze(
        self,
        document: PdfDocument,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        active = False
        entries = numbered = doi_entries = 0
        for page in self._pages(document, context, options):
            for block in page.text_blocks():
                text = collapse_ws(block.text)
                if self._heading.fullmatch(text):
                    active = True
                    continue
                if not active or not text:
                    continue
                match = self._number.match(text)
                if match is None:
                    continue
                entries += 1
                number = match.group(1) or match.group(2)
                body = match.group(3).strip()
                numbered += 1
                doi = self._doi.search(body)
                author_year = self._author_year.search(body)
                if doi:
                    doi_entries += 1
                out.add(
                    "reference.entry",
                    Severity.INFO,
                    f"Detected numbered reference entry {number}.",
                    evidence=(
                        EvidenceRecord(
                            layer=EvidenceLayer.STRUCTURED,
                            value=text,
                            source=block.source,
                            bbox=block.bbox,
                            attributes={
                                "number": int(number),
                                "doi": doi.group(0) if doi else None,
                                "author_year": author_year.group(0) if author_year else None,
                            },
                        ),
                    ),
                    page=page.info.number,
                    bbox=block.bbox,
                    remediation=("Normalize fields while preserving the original reference entry."),
                )
        out.set_metric("reference_entry_count", entries)
        out.set_metric("numbered_reference_count", numbered)
        out.set_metric("doi_reference_count", doi_entries)


__all__ = (
    "CitationAnalysisOperation",
    "FigureCaptionAnalysisOperation",
    "IdentifierAnalysisOperation",
    "LayoutAnalysisOperation",
    "ReferenceEntryAnalysisOperation",
    "SectionHierarchyAnalysisOperation",
    "StructureAnalysisOperation",
)

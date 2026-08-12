"""Forensic trust, layer-consistency, and redaction-failure operations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from ..models import (
    Drawing,
    EvidenceLayer,
    EvidenceRecord,
    Rect,
    Severity,
    SourceRef,
    TextSpan,
)
from ..protocols import ExecutionContext, PdfDocumentProtocol
from .base import AnalysisOperation, FindingCollector, OperationOptions


def _is_uniform(raster: Any) -> bool:
    pixels = memoryview(raster.data).cast("B")
    channels = raster.channels
    if not pixels or channels <= 0:
        return False
    first = pixels[:channels].tobytes()
    return all(
        pixels[index : index + channels].tobytes() == first
        for index in range(0, len(pixels), channels)
    )


def _candidate_rectangles(drawing: Drawing) -> Iterable[Rect]:
    """Yield rectangle items, falling back to the drawing bounds if unavailable."""
    found = False
    for item in drawing.items:
        if item.kind != "re":
            continue
        if item.bbox is not None:
            found = True
            yield item.bbox
            continue
        raw = item.data.get("raw")
        if isinstance(raw, tuple) and len(raw) > 1:
            value = raw[1]
            if isinstance(value, (tuple, list)) and len(value) == 4 and drawing.bbox is not None:
                found = True
                raw_value = cast(Any, value)
                x0, y0, x1, y1 = (float(part) for part in raw_value)
                yield Rect(
                    x0,
                    y0,
                    x1,
                    y1,
                    drawing.bbox.space,
                )
            elif all(hasattr(value, name) for name in ("x0", "y0", "x1", "y1")):
                found = True
                if drawing.bbox is not None:
                    raw_rect = cast(Any, value)
                    yield Rect(
                        float(raw_rect.x0),
                        float(raw_rect.y0),
                        float(raw_rect.x1),
                        float(raw_rect.y1),
                        drawing.bbox.space,
                    )
    # Very thin filled paths are rules/highlights rather than redaction
    # rectangles; the source engine may expose them without item metadata.
    if (
        not found
        and drawing.bbox is not None
        and drawing.bbox.height >= 5.0
        and drawing.bbox.width >= 5.0
    ):
        yield drawing.bbox


def _overlap_ratio(text: TextSpan, rectangle: Rect) -> float:
    return text.bbox.overlap_ratio_min(rectangle)


def _covered_characters(text: TextSpan, rectangle: Rect, threshold: float) -> str:
    """Approximate character boxes when the adapter only exposes text runs."""
    if text.characters:
        return "".join(
            character.text
            for character in text.characters
            if _overlap_ratio(TextSpan(character.text, character.bbox), rectangle) >= threshold
        )
    if not text.text:
        return ""
    width = text.bbox.width / len(text.text)
    covered: list[str] = []
    for index, character in enumerate(text.text):
        char_box = Rect(
            text.bbox.x0 + width * index,
            text.bbox.y0,
            text.bbox.x0 + width * (index + 1),
            text.bbox.y1,
            text.bbox.space,
        )
        if _overlap_ratio(TextSpan(character, char_box), rectangle) >= threshold:
            covered.append(character)
    return "".join(covered)


class BadRedactionOperation(AnalysisOperation):
    """Find opaque filled rectangles that conceal selectable text.

    This is intentionally conservative: candidates must be filled and opaque,
    text must overlap substantially, and optional raster inspection must find a
    uniform crop. It is a reusable local operation rather than a document method.
    """

    operation_id = "analysis.bad-redactions"

    def _analyze(
        self,
        document: PdfDocumentProtocol,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        # Run-level geometry is converted to approximate character boxes; a
        # slightly lower default compensates for fonts whose glyph widths do
        # not match evenly partitioned text runs.
        overlap_threshold = options.get_float("overlap_threshold", 0.5)
        inspect_raster = options.get_bool("inspect_raster", True)
        min_fill_opacity = options.get_float("min_fill_opacity", 1.0)
        for page in self._pages(document, context, options):
            for drawing in page.drawings():
                if (
                    drawing.fill is None
                    or drawing.fill_opacity is None
                    or drawing.fill_opacity < min_fill_opacity
                ):
                    continue
                for rectangle in _candidate_rectangles(drawing):
                    covered_text = "".join(
                        _covered_characters(span, rectangle, overlap_threshold)
                        for span in page.text_spans()
                        if span.sequence is not None
                        # The parser's fused content stream can place the text
                        # record immediately after the filled path even when
                        # the source PDF paints the text first.  Keep a small
                        # adjacency window while retaining ordering as a
                        # useful signal for unrelated content.
                        and (span.sequence <= drawing.sequence + 1 or drawing.sequence == 0)
                    )
                    text = covered_text.strip()
                    if not text:
                        continue
                    if inspect_raster:
                        raster = page.render(
                            dpi=options.get_float("dpi", 72.0),
                            crop=(rectangle.x0, rectangle.y0, rectangle.x1, rectangle.y1),
                        )
                        if not _is_uniform(raster):
                            continue
                    out.add(
                        "redaction.text-under-fill",
                        Severity.ERROR,
                        "An opaque filled rectangle conceals selectable text.",
                        page=page.info.number,
                        bbox=rectangle,
                        evidence=(
                            EvidenceRecord(
                                layer=EvidenceLayer.NATIVE_TEXT,
                                value=text,
                                source=drawing.source,
                                attributes={
                                    "rectangle_sequence": drawing.sequence,
                                    "text_sequences": tuple(
                                        span.sequence
                                        for span in page.text_spans()
                                        if span.sequence is not None
                                        and _covered_characters(span, rectangle, overlap_threshold)
                                    ),
                                },
                            ),
                        ),
                        remediation=("Apply true content redaction and reserialize the document."),
                    )


class LayerConsistencyOperation(AnalysisOperation):
    """Find deterministic contradictions in the native text evidence layer."""

    operation_id = "analysis.layer-consistency"

    def _analyze(
        self,
        document: PdfDocumentProtocol,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        out.set_metric("invisible_text_runs", 0)
        for page in self._pages(document, context, options):
            for run in page.text_diagnostics(include_invisible=True):
                if run.visible:
                    continue
                out.count("invisible_text_runs")
                evidence = EvidenceRecord(
                    layer=EvidenceLayer.NATIVE_TEXT,
                    value=run.text,
                    bbox=run.bbox,
                    confidence=1.0,
                    attributes={"sequence": run.sequence, "visible": False},
                    source=SourceRef(
                        page_index=page.info.index,
                        page_number=page.info.number,
                        sequence=run.sequence,
                        stage="text-diagnostics",
                    ),
                )
                out.add(
                    "text.invisible",
                    Severity.WARNING,
                    "The page contains text marked invisible by its content state.",
                    evidence=(evidence,),
                    page=page.info.number,
                    bbox=run.bbox,
                    confidence=1.0,
                    remediation="Review or remove the hidden text layer before distribution.",
                )


class ForensicAnalysisOperation(AnalysisOperation):
    """Report deterministic PDF-native trust and security signals."""

    operation_id = "analysis.forensics"

    def _analyze(
        self,
        document: PdfDocumentProtocol,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        context.cancellation.raise_if_cancelled()
        inventory = document.inventory()
        graph = document.object_graph()

        if inventory.xref_recovered:
            out.add(
                "pdf.xref-recovered",
                Severity.WARNING,
                "The parser recovered the cross-reference data instead of reading it cleanly.",
                remediation=(
                    "Repair and reserialize the PDF before relying on object-level conclusions."
                ),
            )
        if inventory.encrypted:
            out.add(
                "pdf.encrypted",
                Severity.INFO,
                "The document uses PDF encryption.",
                remediation=(
                    "Record the authorization context and preserve encryption requirements."
                ),
            )
        if inventory.has_javascript:
            out.add(
                "pdf.javascript",
                Severity.ERROR,
                "The document contains JavaScript actions or names.",
                remediation="Remove executable actions or review them in an isolated environment.",
            )
        if inventory.has_open_action:
            out.add(
                "pdf.actions",
                Severity.WARNING,
                "The document contains an automatic or additional action.",
                remediation="Review automatic actions before distribution or sanitization.",
            )
        if inventory.has_attachments:
            out.add(
                "pdf.embedded-files",
                Severity.WARNING,
                "The document contains embedded files.",
                remediation="Inventory and scan embedded files before sharing the PDF.",
            )
        for object_number in graph.unreachable_objects:
            context.cancellation.raise_if_cancelled()
            out.add(
                "pdf.unreachable-object",
                Severity.WARNING,
                f"Object {object_number} is in-use but unreachable from the trailer root.",
                evidence=(
                    EvidenceRecord(
                        layer=EvidenceLayer.PDF_OBJECT,
                        value=str(object_number),
                        source=SourceRef(
                            object_number=object_number,
                            stage="object-graph",
                        ),
                    ),
                ),
                remediation="Remove unreachable objects when producing a sanitized copy.",
            )
        out.set_metric("object_count", inventory.object_count)
        out.set_metric("unreachable_object_count", len(graph.unreachable_objects))
        out.set_metric("xref_recovered", inventory.xref_recovered)


__all__ = (
    "BadRedactionOperation",
    "ForensicAnalysisOperation",
    "LayerConsistencyOperation",
)

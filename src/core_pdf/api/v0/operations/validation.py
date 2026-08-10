"""Local validation operations over document structure and page records."""

from __future__ import annotations

from urllib.parse import urlparse

from ..models import EvidenceLayer, EvidenceRecord, Severity, SourceRef
from ..protocols import ExecutionContext, PdfDocumentProtocol
from .base import AnalysisOperation, FindingCollector, OperationOptions
from .checks import is_interactive_annotation


class AccessibilityValidationOperation(AnalysisOperation):
    """Validate high-level PDF/UA signals available from the local document model."""

    operation_id = "analysis.accessibility"
    emit_finding_count = True

    def _analyze(
        self,
        document: PdfDocumentProtocol,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        context.cancellation.raise_if_cancelled()
        inventory = document.accessibility_inventory()

        def add(
            code: str,
            message: str,
            *,
            severity: Severity = Severity.ERROR,
            page: int | None = None,
        ) -> None:
            out.add(
                code,
                severity,
                message,
                page=page,
                remediation="Repair the document structure and rerun accessibility validation.",
                evidence=(
                    EvidenceRecord(
                        layer=EvidenceLayer.STRUCTURED,
                        value=message,
                        source=SourceRef(page_number=page, stage="accessibility-validation"),
                    ),
                ),
            )

        if not inventory.has_title:
            add("ua.document-title", "The document has no descriptive title.")
        if not inventory.document_language:
            add("ua.document-language", "The document language is not declared.")
        if not inventory.tagged_element_count:
            add("ua.tagged-structure", "The document has no tagged logical structure.")
        if inventory.missing_alternate_text_count:
            add(
                "ua.figure-alternate-text",
                f"{inventory.missing_alternate_text_count} figure(s) lack alternate text.",
            )

        headings: list[tuple[int, int]] = []
        for element in document.structure_elements():
            role = element.role.casefold()
            if role.startswith("h") and role[1:].isdigit():
                headings.append((int(role[1:]), element.page_number or 0))
        for previous, current in zip(headings, headings[1:]):
            if current[0] > previous[0] + 1:
                add(
                    "ua.heading-level-skipped",
                    f"Heading level jumps from H{previous[0]} to H{current[0]}.",
                    page=current[1] or None,
                )
        structure_tables = sum(
            element.role.casefold() in {"table", "tbl"} for element in document.structure_elements()
        )
        if inventory.table_count and structure_tables < inventory.table_count:
            add(
                "ua.table-structure-missing",
                "One or more detected tables have no corresponding tagged Table element.",
            )
        for page in self._pages(document, context, options):
            for table in page.tables():
                if table.rows <= 0 or table.columns <= 0 or not table.cells:
                    add(
                        "ua.table-grid-incomplete",
                        "A table has no complete row, column, or cell grid.",
                        page=page.info.number,
                    )
        out.set_metric("tagged_element_count", inventory.tagged_element_count)
        out.set_metric("figure_count", inventory.figure_count)
        out.set_metric("table_count", inventory.table_count)


class FormValidationOperation(AnalysisOperation):
    """Validate high-level AcroForm fields before local editing or export."""

    operation_id = "analysis.forms"

    _known_types = frozenset(
        {
            "text",
            "tx",
            "button",
            "btn",
            "checkbox",
            "choice",
            "ch",
            "combobox",
            "listbox",
            "signature",
            "sig",
        }
    )
    _choice_types = frozenset({"choice", "ch", "combobox", "listbox"})

    def _analyze(
        self,
        document: PdfDocumentProtocol,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        inventory = document.form_inventory()
        seen: dict[str, int] = {}
        for page in self._pages(document, context, options):
            for form_field in page.form_fields():
                seen[form_field.name] = seen.get(form_field.name, 0) + 1
                if not form_field.name:
                    out.add(
                        "form.name-missing",
                        Severity.ERROR,
                        "A form field has no stable name.",
                        page=page.info.number,
                        remediation="Assign a unique, stable field name.",
                    )
                if not form_field.field_type:
                    out.add(
                        "form.type-missing",
                        Severity.ERROR,
                        f"Form field {form_field.name or '<unnamed>'} has no field type.",
                        page=page.info.number,
                        remediation="Set the field type before writing the document.",
                    )
                normalized_type = form_field.field_type.casefold()
                if form_field.required and not form_field.value:
                    out.add(
                        "form.required-empty",
                        Severity.ERROR,
                        f"Required form field {form_field.name} is empty.",
                        page=page.info.number,
                        remediation="Populate the field or remove its required flag.",
                    )
                if normalized_type not in self._known_types:
                    out.add(
                        "form.type-unknown",
                        Severity.WARNING,
                        (
                            f"Form field {form_field.name or '<unnamed>'} uses unknown "
                            f"type {form_field.field_type!r}."
                        ),
                        page=page.info.number,
                        remediation="Use a supported AcroForm field type.",
                    )
                if (
                    form_field.options
                    and normalized_type in self._choice_types
                    and form_field.value not in form_field.options
                ):
                    out.add(
                        "form.value-not-in-options",
                        Severity.ERROR,
                        f"Form field {form_field.name} has a value outside its options.",
                        page=page.info.number,
                        remediation="Choose one of the declared field options.",
                    )
                if form_field.bbox is None:
                    out.add(
                        "form.geometry-missing",
                        Severity.WARNING,
                        (f"Form field {form_field.name or '<unnamed>'} has no widget geometry."),
                        page=page.info.number,
                        remediation=(
                            "Provide a visible widget rectangle or mark the field non-visual."
                        ),
                    )
        for name in sorted(name for name, count in seen.items() if count > 1 and name):
            out.add(
                "form.duplicate-name",
                Severity.WARNING,
                f"Form field name {name!r} occurs {seen[name]} times.",
                remediation="Use intentional field hierarchies or unique names.",
            )
        out.set_metric("field_count", inventory.field_count)
        out.set_metric("populated_count", inventory.populated_count)
        out.set_metric("empty_count", inventory.empty_count)
        out.set_metric("duplicate_name_count", len(inventory.duplicate_names))


class LinkValidationOperation(AnalysisOperation):
    """Validate local link records and external URI destinations."""

    operation_id = "analysis.links"
    emit_finding_count = True

    def _analyze(
        self,
        document: PdfDocumentProtocol,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        out.set_metric("link_count", 0)
        out.set_metric("external_link_count", 0)
        for page in self._pages(document, context, options):
            for link in page.links():
                out.count("link_count")
                if link.bbox is None:
                    out.add(
                        "link.geometry-missing",
                        Severity.WARNING,
                        "A link has no clickable geometry.",
                        page=page.info.number,
                        remediation="Assign a visible link rectangle.",
                    )
                if link.url is None or not link.url.strip():
                    out.add(
                        "link.destination-missing",
                        Severity.ERROR,
                        "A link has no destination.",
                        page=page.info.number,
                        remediation="Set an internal destination or an external URI.",
                    )
                    continue
                out.count("external_link_count")
                parsed = urlparse(link.url)
                if parsed.scheme not in {"http", "https", "mailto", "ftp"}:
                    out.add(
                        "link.uri-invalid",
                        Severity.WARNING,
                        f"Link URI uses unsupported scheme: {parsed.scheme or '<none>'}.",
                        page=page.info.number,
                        remediation="Use a valid absolute URI or an internal PDF destination.",
                    )
                elif parsed.scheme in {"http", "https", "ftp"} and not parsed.netloc:
                    out.add(
                        "link.uri-host-missing",
                        Severity.ERROR,
                        "External link URI has no host.",
                        page=page.info.number,
                        remediation="Provide a complete external URI.",
                    )


class AttachmentValidationOperation(AnalysisOperation):
    """Validate embedded files and attachment metadata exposed by the local API."""

    operation_id = "analysis.attachments"

    def _analyze(
        self,
        document: PdfDocumentProtocol,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        resources = tuple(document.embedded_resources())
        names: dict[str, int] = {}
        for resource in resources:
            context.cancellation.raise_if_cancelled()
            names[resource.filename] = names.get(resource.filename, 0) + 1
            if resource.byte_count <= 0:
                out.add(
                    "attachment.empty",
                    Severity.WARNING,
                    f"Attachment {resource.filename!r} has an empty payload.",
                    remediation="Remove the attachment or provide a non-empty payload.",
                )
            if not resource.filename.strip():
                out.add(
                    "attachment.filename-missing",
                    Severity.ERROR,
                    "An embedded resource has no filename.",
                    remediation="Assign a stable filename before export.",
                )
        for filename, count in sorted(names.items()):
            if filename and count > 1:
                out.add(
                    "attachment.duplicate-filename",
                    Severity.WARNING,
                    f"Attachment filename {filename!r} occurs {count} times.",
                    remediation="Use unique attachment filenames or intentional versioning.",
                )
        out.set_metric("attachment_count", len(resources))
        out.set_metric("empty_count", sum(resource.byte_count <= 0 for resource in resources))
        out.set_metric("duplicate_filename_count", sum(count > 1 for count in names.values()))


class FontValidationOperation(AnalysisOperation):
    """Validate page font resource resolution and text decoding signals."""

    operation_id = "analysis.fonts"
    emit_finding_count = True

    def _analyze(
        self,
        document: PdfDocumentProtocol,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        out.set_metric("page_count", 0)
        out.set_metric("font_count", 0)
        for page in self._pages(document, context, options):
            resource = next(
                (
                    item
                    for item in document.resource_inventory(pages=(page.info.number,))
                    if item.page_number == page.info.number
                ),
                None,
            )
            out.count("page_count")
            if resource is None:
                continue
            out.count("font_count", len(resource.font_names))
            if resource.has_fonts and not resource.font_names:
                out.add(
                    "font.names-missing",
                    Severity.ERROR,
                    "A page uses fonts but exposes no resolved font names.",
                    page=page.info.number,
                    remediation="Repair the page font resource dictionary and encoding.",
                )
        for diagnostic in document.resource_diagnostics():
            if "font" not in diagnostic.code.casefold():
                continue
            out.add(
                f"font.{diagnostic.code}",
                diagnostic.severity,
                diagnostic.message,
                page=diagnostic.page_number,
                remediation="Repair the font resource or provide a usable Unicode mapping.",
            )


class ImageValidationOperation(AnalysisOperation):
    """Validate image records for usable geometry and placement."""

    operation_id = "analysis.images"
    emit_finding_count = True

    def _analyze(
        self,
        document: PdfDocumentProtocol,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        out.set_metric("image_count", 0)
        out.set_metric("invalid_count", 0)
        for page in self._pages(document, context, options):
            for image in page.images():
                out.count("image_count")
                if image.width <= 0 or image.height <= 0 or image.channels <= 0:
                    out.count("invalid_count")
                    out.add(
                        "image.geometry-invalid",
                        Severity.ERROR,
                        "Image has invalid dimensions or channel count.",
                        page=page.info.number,
                        bbox=image.bbox,
                        remediation="Replace the image with a decodable raster resource.",
                    )
                if image.bbox is None:
                    out.add(
                        "image.geometry-missing",
                        Severity.WARNING,
                        "Image has no placement geometry.",
                        page=page.info.number,
                        remediation="Provide image placement geometry or remove the resource.",
                    )


class GeometryValidationOperation(AnalysisOperation):
    """Promote page geometry diagnostics into a document analysis report."""

    operation_id = "analysis.geometry"

    def _analyze(
        self,
        document: PdfDocumentProtocol,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        issue_counts: dict[str, int] = {}
        repairable_count = 0
        for page in self._pages(document, context, options):
            for issue in page.geometry_issues():
                issue_counts[issue.code] = issue_counts.get(issue.code, 0) + 1
                repairable_count += issue.repairable
                out.add(
                    f"geometry.{issue.code}",
                    issue.severity,
                    issue.message or issue.subject,
                    page=page.info.number,
                    bbox=issue.bbox,
                    remediation=(
                        "Normalize the affected geometry before layout, table, or "
                        "reading-order analysis."
                        if issue.repairable
                        else "Inspect the source geometry and choose an explicit recovery policy."
                    ),
                )
        out.set_metric("issue_count", len(out.findings))
        out.set_metric("repairable_count", repairable_count)
        out.set_metric("issue_codes", issue_counts)


class AnnotationValidationOperation(AnalysisOperation):
    """Validate annotation records for usable and accessible interaction."""

    operation_id = "analysis.annotations"
    emit_finding_count = True

    def _analyze(
        self,
        document: PdfDocumentProtocol,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        out.set_metric("annotation_count", 0)
        for page in self._pages(document, context, options):
            for annotation in page.annotations():
                out.count("annotation_count")
                if not annotation.subtype.strip():
                    out.add(
                        "annotation.subtype-missing",
                        Severity.ERROR,
                        "Annotation has no subtype.",
                        page=page.info.number,
                        remediation="Set a valid annotation subtype.",
                    )
                if annotation.bbox is None:
                    out.add(
                        "annotation.geometry-missing",
                        Severity.WARNING,
                        "Annotation has no interaction geometry.",
                        page=page.info.number,
                        remediation="Provide an annotation rectangle or remove the annotation.",
                    )
                if is_interactive_annotation(annotation) and not (
                    annotation.contents.strip() or annotation.destination
                ):
                    out.add(
                        "annotation.description-missing",
                        Severity.WARNING,
                        (f"{annotation.subtype} annotation has no description or destination."),
                        page=page.info.number,
                        remediation="Add accessible text or a valid destination.",
                    )


class DocumentIntegrityOperation(AnalysisOperation):
    """Validate document-level structural recovery and action signals."""

    operation_id = "analysis.document-integrity"
    emit_finding_count = True

    def _analyze(
        self,
        document: PdfDocumentProtocol,
        context: ExecutionContext,
        options: OperationOptions,
        out: FindingCollector,
    ) -> None:
        context.cancellation.raise_if_cancelled()
        inventory = document.inventory()
        actions = document.action_inventory()
        if inventory.page_count == 0:
            out.add(
                "document.pages-empty",
                Severity.ERROR,
                "The document contains no pages.",
                remediation="Provide a valid page tree before extraction or export.",
            )
        if inventory.xref_recovered:
            out.add(
                "document.xref-recovered",
                Severity.WARNING,
                "The cross-reference table required recovery.",
                remediation=(
                    "Repair the source PDF and preserve the recovery diagnostic in provenance."
                ),
            )
        if inventory.page_tree_recovered:
            out.add(
                "document.page-tree-recovered",
                Severity.WARNING,
                "The page tree required recovery.",
                remediation="Repair the source page tree before relying on page-level provenance.",
            )
        if inventory.has_javascript:
            out.add(
                "document.javascript-present",
                Severity.WARNING,
                "The document contains JavaScript actions.",
                remediation=(
                    "Remove or explicitly quarantine executable actions for untrusted workflows."
                ),
            )
        if inventory.has_open_action:
            out.add(
                "document.open-action-present",
                Severity.WARNING,
                "The document contains an automatic open or additional action.",
                remediation="Review or remove automatic actions before sanitization or archival.",
            )
        out.set_metric("page_count", inventory.page_count)
        out.set_metric("object_count", inventory.object_count)
        out.set_metric("action_count", actions.action_count)


__all__ = (
    "AccessibilityValidationOperation",
    "AnnotationValidationOperation",
    "AttachmentValidationOperation",
    "DocumentIntegrityOperation",
    "FontValidationOperation",
    "FormValidationOperation",
    "GeometryValidationOperation",
    "ImageValidationOperation",
    "LinkValidationOperation",
)

# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 14.13: associated files, the AF arrays linking files to PDF objects.

PDF 2.0 added AF. PDF/A-3 (ISO 19005-3) allows it in PDF 1.7 files too, which is how hybrid
invoices such as ZUGFeRD and Factur-X carry their XML, so a reader takes AF whatever the
version a file declares.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfValueResolver
from core_pdf.impl.spec.standards import PdfVersion, SemanticContext

# ISO 32000-2 Table 43: the AFRelationship values. Other names are second-class names.
AF_RELATIONSHIPS = (
    "Source",
    "Data",
    "Alternative",
    "Supplement",
    "EncryptedPayload",
    "FormData",
    "Schema",
    "Unspecified",
)
DEFAULT_AF_RELATIONSHIP = "Unspecified"


def raise_malformed(message: str) -> None:
    raise ValueError(message)


@cites("ISO 32000-2", "14.13", introduced="2.0", rule="AF, an object's associated files")
@cites("ISO 32000-2", "14.13", profile="pdfa-3b", rule="PDF/A-3 allows AF in PDF 1.7 files")
def associated_file_specifications(
    resolver: PdfValueResolver,
    owner: PdfDict,
    *,
    on_malformed: Callable[[str], None] = raise_malformed,
) -> list[PdfDict]:
    """The file specification dictionaries of owner's AF array, in order.

    An AF that is not an array, or an entry that is not a dictionary, is reported to
    on_malformed, which raises ValueError unless the caller passes one that returns.
    """
    value = resolver.resolve(owner.get("AF"))
    if value is None:
        return []
    if isinstance(value, dict):
        # A lone dictionary in place of a one-element array.
        on_malformed("invalid AF array")
        return [value]
    if not isinstance(value, list):
        on_malformed("invalid AF array")
        return []
    specifications: list[PdfDict] = []
    for entry in value:
        specification = resolver.resolve(entry)
        if isinstance(specification, dict):
            specifications.append(specification)
        else:
            on_malformed("invalid AF file specification")
    return specifications


@cites(
    "ISO 32000-2",
    "7.11.3",
    introduced="2.0",
    rule="AFRelationship, how an associated file relates to its object; Unspecified by default",
)
def associated_file_relationship(resolver: PdfValueResolver, specification: PdfDict) -> str:
    value = specification.get("AFRelationship")
    if value is None:
        return DEFAULT_AF_RELATIONSHIP
    name = resolver.resolve_name(value)
    if name is None:
        raise ValueError("invalid AFRelationship")
    return name


# The PDF/A parts that allow AF: PDF/A-3 in PDF 1.7 files, and PDF/A-4, which is PDF 2.0.
AF_PROFILE_PREFIXES = ("pdfa-3", "pdfa-4")


@cites("ISO 32000-2", "14.13.6", since="2.0", rule="structure elements hold AF")
@cites("ISO 32000-2", "14.13.6", profile="pdfa-3b", rule="PDF/A-3 structure elements hold AF")
def structure_elements_hold_af(
    context: SemanticContext | None, profiles: Iterable[str | None] = ()
) -> bool:
    """Whether a structure element of this document may carry AF.

    True from PDF 2.0, in a file claiming PDF/A-3 or PDF/A-4, and when the version is unknown.
    """
    version = None if context is None else context.version
    if version is None or version >= PdfVersion(2, 0):
        return True
    return any(profile and profile.startswith(AF_PROFILE_PREFIXES) for profile in profiles)


__all__ = (
    "AF_PROFILE_PREFIXES",
    "AF_RELATIONSHIPS",
    "DEFAULT_AF_RELATIONSHIP",
    "associated_file_relationship",
    "associated_file_specifications",
    "structure_elements_hold_af",
)

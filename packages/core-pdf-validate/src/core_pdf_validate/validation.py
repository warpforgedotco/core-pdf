# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from core_pdf_validate.models import ProfileResult, ValidationBackend, ValidationReport
from core_pdf_validate.verapdf import VeraPdfBackend

__all__ = ["validate"]


def declared_targets(source: bytes) -> tuple[tuple[str, ...], tuple[str, ...]]:
    from core_pdf import PdfDocument, PdfError

    try:
        with PdfDocument(source) as document:
            standards = document.standards
    except (PdfError, ValueError, OSError) as error:
        return (), (f"declaration_read_failed: {error}", "no_target_identified")

    targets: list[str] = []
    diagnostics = [f"{item.code}: {item.message} ({item.source})" for item in standards.diagnostics]
    for claim in standards.profile_claims:
        if claim.identifier is None:
            diagnostics.append(f"unresolved_profile_claim: {claim.family} ({claim.source})")
        elif claim.identifier not in targets:
            targets.append(claim.identifier)
    if not targets:
        diagnostics.append("no_target_identified")
    return tuple(targets), tuple(diagnostics)


def validate(
    source: str | os.PathLike[str] | bytes,
    *,
    profiles: str | Sequence[str],
    backend: ValidationBackend | None = None,
) -> ValidationReport:
    if profiles == "declared":
        targets: tuple[str, ...] = ()
    else:
        requested = (profiles,) if isinstance(profiles, str) else tuple(profiles)
        if not requested or any(
            not isinstance(target, str) or not target or target == "declared"
            for target in requested
        ):
            raise ValueError("profiles must contain canonical IDs, or be the string 'declared'")
        targets = tuple(dict.fromkeys(requested))
    original = source if isinstance(source, bytes) else Path(source).read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    diagnostics: tuple[str, ...] = ()
    if profiles == "declared":
        targets, diagnostics = declared_targets(original)
    if not targets:
        return ValidationReport(digest, (), (), diagnostics)

    selected = backend if backend is not None else VeraPdfBackend()
    supported = {profile.identifier for profile in selected.supported_profiles}
    results: list[ProfileResult] = []
    with TemporaryDirectory(prefix="core-pdf-validate-") as temporary:
        snapshot = Path(temporary) / "source.pdf"
        for target in targets:
            if target not in supported:
                results.append(
                    ProfileResult(
                        target,
                        "unsupported_profile",
                        "not_checked",
                        selected.name,
                        diagnostics=(f"Unsupported validation target: {target}",),
                    )
                )
                continue
            snapshot.write_bytes(original)
            result = selected.validate(snapshot, profile=target)
            if result.profile != target:
                raise ValueError("Backend result profile does not match the requested target")
            results.append(result)
    return ValidationReport(digest, targets, tuple(results), diagnostics)

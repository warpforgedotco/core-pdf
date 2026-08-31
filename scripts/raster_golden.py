"""Versioned, cross-platform raster-golden support and regeneration logic.

Ordinary pages retain exact RGBA SHA-256 digests. Pages whose irreversible
JPEG 2000 decode is known to differ across CPU implementations retain a
canonical lossless PNG plus measured RGB error limits::

    uv run python scripts/update_raster_golden.py

Reference rasters must normally be generated with the pinned Ubuntu x86_64
codec environment. ``--allow-noncanonical-write`` is an explicitly destructive
escape hatch for bootstrapping on a different platform; the manifest records
that provenance and never adjusts tolerance limits automatically. A
noncanonical rewrite intentionally fails the canonical-provenance test until a
canonical regeneration replaces it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Iterable

import imagecodecs
import numpy

from core_pdf import PdfDocument
from core_pdf.impl.render.raster_image import RasterImage
from core_pdf.impl.spec.s_07_filters.decode_spec import normalize_stream_decode_spec
from core_pdf.impl.spec.s_07_filters.decoders import decode_jpx_image
from core_pdf.impl.spec.s_07_filters.jpeg2000 import (
    internal_jpx_uses_irreversible_wavelet,
)
from core_pdf.impl.spec.s_07_filters.pipeline import decode_one_filter
from core_pdf.impl.spec.s_07_filters.registry import FILTER_DESCRIPTOR_BY_NAME
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf.impl.spec.s_07_syntax_primitives.pdfdict import lookup_dict_key

internal_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = internal_ROOT / "tests" / "fixtures" / "SCORE-Bench" / "src"
SNAPSHOT = internal_ROOT / "tests" / "snapshots" / "raster" / "first_page_scale1.json"
REFERENCE_DIRECTORY = SNAPSHOT.parent / "first_page_scale1_refs"

SNAPSHOT_VERSION = 2
RASTER_PAGE = 0
RASTER_SCALE = 1.0
RASTER_BACKGROUND = (255, 255, 255, 255)
DEFAULT_WORKERS = 2

CANONICAL_SOURCE = {
    "operating_system": "ubuntu-24.04",
    "architecture": "x86_64",
    "imagecodecs": "2026.6.26",
    "jpeg2000": "openjpeg 2.5.4",
}


class RasterSnapshotError(ValueError):
    """Raised when the checked-in raster snapshot is malformed."""


@dataclass(frozen=True, slots=True)
class RasterTolerance:
    max_rgb_channel_delta: int
    max_changed_rgb_samples: int
    max_total_rgb_delta: int


@dataclass(frozen=True, slots=True)
class TolerantRasterSnapshot:
    canonical_sha256: str
    reference: str
    reference_path: pathlib.Path
    limits: RasterTolerance


@dataclass(frozen=True, slots=True)
class RasterSnapshot:
    exact: dict[str, str]
    tolerant: dict[str, TolerantRasterSnapshot]
    canonical_source: dict[str, str]

    @property
    def names(self) -> set[str]:
        return set(self.exact) | set(self.tolerant)


@dataclass(frozen=True, slots=True)
class RasterDifference:
    actual_shape: tuple[int, ...]
    reference_shape: tuple[int, ...]
    alpha_changed_samples: int | None
    max_rgb_channel_delta: int | None
    changed_rgb_samples: int | None
    changed_rgb_pixels: int | None
    total_rgb_delta: int | None

    def within(self, limits: RasterTolerance) -> bool:
        return (
            self.actual_shape == self.reference_shape
            and self.alpha_changed_samples == 0
            and self.max_rgb_channel_delta is not None
            and self.max_rgb_channel_delta <= limits.max_rgb_channel_delta
            and self.changed_rgb_samples is not None
            and self.changed_rgb_samples <= limits.max_changed_rgb_samples
            and self.total_rgb_delta is not None
            and self.total_rgb_delta <= limits.max_total_rgb_delta
        )

    def describe(self, limits: RasterTolerance) -> str:
        if self.actual_shape != self.reference_shape:
            return f"shape changed: actual={self.actual_shape}, reference={self.reference_shape}"
        return (
            f"alpha_changed_samples={self.alpha_changed_samples}; "
            f"max_rgb_channel_delta={self.max_rgb_channel_delta} "
            f"(limit {limits.max_rgb_channel_delta}); "
            f"changed_rgb_samples={self.changed_rgb_samples} "
            f"(limit {limits.max_changed_rgb_samples}); "
            f"changed_rgb_pixels={self.changed_rgb_pixels}; "
            f"total_rgb_delta={self.total_rgb_delta} "
            f"(limit {limits.max_total_rgb_delta})"
        )


@dataclass(frozen=True, slots=True)
class JpxPolicyScan:
    irreversible: bool
    unclassified: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RenderedGolden:
    name: str
    digest: str
    irreversible: bool
    unclassified: tuple[str, ...]
    reference_png: bytes | None


def internal_sha256(
    data: bytes | bytearray | memoryview | numpy.ndarray[Any, Any],
) -> str:
    return hashlib.sha256(data).hexdigest()


def raster_sha256(raster: RasterImage) -> str:
    """Hash the exact interleaved raster bytes."""
    return internal_sha256(raster.pixels)


def render_first_page(pdf: pathlib.Path) -> RasterImage:
    """Render page one with the fixed golden-raster settings."""
    with PdfDocument.open(pdf) as document:
        rendered = document.pages[RASTER_PAGE].render()
        return rendered.rasterize(
            scale=RASTER_SCALE,
            background=RASTER_BACKGROUND,
            cache=False,
        )


def internal_dictionary_mentions_jpx(dictionary: dict[Any, Any]) -> bool:
    raw_filters = lookup_dict_key(dictionary, "Filter")
    if raw_filters is None:
        raw_filters = lookup_dict_key(dictionary, "F")
    if raw_filters is None:
        raw_filters = lookup_dict_key(dictionary, "FFilter")
    filters = raw_filters if isinstance(raw_filters, (list, tuple)) else (raw_filters,)
    return any(normalize_pdf_name(item) in {"JPXDecode", "jpxdecode"} for item in filters)


def internal_classify_declared_jpx(
    raw: bytes | bytearray | memoryview,
    dictionary: dict[Any, Any],
) -> tuple[bool | None, bool]:
    """Return ``(irreversible, declared)`` for one encoded image stream."""
    try:
        spec = normalize_stream_decode_spec(dictionary)
    except Exception:
        return None, internal_dictionary_mentions_jpx(dictionary)

    jpx_indexes = [
        index
        for index, filter_name in enumerate(spec.filters)
        if (descriptor := FILTER_DESCRIPTOR_BY_NAME.get(filter_name)) is not None
        and descriptor.decoder == "jpx"
    ]
    if not jpx_indexes:
        return None, False
    if len(jpx_indexes) != 1:
        return None, True

    jpx_index = jpx_indexes[0]
    source = bytes(raw)
    try:
        for index in range(jpx_index):
            source = decode_one_filter(
                source,
                spec.filters[index],
                spec.params[index],
                dictionary=dictionary,
                parent_dictionary=None,
            )
    except Exception:
        return None, True
    return internal_jpx_uses_irreversible_wavelet(source), True


def internal_display_data(items: Iterable[object]) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield top-level and nested drawing dictionaries without following cycles."""
    pending = [(f"item {index}", getattr(item, "data", None)) for index, item in enumerate(items)]
    visited: set[int] = set()
    while pending:
        location, data = pending.pop()
        if not isinstance(data, dict) or id(data) in visited:
            continue
        visited.add(id(data))
        yield location, data
        nested = data.get("items")
        if isinstance(nested, (list, tuple)):
            pending.extend(
                (f"{location}/nested {index}", child)
                for index, child in enumerate(nested)
                if isinstance(child, dict)
            )


def internal_scan_jpx_policy(items: Iterable[object]) -> JpxPolicyScan:
    irreversible = False
    unclassified: list[str] = []
    for location, data in internal_display_data(items):
        dictionary = data.get("dictionary")
        raw = data.get("raw_data")
        if isinstance(dictionary, dict) and isinstance(raw, (bytes, bytearray, memoryview)):
            result, declared = internal_classify_declared_jpx(raw, dictionary)
            if declared and result is None:
                unclassified.append(location)
            irreversible |= result is True

        if not isinstance(dictionary, dict):
            continue
        mask_dictionary = dictionary.get("__soft_mask_dictionary__")
        mask_raw = dictionary.get("__soft_mask_raw_data__")
        if not isinstance(mask_dictionary, dict) or not isinstance(
            mask_raw, (bytes, bytearray, memoryview)
        ):
            continue
        result, declared = internal_classify_declared_jpx(mask_raw, mask_dictionary)
        mask_location = f"{location}/soft mask"
        if declared and result is None:
            unclassified.append(mask_location)
        irreversible |= result is True
    return JpxPolicyScan(irreversible, tuple(sorted(unclassified)))


def render_first_page_with_jpx_policy(pdf: pathlib.Path) -> tuple[RasterImage, JpxPolicyScan]:
    """Render page one and inspect every painted JPX image conservatively."""
    with PdfDocument.open(pdf) as document:
        rendered = document.pages[RASTER_PAGE].render()
        policy = internal_scan_jpx_policy(rendered.display_list.items)
        raster = rendered.rasterize(
            scale=RASTER_SCALE,
            background=RASTER_BACKGROUND,
            cache=False,
        )
        return raster, policy


def internal_render_golden(task: tuple[pathlib.Path, bool]) -> RenderedGolden:
    """Render one independent PDF in an updater worker process."""
    pdf, needs_reference = task
    raster, policy = render_first_page_with_jpx_policy(pdf)
    digest = raster_sha256(raster)
    reference_png = (
        bytes(imagecodecs.png_encode(raster.array(), level=9)) if needs_reference else None
    )
    return RenderedGolden(
        name=pdf.name,
        digest=digest,
        irreversible=policy.irreversible,
        unclassified=policy.unclassified,
        reference_png=reference_png,
    )


def corpus_pdfs() -> list[pathlib.Path]:
    """Return every PDF included in the raster golden corpus."""
    if not CORPUS.is_dir():
        return []
    return sorted(CORPUS.glob("*.pdf"))


def compare_raster_to_reference(
    actual: RasterImage,
    reference: numpy.ndarray[Any, Any],
) -> RasterDifference:
    """Measure exact alpha and bounded RGB differences against a reference."""
    if reference.dtype != numpy.dtype(numpy.uint8):
        raise RasterSnapshotError("raster reference must use uint8 samples")
    if reference.ndim != 3 or reference.shape[2] != 4:
        raise RasterSnapshotError("raster reference must have shape (height, width, 4)")

    actual_array = actual.array()
    actual_shape = tuple(int(value) for value in actual_array.shape)
    reference_shape = tuple(int(value) for value in reference.shape)
    if actual_shape != reference_shape:
        return RasterDifference(
            actual_shape,
            reference_shape,
            None,
            None,
            None,
            None,
            None,
        )

    alpha_changed = int(numpy.count_nonzero(actual_array[:, :, 3] != reference[:, :, 3]))
    rgb_delta = numpy.abs(
        actual_array[:, :, :3].astype(numpy.int16) - reference[:, :, :3].astype(numpy.int16)
    )
    return RasterDifference(
        actual_shape,
        reference_shape,
        alpha_changed,
        int(rgb_delta.max()),
        int(numpy.count_nonzero(rgb_delta)),
        int(numpy.count_nonzero(numpy.any(rgb_delta != 0, axis=2))),
        int(rgb_delta.sum(dtype=numpy.uint64)),
    )


def internal_snapshot_reference_path(
    reference: str,
    *,
    snapshot_path: pathlib.Path = SNAPSHOT,
) -> pathlib.Path:
    relative = pathlib.PurePosixPath(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise RasterSnapshotError(f"invalid raster reference path: {reference!r}")
    reference_directory = snapshot_path.parent / REFERENCE_DIRECTORY.name
    candidate = (snapshot_path.parent / pathlib.Path(*relative.parts)).resolve()
    if not candidate.is_relative_to(reference_directory.resolve()):
        raise RasterSnapshotError(
            f"raster reference is outside the reference directory: {reference}"
        )
    return candidate


def load_reference_raster(entry: TolerantRasterSnapshot) -> numpy.ndarray[Any, Any]:
    """Load and validate one lossless reference raster."""
    path = entry.reference_path
    if not path.is_file():
        raise RasterSnapshotError(f"raster reference is missing: {path}")
    try:
        reference = numpy.asarray(imagecodecs.png_decode(path.read_bytes()))
    except Exception as exc:
        raise RasterSnapshotError(f"could not decode raster reference: {path}") from exc
    if reference.dtype != numpy.dtype(numpy.uint8):
        raise RasterSnapshotError(f"raster reference is not uint8: {path}")
    if reference.ndim != 3 or reference.shape[2] != 4:
        raise RasterSnapshotError(f"raster reference is not RGBA: {path}")
    digest = internal_sha256(reference.tobytes())
    if digest != entry.canonical_sha256:
        raise RasterSnapshotError(
            f"raster reference digest mismatch for {path}: "
            f"manifest={entry.canonical_sha256}, decoded={digest}"
        )
    return reference


def internal_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise RasterSnapshotError(f"{field} must be a non-negative integer")
    return value


def internal_sha256_string(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RasterSnapshotError(f"{field} must be a lowercase SHA-256 digest")
    return value


def load_snapshot(path: pathlib.Path = SNAPSHOT) -> RasterSnapshot:
    """Load and strictly validate the versioned raster manifest."""
    if not path.is_file():
        raise RasterSnapshotError(f"raster snapshot is missing: {path}")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RasterSnapshotError(f"could not read raster snapshot: {path}") from exc
    if not isinstance(data, dict) or data.get("version") != SNAPSHOT_VERSION:
        raise RasterSnapshotError(f"raster snapshot must use schema version {SNAPSHOT_VERSION}")
    expected_render = {
        "page": RASTER_PAGE,
        "scale": RASTER_SCALE,
        "background": list(RASTER_BACKGROUND),
    }
    if data.get("render") != expected_render:
        raise RasterSnapshotError("raster snapshot render settings do not match the renderer")
    raw_source = data.get("canonical_source")
    if not isinstance(raw_source, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw_source.items()
    ):
        raise RasterSnapshotError("raster snapshot canonical_source must be a string object")
    if set(raw_source) != set(CANONICAL_SOURCE):
        raise RasterSnapshotError(
            f"raster snapshot canonical_source fields must be {sorted(CANONICAL_SOURCE)}"
        )
    canonical_source = dict(raw_source)

    exact_data = data.get("exact")
    tolerant_data = data.get("tolerant")
    if not isinstance(exact_data, dict) or not isinstance(tolerant_data, dict):
        raise RasterSnapshotError("raster snapshot exact and tolerant sections must be objects")

    exact: dict[str, str] = {}
    for name, digest in exact_data.items():
        if not isinstance(name, str):
            raise RasterSnapshotError("raster snapshot names must be strings")
        exact[name] = internal_sha256_string(digest, field=f"exact[{name!r}]")

    tolerant: dict[str, TolerantRasterSnapshot] = {}
    references: set[str] = set()
    for name, raw_entry in tolerant_data.items():
        if not isinstance(name, str) or not isinstance(raw_entry, dict):
            raise RasterSnapshotError("tolerant raster entries must be named objects")
        raw_limits = raw_entry.get("limits")
        if not isinstance(raw_limits, dict):
            raise RasterSnapshotError(f"tolerant[{name!r}].limits must be an object")
        reference = raw_entry.get("reference")
        if not isinstance(reference, str):
            raise RasterSnapshotError(f"tolerant[{name!r}].reference must be a string")
        reference_path = internal_snapshot_reference_path(reference, snapshot_path=path)
        if reference in references:
            raise RasterSnapshotError(f"duplicate raster reference path: {reference}")
        references.add(reference)
        tolerant[name] = TolerantRasterSnapshot(
            canonical_sha256=internal_sha256_string(
                raw_entry.get("canonical_sha256"),
                field=f"tolerant[{name!r}].canonical_sha256",
            ),
            reference=reference,
            reference_path=reference_path,
            limits=RasterTolerance(
                max_rgb_channel_delta=internal_nonnegative_int(
                    raw_limits.get("max_rgb_channel_delta"),
                    field=f"tolerant[{name!r}].limits.max_rgb_channel_delta",
                ),
                max_changed_rgb_samples=internal_nonnegative_int(
                    raw_limits.get("max_changed_rgb_samples"),
                    field=f"tolerant[{name!r}].limits.max_changed_rgb_samples",
                ),
                max_total_rgb_delta=internal_nonnegative_int(
                    raw_limits.get("max_total_rgb_delta"),
                    field=f"tolerant[{name!r}].limits.max_total_rgb_delta",
                ),
            ),
        )

    overlap = set(exact) & set(tolerant)
    if overlap:
        raise RasterSnapshotError(f"raster entries cannot be exact and tolerant: {sorted(overlap)}")
    return RasterSnapshot(exact, tolerant, canonical_source)


def internal_jpx_stage_diagnostics(pdf: pathlib.Path) -> str:
    """Hash compressed and decoded JPX stages for a failed tolerant comparison."""
    diagnostics: list[str] = []
    with PdfDocument.open(pdf) as document:
        rendered = document.pages[RASTER_PAGE].render()
        for index, item in enumerate(rendered.display_list.items):
            data = getattr(item, "data", None)
            if not isinstance(data, dict):
                continue
            raw = data.get("raw_data")
            if not isinstance(raw, (bytes, bytearray, memoryview)):
                continue
            raw_view = memoryview(raw)
            irreversible = internal_jpx_uses_irreversible_wavelet(raw_view)
            if irreversible is None:
                continue
            raw_digest = internal_sha256(raw_view)
            try:
                decoded = decode_jpx_image(raw_view)
                decoded_digest = internal_sha256(memoryview(decoded))
                shape = "x".join(str(value) for value in decoded.shape)
                decoded_text = f"decoded={decoded_digest}, shape={shape}"
            except Exception as exc:
                decoded_text = f"decode_error={type(exc).__name__}: {exc}"
            wavelet = "irreversible-9/7" if irreversible else "reversible-5/3"
            diagnostics.append(f"item {index}: {wavelet}, raw={raw_digest}, {decoded_text}")
    return "; ".join(diagnostics) or "no classifiable JPX display item"


def raster_snapshot_failure(pdf: pathlib.Path, snapshot: RasterSnapshot) -> str | None:
    """Return a diagnostic failure string, or ``None`` when a raster matches."""
    raster = render_first_page(pdf)
    actual_digest = raster_sha256(raster)
    expected_digest = snapshot.exact.get(pdf.name)
    if expected_digest is not None:
        if actual_digest == expected_digest:
            return None
        return f"exact digest changed: expected={expected_digest}, actual={actual_digest}"

    entry = snapshot.tolerant.get(pdf.name)
    if entry is None:
        return "no exact or tolerant raster snapshot entry"
    reference = load_reference_raster(entry)
    actual_shape = tuple(int(value) for value in raster.array().shape)
    reference_shape = tuple(int(value) for value in reference.shape)
    if actual_shape != reference_shape:
        return f"shape changed: actual={actual_shape}, reference={reference_shape}"
    if actual_digest == entry.canonical_sha256:
        return None

    difference = compare_raster_to_reference(raster, reference)
    if difference.within(entry.limits):
        return None
    return (
        f"tolerant raster exceeded its measured bounds: actual={actual_digest}; "
        f"{difference.describe(entry.limits)}; "
        f"JPX stages: {internal_jpx_stage_diagnostics(pdf)}"
    )


def internal_operating_system_id(*, system: str | None = None) -> str:
    current_system = (system or platform.system()).lower()
    if current_system != "linux":
        return current_system
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        return "linux-unknown"
    distribution = release.get("ID", "linux").lower()
    version = release.get("VERSION_ID", "unknown")
    return f"{distribution}-{version}"


def current_raster_source(
    *,
    system: str | None = None,
    machine: str | None = None,
    operating_system: str | None = None,
    imagecodecs_version: str | None = None,
    jpeg2000_version: str | None = None,
) -> dict[str, str]:
    """Describe every environment property that defines the canonical baseline."""
    architecture = (machine or platform.machine()).lower()
    if architecture == "amd64":
        architecture = "x86_64"
    return {
        "operating_system": operating_system or internal_operating_system_id(system=system),
        "architecture": architecture,
        "imagecodecs": imagecodecs_version or imagecodecs.__version__,
        "jpeg2000": jpeg2000_version or imagecodecs.jpeg2k_version(),
    }


def is_canonical_regeneration_platform(
    *,
    system: str | None = None,
    machine: str | None = None,
    operating_system: str | None = None,
    imagecodecs_version: str | None = None,
    jpeg2000_version: str | None = None,
) -> bool:
    """Return whether this exactly matches the pinned baseline environment."""
    return (
        current_raster_source(
            system=system,
            machine=machine,
            operating_system=operating_system,
            imagecodecs_version=imagecodecs_version,
            jpeg2000_version=jpeg2000_version,
        )
        == CANONICAL_SOURCE
    )


def internal_reference_name(pdf_name: str, digest: str) -> str:
    stem = pathlib.Path(pdf_name).stem
    return f"{REFERENCE_DIRECTORY.name}/{stem}.{digest[:16]}.png"


def internal_snapshot_json(snapshot: RasterSnapshot) -> str:
    tolerant: dict[str, object] = {}
    for name, entry in sorted(snapshot.tolerant.items()):
        tolerant[name] = {
            "canonical_sha256": entry.canonical_sha256,
            "reference": entry.reference,
            "limits": {
                "max_rgb_channel_delta": entry.limits.max_rgb_channel_delta,
                "max_changed_rgb_samples": entry.limits.max_changed_rgb_samples,
                "max_total_rgb_delta": entry.limits.max_total_rgb_delta,
            },
        }
    data = {
        "version": SNAPSHOT_VERSION,
        "canonical_source": snapshot.canonical_source,
        "render": {
            "page": RASTER_PAGE,
            "scale": RASTER_SCALE,
            "background": list(RASTER_BACKGROUND),
        },
        "exact": dict(sorted(snapshot.exact.items())),
        "tolerant": tolerant,
    }
    return json.dumps(data, indent=1) + "\n"


def internal_parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-noncanonical-write",
        action="store_true",
        help=(
            "DESTRUCTIVELY bootstrap outside the pinned environment; the resulting "
            "manifest intentionally fails its canonical-provenance test"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"isolated renderer processes to use (default: {DEFAULT_WORKERS})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = internal_parse_args(argv)
    if args.workers < 1:
        print("--workers must be at least 1", file=sys.stderr)
        return 2
    raster_source = current_raster_source()
    if raster_source != CANONICAL_SOURCE and not args.allow_noncanonical_write:
        print(
            "refusing to regenerate raster references outside the pinned canonical "
            f"environment; expected={CANONICAL_SOURCE}, actual={raster_source}; pass "
            "--allow-noncanonical-write only for an intentional destructive rewrite",
            file=sys.stderr,
        )
        return 2

    try:
        existing = load_snapshot()
    except RasterSnapshotError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    pdfs = corpus_pdfs()
    if not pdfs:
        print(f"corpus not found or empty at {CORPUS}", file=sys.stderr)
        return 1

    corpus_names = {pdf.name for pdf in pdfs}
    stale_tolerant = set(existing.tolerant) - corpus_names
    if stale_tolerant:
        print(
            f"tolerant policy names missing from corpus: {sorted(stale_tolerant)}",
            file=sys.stderr,
        )
        return 1

    started = time.monotonic()
    exact: dict[str, str] = {}
    tolerant: dict[str, TolerantRasterSnapshot] = {}
    unconfigured_irreversible: list[str] = []
    incorrectly_tolerant: list[str] = []
    unclassified_jpx: list[str] = []

    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="raster-golden-", dir=SNAPSHOT.parent) as temp_name:
        temp_root = pathlib.Path(temp_name)
        tasks = [(pdf, pdf.name in existing.tolerant) for pdf in pdfs]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(internal_render_golden, tasks)
            for index, result in enumerate(results, start=1):
                policy = existing.tolerant.get(result.name)
                if result.unclassified:
                    locations = ", ".join(result.unclassified)
                    unclassified_jpx.append(f"{result.name}: {locations}")
                if result.irreversible and policy is None:
                    unconfigured_irreversible.append(result.name)
                if policy is not None and not result.irreversible:
                    incorrectly_tolerant.append(result.name)

                if policy is None:
                    exact[result.name] = result.digest
                else:
                    if result.reference_png is None:
                        raise AssertionError("tolerant raster worker omitted its reference PNG")
                    reference = internal_reference_name(result.name, result.digest)
                    reference_path = temp_root / pathlib.Path(
                        *pathlib.PurePosixPath(reference).parts
                    )
                    reference_path.parent.mkdir(parents=True, exist_ok=True)
                    reference_path.write_bytes(result.reference_png)
                    tolerant[result.name] = TolerantRasterSnapshot(
                        canonical_sha256=result.digest,
                        reference=reference,
                        reference_path=internal_snapshot_reference_path(reference),
                        limits=policy.limits,
                    )
                if index % 10 == 0 or index == len(pdfs):
                    print(f"rendered {index}/{len(pdfs)} documents", flush=True)

        if unclassified_jpx:
            print(
                f"declared JPX streams could not be classified safely: {unclassified_jpx}",
                file=sys.stderr,
            )
            return 1

        if unconfigured_irreversible:
            print(
                "irreversible JPX pages need calibrated tolerant policies: "
                f"{unconfigured_irreversible}",
                file=sys.stderr,
            )
            return 1
        if incorrectly_tolerant:
            print(
                f"tolerant policies no longer contain irreversible JPX: {incorrectly_tolerant}",
                file=sys.stderr,
            )
            return 1

        updated = RasterSnapshot(exact, tolerant, raster_source)
        REFERENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        expected_references: set[pathlib.Path] = set()
        for entry in tolerant.values():
            source_reference = temp_root / pathlib.Path(
                *pathlib.PurePosixPath(entry.reference).parts
            )
            destination = internal_snapshot_reference_path(entry.reference)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source_reference, destination)
            expected_references.add(destination.resolve())

        temporary_snapshot = SNAPSHOT.with_suffix(".json.tmp")
        temporary_snapshot.write_text(internal_snapshot_json(updated))
        os.replace(temporary_snapshot, SNAPSHOT)

        removed = 0
        for reference_path in REFERENCE_DIRECTORY.glob("*.png"):
            if reference_path.resolve() not in expected_references:
                reference_path.unlink()
                removed += 1

    print(f"updated {SNAPSHOT} in {time.monotonic() - started:.1f}s")
    if removed:
        print(f"removed {removed} stale raster reference(s)")
    return 0

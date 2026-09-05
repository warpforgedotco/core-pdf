"""Platform-neutral raster-golden comparison and calibration.

Exact pages use RGBA SHA-256 digests. Pages containing irreversible JPEG 2000
use a lossless reference plus a sparse RGB envelope observed across supported
CI platforms. Collection never changes tracked files; only a complete merge
can redefine the portability contract.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Iterable, cast

import imagecodecs
import numpy

from core_pdf import PdfDocument
from core_pdf.impl._impl.render.model import RasterImage
from core_pdf.impl.spec.s_07_filters.decode_spec import normalize_stream_decode_spec
from core_pdf.impl.spec.s_07_filters.decoders import decode_jpx_image
from core_pdf.impl.spec.s_07_filters.jpeg2000 import internal_jpx_uses_irreversible_wavelet
from core_pdf.impl.spec.s_07_filters.pipeline import decode_one_filter
from core_pdf.impl.spec.s_07_filters.registry import FILTER_DESCRIPTOR_BY_NAME
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import normalize_pdf_name

internal_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = internal_ROOT / "tests" / "fixtures" / "SCORE-Bench" / "src"
SNAPSHOT = internal_ROOT / "tests" / "snapshots" / "raster" / "first_page_scale1.json"
REFERENCE_DIRECTORY = SNAPSHOT.parent / "first_page_scale1_refs"

SNAPSHOT_VERSION = 3
OBSERVATION_VERSION = 1
RASTER_PAGE = 0
RASTER_SCALE = 1.0
RASTER_BACKGROUND = (255, 255, 255, 255)
DEFAULT_WORKERS = 2
EXPECTED_CODEC_STACK = {"imagecodecs": "2026.6.26", "jpeg2000": "openjpeg 2.5.4"}
SUPPORTED_PLATFORMS = {
    "linux-x86_64": ("linux", "x86_64"),
    "macos-arm64": ("darwin", "arm64"),
}


class RasterSnapshotError(ValueError):
    """Raised when a raster snapshot or observation is malformed."""


@dataclass(frozen=True, slots=True)
class PortableRasterSnapshot:
    reference_sha256: str
    reference: str
    reference_path: pathlib.Path
    variants: dict[str, str]
    rgb_ranges: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class RasterSnapshot:
    exact: dict[str, str]
    portable: dict[str, PortableRasterSnapshot]
    codec_stack: dict[str, str]

    @property
    def names(self) -> set[str]:
        return set(self.exact) | set(self.portable)


@dataclass(frozen=True, slots=True)
class RasterEnvelopeDifference:
    actual_shape: tuple[int, ...]
    reference_shape: tuple[int, ...]
    alpha_offsets: tuple[int, ...]
    unexpected_rgb_offsets: tuple[int, ...]
    out_of_range: tuple[tuple[int, int, int, int], ...]

    @property
    def accepted(self) -> bool:
        return (
            self.actual_shape == self.reference_shape
            and not self.alpha_offsets
            and not self.unexpected_rgb_offsets
            and not self.out_of_range
        )

    def describe(self) -> str:
        if self.actual_shape != self.reference_shape:
            return f"shape changed: actual={self.actual_shape}, reference={self.reference_shape}"
        return (
            f"alpha_offsets={internal_preview(self.alpha_offsets)}; "
            f"unexpected_rgb_offsets={internal_preview(self.unexpected_rgb_offsets)}; "
            f"out_of_range={internal_preview(self.out_of_range)}"
        )


@dataclass(frozen=True, slots=True)
class JpxPolicyScan:
    irreversible: bool
    unclassified: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JpxStageObservation:
    location: str
    raw_sha256: str
    decoded_sha256: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RenderedGolden:
    name: str
    digest: str
    width: int
    height: int
    irreversible: bool
    unclassified: tuple[str, ...]
    jpx_stages: tuple[JpxStageObservation, ...]
    reference_png: bytes | None


@dataclass(frozen=True, slots=True)
class ObservedPage:
    digest: str
    width: int
    height: int
    irreversible: bool
    unclassified: tuple[str, ...]
    jpx_stages: tuple[JpxStageObservation, ...]
    raster_path: pathlib.Path | None


@dataclass(frozen=True, slots=True)
class RasterObservation:
    platform_id: str
    revision: str
    runtime: dict[str, str]
    codec_stack: dict[str, str]
    pages: dict[str, ObservedPage]


def internal_preview(values: tuple[Any, ...], limit: int = 8) -> str:
    preview = list(values[:limit])
    suffix = f" (+{len(values) - limit} more)" if len(values) > limit else ""
    return f"{preview}{suffix}"


def internal_sha256(
    data: bytes | bytearray | memoryview | numpy.ndarray[Any, Any],
) -> str:
    return hashlib.sha256(data).hexdigest()


def raster_sha256(raster: RasterImage) -> str:
    """Hash the exact interleaved raster bytes."""
    return internal_sha256(raster.pixels)


def internal_codec_stack() -> dict[str, str]:
    return {"imagecodecs": imagecodecs.__version__, "jpeg2000": imagecodecs.jpeg2k_version()}


def internal_render_settings() -> dict[str, object]:
    return {"page": RASTER_PAGE, "scale": RASTER_SCALE, "background": list(RASTER_BACKGROUND)}


def internal_operating_system_id() -> str:
    current_system = platform.system().lower()
    if current_system != "linux":
        return current_system
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        return "linux-unknown"
    return f"{release.get('ID', 'linux').lower()}-{release.get('VERSION_ID', 'unknown')}"


def current_raster_runtime() -> dict[str, str]:
    architecture = platform.machine().lower()
    architecture = {"amd64": "x86_64", "aarch64": "arm64"}.get(architecture, architecture)
    return {
        "system": platform.system().lower(),
        "operating_system": internal_operating_system_id(),
        "architecture": architecture,
    }


def render_first_page(pdf: pathlib.Path) -> RasterImage:
    """Render page one with the fixed golden-raster settings."""
    return render_first_page_with_jpx_policy(pdf)[0]


def internal_dictionary_mentions_jpx(dictionary: dict[Any, Any]) -> bool:
    raw_filters = dictionary.get("Filter")
    if raw_filters is None:
        raw_filters = dictionary.get("F")
    if raw_filters is None:
        raw_filters = dictionary.get("FFilter")
    filters = raw_filters if isinstance(raw_filters, (list, tuple)) else (raw_filters,)
    return any(normalize_pdf_name(item) in {"JPXDecode", "jpxdecode"} for item in filters)


def internal_jpx_source(
    raw: bytes | bytearray | memoryview, dictionary: dict[Any, Any]
) -> tuple[bytes, bool | None] | None:
    try:
        spec = normalize_stream_decode_spec(dictionary)
    except Exception:
        return None
    indexes = [
        index
        for index, filter_name in enumerate(spec.filters)
        if (descriptor := FILTER_DESCRIPTOR_BY_NAME.get(filter_name)) is not None
        and descriptor.decoder == "jpx"
    ]
    if len(indexes) != 1:
        return None
    source = bytes(raw)
    try:
        for index in range(indexes[0]):
            source = decode_one_filter(
                source,
                spec.filters[index],
                spec.params[index],
                dictionary=dictionary,
                parent_dictionary=None,
            )
    except Exception:
        return None
    return source, internal_jpx_uses_irreversible_wavelet(source)


def internal_classify_declared_jpx(
    raw: bytes | bytearray | memoryview, dictionary: dict[Any, Any]
) -> tuple[bool | None, bool]:
    """Return ``(irreversible, declared)`` for one encoded image stream."""
    result = internal_jpx_source(raw, dictionary)
    if result is not None:
        return result[1], True
    return None, internal_dictionary_mentions_jpx(dictionary)


def internal_display_data(items: Iterable[object]) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield top-level and nested drawing dictionaries without following cycles."""
    pending: list[tuple[str, object]] = []
    for index, item in enumerate(items):
        data = getattr(item, "data", None)
        if not isinstance(data, dict):
            to_data = getattr(item, "to_data", None)
            data = to_data() if callable(to_data) else None
        pending.append((f"item {index}", data))
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


def internal_image_sources(
    items: Iterable[object],
) -> Iterable[tuple[str, bytes | bytearray | memoryview, dict[Any, Any]]]:
    for location, data in internal_display_data(items):
        dictionary = data.get("dictionary")
        raw = data.get("raw_data")
        if isinstance(dictionary, dict) and isinstance(raw, (bytes, bytearray, memoryview)):
            yield location, raw, dictionary
        soft_mask = data.get("soft_mask")
        mask_dictionary = getattr(soft_mask, "dictionary", None)
        mask_raw = getattr(soft_mask, "raw", None)
        if isinstance(mask_dictionary, dict) and isinstance(
            mask_raw, (bytes, bytearray, memoryview)
        ):
            yield f"{location}/soft mask", mask_raw, mask_dictionary


def internal_scan_jpx_policy(items: Iterable[object]) -> JpxPolicyScan:
    irreversible = False
    unclassified: list[str] = []
    for location, raw, dictionary in internal_image_sources(items):
        result, declared = internal_classify_declared_jpx(raw, dictionary)
        if declared and result is None:
            unclassified.append(location)
        irreversible |= result is True
    return JpxPolicyScan(irreversible, tuple(sorted(unclassified)))


def internal_jpx_stage_observations(items: Iterable[object]) -> tuple[JpxStageObservation, ...]:
    observations: list[JpxStageObservation] = []
    for location, raw, dictionary in internal_image_sources(items):
        source = internal_jpx_source(raw, dictionary)
        if source is None or source[1] is not True:
            continue
        encoded = source[0]
        decoded = decode_jpx_image(encoded)
        observations.append(
            JpxStageObservation(
                location,
                internal_sha256(encoded),
                internal_sha256(memoryview(decoded)),
                tuple(int(value) for value in decoded.shape),
            )
        )
    return tuple(sorted(observations, key=lambda item: item.location))


def render_first_page_with_jpx_policy(
    pdf: pathlib.Path,
) -> tuple[RasterImage, JpxPolicyScan]:
    """Render page one and inspect every painted JPX image conservatively."""
    with PdfDocument.open(pdf) as document:
        rendered = document.pages[RASTER_PAGE].render()
        policy = internal_scan_jpx_policy(rendered.display_list.items)
        raster = rendered.rasterize(scale=RASTER_SCALE, background=RASTER_BACKGROUND)
        return raster, policy


def internal_render_golden(pdf: pathlib.Path) -> RenderedGolden:
    """Render one independent PDF in an observation worker process."""
    with PdfDocument.open(pdf) as document:
        rendered = document.pages[RASTER_PAGE].render()
        policy = internal_scan_jpx_policy(rendered.display_list.items)
        raster = rendered.rasterize(scale=RASTER_SCALE, background=RASTER_BACKGROUND)
        digest = raster_sha256(raster)
        stages = (
            internal_jpx_stage_observations(rendered.display_list.items)
            if policy.irreversible
            else ()
        )
        reference_png = (
            bytes(imagecodecs.png_encode(raster.array(), level=9)) if policy.irreversible else None
        )
        return RenderedGolden(
            pdf.name,
            digest,
            raster.width,
            raster.height,
            policy.irreversible,
            policy.unclassified,
            stages,
            reference_png,
        )


def corpus_pdfs() -> list[pathlib.Path]:
    """Return every PDF included in the raster golden corpus."""
    return sorted(CORPUS.glob("*.pdf")) if CORPUS.is_dir() else []


def compare_raster_to_reference(
    actual: RasterImage,
    reference: numpy.ndarray[Any, Any],
    rgb_ranges: tuple[tuple[int, int, int], ...],
) -> RasterEnvelopeDifference:
    """Compare a raster against an exact reference plus sparse RGB ranges."""
    if reference.dtype != numpy.dtype(numpy.uint8):
        raise RasterSnapshotError("raster reference must use uint8 samples")
    if reference.ndim != 3 or reference.shape[2] != 4:
        raise RasterSnapshotError("raster reference must have shape (height, width, 4)")
    actual_array = actual.array()
    actual_shape = tuple(int(value) for value in actual_array.shape)
    reference_shape = tuple(int(value) for value in reference.shape)
    if actual_shape != reference_shape:
        return RasterEnvelopeDifference(actual_shape, reference_shape, (), (), ())
    actual_flat = actual_array.reshape(-1)
    reference_flat = reference.reshape(-1)
    changed = numpy.flatnonzero(actual_flat != reference_flat)
    alpha_offsets = tuple(int(offset) for offset in changed if int(offset) % 4 == 3)
    configured = {offset: (minimum, maximum) for offset, minimum, maximum in rgb_ranges}
    unexpected: list[int] = []
    out_of_range: list[tuple[int, int, int, int]] = []
    for raw_offset in changed:
        offset = int(raw_offset)
        if offset % 4 == 3:
            continue
        bounds = configured.get(offset)
        if bounds is None:
            unexpected.append(offset)
            continue
        value = int(actual_flat[offset])
        if not bounds[0] <= value <= bounds[1]:
            out_of_range.append((offset, value, bounds[0], bounds[1]))
    return RasterEnvelopeDifference(
        actual_shape,
        reference_shape,
        alpha_offsets,
        tuple(unexpected),
        tuple(out_of_range),
    )


def internal_snapshot_reference_path(
    reference: str, *, snapshot_path: pathlib.Path = SNAPSHOT
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


def load_reference_raster(entry: PortableRasterSnapshot) -> numpy.ndarray[Any, Any]:
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
    if digest != entry.reference_sha256:
        raise RasterSnapshotError(
            f"raster reference digest mismatch for {path}: "
            f"manifest={entry.reference_sha256}, decoded={digest}"
        )
    return reference


def internal_sha256_string(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RasterSnapshotError(f"{field} must be a lowercase SHA-256 digest")
    return value


def internal_string_dict(value: object, *, field: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise RasterSnapshotError(f"{field} must be a string object")
    return {str(key): str(item) for key, item in value.items()}


def internal_parse_rgb_ranges(value: object, *, field: str) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(value, list):
        raise RasterSnapshotError(f"{field} must be an array")
    result: list[tuple[int, int, int]] = []
    previous = -1
    for index, item in enumerate(value):
        if (
            not isinstance(item, list)
            or len(item) != 3
            or any(type(part) is not int for part in item)
        ):
            raise RasterSnapshotError(f"{field}[{index}] must be [offset, minimum, maximum]")
        offset, minimum, maximum = cast(list[int], item)
        if offset <= previous or offset < 0 or offset % 4 == 3:
            raise RasterSnapshotError(f"{field} offsets must be sorted unique RGB byte offsets")
        if not 0 <= minimum < maximum <= 255:
            raise RasterSnapshotError(f"{field}[{index}] has invalid sample bounds")
        result.append((offset, minimum, maximum))
        previous = offset
    return tuple(result)


def load_snapshot(path: pathlib.Path = SNAPSHOT) -> RasterSnapshot:
    """Load and strictly validate the platform-neutral raster manifest."""
    if not path.is_file():
        raise RasterSnapshotError(f"raster snapshot is missing: {path}")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RasterSnapshotError(f"could not read raster snapshot: {path}") from exc
    if not isinstance(data, dict) or data.get("version") != SNAPSHOT_VERSION:
        raise RasterSnapshotError(f"raster snapshot must use schema version {SNAPSHOT_VERSION}")
    if data.get("render") != internal_render_settings():
        raise RasterSnapshotError("raster snapshot render settings do not match the renderer")
    codec_stack = internal_string_dict(data.get("codec_stack"), field="codec_stack")
    if codec_stack != EXPECTED_CODEC_STACK:
        raise RasterSnapshotError(
            f"raster snapshot codec stack mismatch: expected={EXPECTED_CODEC_STACK}, "
            f"actual={codec_stack}"
        )
    exact_data = data.get("exact")
    portable_data = data.get("portable")
    if not isinstance(exact_data, dict) or not isinstance(portable_data, dict):
        raise RasterSnapshotError("raster snapshot exact and portable sections must be objects")
    exact: dict[str, str] = {}
    for name, digest in exact_data.items():
        if not isinstance(name, str):
            raise RasterSnapshotError("raster snapshot names must be strings")
        exact[name] = internal_sha256_string(digest, field=f"exact[{name!r}]")

    portable: dict[str, PortableRasterSnapshot] = {}
    references: set[str] = set()
    for name, raw_entry in portable_data.items():
        if not isinstance(name, str) or not isinstance(raw_entry, dict):
            raise RasterSnapshotError("portable raster entries must be named objects")
        reference = raw_entry.get("reference")
        if not isinstance(reference, str):
            raise RasterSnapshotError(f"portable[{name!r}].reference must be a string")
        if reference in references:
            raise RasterSnapshotError(f"duplicate raster reference path: {reference}")
        references.add(reference)
        variants = internal_string_dict(
            raw_entry.get("variants"), field=f"portable[{name!r}].variants"
        )
        if set(variants) != set(SUPPORTED_PLATFORMS):
            raise RasterSnapshotError(
                f"portable[{name!r}].variants must cover {sorted(SUPPORTED_PLATFORMS)}"
            )
        variants = {
            platform_id: internal_sha256_string(
                digest, field=f"portable[{name!r}].variants[{platform_id!r}]"
            )
            for platform_id, digest in variants.items()
        }
        entry = PortableRasterSnapshot(
            internal_sha256_string(
                raw_entry.get("reference_sha256"),
                field=f"portable[{name!r}].reference_sha256",
            ),
            reference,
            internal_snapshot_reference_path(reference, snapshot_path=path),
            variants,
            internal_parse_rgb_ranges(
                raw_entry.get("rgb_ranges"), field=f"portable[{name!r}].rgb_ranges"
            ),
        )
        if entry.reference_sha256 not in set(entry.variants.values()):
            raise RasterSnapshotError(f"portable[{name!r}] reference is not an observed variant")
        reference_array = load_reference_raster(entry)
        flat = reference_array.reshape(-1)
        for offset, minimum, maximum in entry.rgb_ranges:
            if offset >= flat.size:
                raise RasterSnapshotError(f"portable[{name!r}] RGB offset is out of bounds")
            if not minimum <= int(flat[offset]) <= maximum:
                raise RasterSnapshotError(
                    f"portable[{name!r}] RGB range does not contain its reference sample"
                )
        portable[name] = entry
    overlap = set(exact) & set(portable)
    if overlap:
        raise RasterSnapshotError(f"raster entries cannot be exact and portable: {sorted(overlap)}")
    return RasterSnapshot(exact, portable, codec_stack)


def internal_jpx_stage_diagnostics(pdf: pathlib.Path) -> str:
    """Hash compressed and decoded irreversible-JPX stages for a failure."""
    with PdfDocument.open(pdf) as document:
        rendered = document.pages[RASTER_PAGE].render()
        stages = internal_jpx_stage_observations(rendered.display_list.items)
    return (
        "; ".join(
            f"{stage.location}: raw={stage.raw_sha256}, decoded={stage.decoded_sha256}, "
            f"shape={'x'.join(str(value) for value in stage.shape)}"
            for stage in stages
        )
        or "no classifiable irreversible JPX display item"
    )


def raster_snapshot_failure(pdf: pathlib.Path, snapshot: RasterSnapshot) -> str | None:
    """Return a diagnostic failure string, or ``None`` when a raster matches."""
    raster, policy = render_first_page_with_jpx_policy(pdf)
    if policy.unclassified:
        return f"declared JPX streams could not be classified: {policy.unclassified}"
    actual_digest = raster_sha256(raster)
    expected_digest = snapshot.exact.get(pdf.name)
    if expected_digest is not None:
        if policy.irreversible:
            return "irreversible JPX page is incorrectly configured as exact"
        if actual_digest == expected_digest:
            return None
        return f"exact digest changed: expected={expected_digest}, actual={actual_digest}"
    entry = snapshot.portable.get(pdf.name)
    if entry is None:
        return "no exact or portable raster snapshot entry"
    if not policy.irreversible:
        return "portable raster page no longer contains irreversible JPX"
    reference = load_reference_raster(entry)
    difference = compare_raster_to_reference(raster, reference, entry.rgb_ranges)
    if difference.accepted:
        return None
    return (
        f"portable raster escaped its calibrated envelope: actual={actual_digest}; "
        f"{difference.describe()}; JPX stages: {internal_jpx_stage_diagnostics(pdf)}"
    )


def internal_reference_name(pdf_name: str, digest: str) -> str:
    return f"{REFERENCE_DIRECTORY.name}/{pathlib.Path(pdf_name).stem}.{digest[:16]}.png"


def internal_snapshot_json(snapshot: RasterSnapshot) -> str:
    portable: dict[str, object] = {}
    for name, entry in sorted(snapshot.portable.items()):
        portable[name] = {
            "reference_sha256": entry.reference_sha256,
            "reference": entry.reference,
            "variants": dict(sorted(entry.variants.items())),
            "rgb_ranges": [list(item) for item in entry.rgb_ranges],
        }
    data = {
        "version": SNAPSHOT_VERSION,
        "codec_stack": snapshot.codec_stack,
        "render": internal_render_settings(),
        "exact": dict(sorted(snapshot.exact.items())),
        "portable": portable,
    }
    return json.dumps(data, indent=1) + "\n"


def internal_stage_json(stage: JpxStageObservation) -> dict[str, object]:
    return {
        "location": stage.location,
        "raw_sha256": stage.raw_sha256,
        "decoded_sha256": stage.decoded_sha256,
        "shape": list(stage.shape),
    }


def internal_observation_json(
    platform_id: str,
    revision: str,
    runtime: dict[str, str],
    pages: dict[str, ObservedPage],
) -> str:
    page_data: dict[str, object] = {}
    for name, page in sorted(pages.items()):
        entry: dict[str, object] = {
            "sha256": page.digest,
            "width": page.width,
            "height": page.height,
            "irreversible_jpx": page.irreversible,
            "unclassified_jpx": list(page.unclassified),
            "jpx_stages": [internal_stage_json(stage) for stage in page.jpx_stages],
        }
        if page.raster_path is not None:
            entry["raster"] = page.raster_path.as_posix()
        page_data[name] = entry
    data = {
        "version": OBSERVATION_VERSION,
        "platform_id": platform_id,
        "revision": revision,
        "runtime": runtime,
        "codec_stack": internal_codec_stack(),
        "render": internal_render_settings(),
        "pages": page_data,
    }
    return json.dumps(data, indent=1) + "\n"


def internal_validate_platform(platform_id: str, runtime: dict[str, str]) -> None:
    expected = SUPPORTED_PLATFORMS.get(platform_id)
    if expected is None:
        raise RasterSnapshotError(f"unsupported raster platform id: {platform_id}")
    actual = (runtime.get("system"), runtime.get("architecture"))
    if actual != expected:
        raise RasterSnapshotError(
            f"platform {platform_id} requires system/architecture {expected}, got {actual}"
        )


def internal_git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=internal_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def collect_observation(
    output: pathlib.Path,
    *,
    platform_id: str,
    revision: str,
    workers: int = DEFAULT_WORKERS,
) -> None:
    """Collect one platform observation without modifying golden files."""
    if workers < 1:
        raise RasterSnapshotError("workers must be at least 1")
    runtime = current_raster_runtime()
    internal_validate_platform(platform_id, runtime)
    codec_stack = internal_codec_stack()
    if codec_stack != EXPECTED_CODEC_STACK:
        raise RasterSnapshotError(
            f"codec stack mismatch: expected={EXPECTED_CODEC_STACK}, actual={codec_stack}"
        )
    if output.exists() and any(output.iterdir()):
        raise RasterSnapshotError(f"observation output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pdfs = corpus_pdfs()
    if not pdfs:
        raise RasterSnapshotError(f"corpus not found or empty at {CORPUS}")
    pages: dict[str, ObservedPage] = {}
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(internal_render_golden, pdfs), start=1):
            raster_path: pathlib.Path | None = None
            if result.reference_png is not None:
                relative = pathlib.Path("rasters") / f"{pathlib.Path(result.name).stem}.png"
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(result.reference_png)
                raster_path = relative
            pages[result.name] = ObservedPage(
                result.digest,
                result.width,
                result.height,
                result.irreversible,
                result.unclassified,
                result.jpx_stages,
                raster_path,
            )
            if index % 10 == 0 or index == len(pdfs):
                print(f"rendered {index}/{len(pdfs)} documents", flush=True)
    (output / "observation.json").write_text(
        internal_observation_json(platform_id, revision, runtime, pages)
    )
    print(f"collected {platform_id} observation in {time.monotonic() - started:.1f}s")


def internal_positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise RasterSnapshotError(f"{field} must be a positive integer")
    return value


def internal_parse_stage(value: object, *, field: str) -> JpxStageObservation:
    if not isinstance(value, dict):
        raise RasterSnapshotError(f"{field} must be an object")
    location = value.get("location")
    shape = value.get("shape")
    if not isinstance(location, str) or not isinstance(shape, list) or not shape:
        raise RasterSnapshotError(f"{field} has invalid location or shape")
    return JpxStageObservation(
        location,
        internal_sha256_string(value.get("raw_sha256"), field=f"{field}.raw_sha256"),
        internal_sha256_string(value.get("decoded_sha256"), field=f"{field}.decoded_sha256"),
        tuple(internal_positive_int(item, field=f"{field}.shape") for item in shape),
    )


def load_observation(directory: pathlib.Path) -> RasterObservation:
    """Load and validate one collected platform observation."""
    manifest = directory / "observation.json"
    try:
        data = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RasterSnapshotError(f"could not read raster observation: {manifest}") from exc
    if not isinstance(data, dict) or data.get("version") != OBSERVATION_VERSION:
        raise RasterSnapshotError(
            f"raster observation must use schema version {OBSERVATION_VERSION}"
        )
    if data.get("render") != internal_render_settings():
        raise RasterSnapshotError("observation render settings do not match the renderer")
    platform_id = data.get("platform_id")
    revision = data.get("revision")
    if not isinstance(platform_id, str) or not isinstance(revision, str) or not revision:
        raise RasterSnapshotError("observation platform_id and revision must be strings")
    runtime = internal_string_dict(data.get("runtime"), field="runtime")
    internal_validate_platform(platform_id, runtime)
    codec_stack = internal_string_dict(data.get("codec_stack"), field="codec_stack")
    if codec_stack != EXPECTED_CODEC_STACK:
        raise RasterSnapshotError(
            f"observation codec stack mismatch: expected={EXPECTED_CODEC_STACK}, "
            f"actual={codec_stack}"
        )
    raw_pages = data.get("pages")
    if not isinstance(raw_pages, dict):
        raise RasterSnapshotError("observation pages must be an object")
    pages: dict[str, ObservedPage] = {}
    for name, raw_page in raw_pages.items():
        if not isinstance(name, str) or not isinstance(raw_page, dict):
            raise RasterSnapshotError("observation pages must be named objects")
        irreversible = raw_page.get("irreversible_jpx")
        unclassified = raw_page.get("unclassified_jpx")
        raw_stages = raw_page.get("jpx_stages")
        if (
            type(irreversible) is not bool
            or not isinstance(unclassified, list)
            or not all(isinstance(item, str) for item in unclassified)
        ):
            raise RasterSnapshotError(f"observation page {name!r} has invalid JPX policy")
        if not isinstance(raw_stages, list):
            raise RasterSnapshotError(f"observation page {name!r} has invalid JPX stages")
        raster_value = raw_page.get("raster")
        raster_path = None
        if raster_value is not None:
            if not isinstance(raster_value, str):
                raise RasterSnapshotError(f"observation page {name!r} raster must be a string")
            relative = pathlib.PurePosixPath(raster_value)
            candidate = (directory / pathlib.Path(*relative.parts)).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not candidate.is_relative_to(directory.resolve())
            ):
                raise RasterSnapshotError(f"observation page {name!r} has unsafe raster path")
            raster_path = candidate
        if irreversible != (raster_path is not None):
            raise RasterSnapshotError(
                f"observation page {name!r} must carry a raster exactly when JPX is irreversible"
            )
        pages[name] = ObservedPage(
            internal_sha256_string(raw_page.get("sha256"), field=f"pages[{name!r}].sha256"),
            internal_positive_int(raw_page.get("width"), field=f"pages[{name!r}].width"),
            internal_positive_int(raw_page.get("height"), field=f"pages[{name!r}].height"),
            irreversible,
            tuple(unclassified),
            tuple(
                internal_parse_stage(stage, field=f"pages[{name!r}].jpx_stages[{index}]")
                for index, stage in enumerate(raw_stages)
            ),
            raster_path,
        )
    return RasterObservation(platform_id, revision, runtime, codec_stack, pages)


def internal_load_observed_raster(page: ObservedPage) -> numpy.ndarray[Any, Any]:
    path = page.raster_path
    if path is None or not path.is_file():
        raise RasterSnapshotError(f"portable observation raster is missing: {path}")
    try:
        raster = numpy.asarray(imagecodecs.png_decode(path.read_bytes()))
    except Exception as exc:
        raise RasterSnapshotError(f"could not decode observation raster: {path}") from exc
    if raster.dtype != numpy.dtype(numpy.uint8) or raster.shape != (
        page.height,
        page.width,
        4,
    ):
        raise RasterSnapshotError(f"observation raster has invalid layout: {path}")
    digest = internal_sha256(raster.tobytes())
    if digest != page.digest:
        raise RasterSnapshotError(
            f"observation raster digest mismatch: expected={page.digest}, actual={digest}"
        )
    return raster


def internal_rgb_envelope(
    rasters: list[numpy.ndarray[Any, Any]],
) -> tuple[tuple[int, int, int], ...]:
    shapes = {tuple(raster.shape) for raster in rasters}
    if len(shapes) != 1:
        raise RasterSnapshotError(f"portable raster shapes differ: {sorted(shapes)}")
    stacked = numpy.stack([raster.reshape(-1) for raster in rasters])
    minimum = stacked.min(axis=0)
    maximum = stacked.max(axis=0)
    changed = numpy.flatnonzero(minimum != maximum)
    alpha = tuple(int(offset) for offset in changed if int(offset) % 4 == 3)
    if alpha:
        raise RasterSnapshotError(
            f"portable raster alpha differs at offsets {internal_preview(alpha)}"
        )
    return tuple((int(offset), int(minimum[offset]), int(maximum[offset])) for offset in changed)


def merge_observations(
    directories: Iterable[pathlib.Path], *, snapshot_path: pathlib.Path = SNAPSHOT
) -> RasterSnapshot:
    """Merge a complete platform set and atomically update golden files."""
    observations = [load_observation(directory) for directory in directories]
    by_platform = {observation.platform_id: observation for observation in observations}
    if len(by_platform) != len(observations):
        raise RasterSnapshotError("duplicate platform observations")
    if set(by_platform) != set(SUPPORTED_PLATFORMS):
        raise RasterSnapshotError(f"observations must cover exactly {sorted(SUPPORTED_PLATFORMS)}")
    revisions = {observation.revision for observation in observations}
    if len(revisions) != 1:
        raise RasterSnapshotError(f"observation revisions differ: {sorted(revisions)}")
    inventories = {frozenset(observation.pages) for observation in observations}
    if len(inventories) != 1:
        raise RasterSnapshotError("observation corpus inventories differ")

    exact: dict[str, str] = {}
    portable: dict[str, PortableRasterSnapshot] = {}
    reference_directory = snapshot_path.parent / REFERENCE_DIRECTORY.name
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="raster-golden-", dir=snapshot_path.parent) as name:
        temp_root = pathlib.Path(name)
        for pdf_name in sorted(next(iter(inventories))):
            pages = {
                platform_id: observation.pages[pdf_name]
                for platform_id, observation in by_platform.items()
            }
            policies = {page.irreversible for page in pages.values()}
            if len(policies) != 1:
                raise RasterSnapshotError(f"JPX policy differs across platforms for {pdf_name}")
            unclassified = {
                platform_id: page.unclassified
                for platform_id, page in pages.items()
                if page.unclassified
            }
            if unclassified:
                raise RasterSnapshotError(f"unclassified JPX in {pdf_name}: {unclassified}")
            stage_sources = {
                tuple((stage.location, stage.raw_sha256, stage.shape) for stage in page.jpx_stages)
                for page in pages.values()
            }
            if len(stage_sources) != 1:
                raise RasterSnapshotError(
                    f"JPX source inventory differs across platforms for {pdf_name}"
                )
            if not next(iter(policies)):
                digests = {page.digest for page in pages.values()}
                if len(digests) != 1:
                    variants = {key: page.digest for key, page in pages.items()}
                    raise RasterSnapshotError(
                        f"non-JPX raster differs across platforms for {pdf_name}: {variants}"
                    )
                exact[pdf_name] = next(iter(digests))
                continue

            rasters = {
                platform_id: internal_load_observed_raster(page)
                for platform_id, page in pages.items()
            }
            raster_digests = {page.digest for page in pages.values()}
            decoded_digests = {
                tuple(stage.decoded_sha256 for stage in page.jpx_stages) for page in pages.values()
            }
            if len(raster_digests) > 1 and len(decoded_digests) == 1:
                raise RasterSnapshotError(
                    f"portable raster differs without JPX decoder variation for {pdf_name}"
                )
            anchor_platform = min(pages, key=lambda item: pages[item].digest)
            anchor_page = pages[anchor_platform]
            anchor = rasters[anchor_platform]
            reference = internal_reference_name(pdf_name, anchor_page.digest)
            temporary_reference = temp_root / pathlib.Path(*pathlib.PurePosixPath(reference).parts)
            temporary_reference.parent.mkdir(parents=True, exist_ok=True)
            temporary_reference.write_bytes(bytes(imagecodecs.png_encode(anchor, level=9)))
            portable[pdf_name] = PortableRasterSnapshot(
                anchor_page.digest,
                reference,
                internal_snapshot_reference_path(reference, snapshot_path=snapshot_path),
                {platform_id: page.digest for platform_id, page in pages.items()},
                internal_rgb_envelope(list(rasters.values())),
            )

        updated = RasterSnapshot(exact, portable, dict(EXPECTED_CODEC_STACK))
        reference_directory.mkdir(parents=True, exist_ok=True)
        expected_references: set[pathlib.Path] = set()
        for entry in portable.values():
            source = temp_root / pathlib.Path(*pathlib.PurePosixPath(entry.reference).parts)
            destination = internal_snapshot_reference_path(
                entry.reference, snapshot_path=snapshot_path
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            expected_references.add(destination.resolve())
        temporary_snapshot = snapshot_path.with_suffix(".json.tmp")
        temporary_snapshot.write_text(internal_snapshot_json(updated))
        os.replace(temporary_snapshot, snapshot_path)
        for reference_path in reference_directory.glob("*.png"):
            if reference_path.resolve() not in expected_references:
                reference_path.unlink()
    return updated


def internal_parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="collect one platform observation")
    collect.add_argument("--platform-id", choices=sorted(SUPPORTED_PLATFORMS), required=True)
    collect.add_argument("--output", type=pathlib.Path, required=True)
    collect.add_argument("--revision", default=None)
    collect.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    merge = subparsers.add_parser("merge", help="merge all supported observations")
    merge.add_argument("--observation", type=pathlib.Path, action="append", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = internal_parse_args(argv)
    try:
        if args.command == "collect":
            collect_observation(
                args.output,
                platform_id=args.platform_id,
                revision=args.revision or internal_git_revision(),
                workers=args.workers,
            )
        else:
            started = time.monotonic()
            try:
                existing = load_snapshot()
            except RasterSnapshotError:
                existing = None
            updated = merge_observations(args.observation)
            variants = {
                name: len(set(entry.variants.values())) for name, entry in updated.portable.items()
            }
            ranges = {name: len(entry.rgb_ranges) for name, entry in updated.portable.items()}
            maximum_spans = {
                name: max(
                    (maximum - minimum for _, minimum, maximum in entry.rgb_ranges),
                    default=0,
                )
                for name, entry in updated.portable.items()
            }
            changed_exact = (
                sorted(
                    name
                    for name, digest in updated.exact.items()
                    if existing.exact.get(name) != digest
                )
                if existing is not None
                else sorted(updated.exact)
            )
            print(
                f"updated {SNAPSHOT} in {time.monotonic() - started:.1f}s; "
                f"exact={len(updated.exact)}, portable={len(updated.portable)}; "
                f"changed_exact={changed_exact}; variants={variants}; "
                f"rgb_ranges={ranges}; maximum_spans={maximum_spans}"
            )
    except (OSError, RasterSnapshotError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0

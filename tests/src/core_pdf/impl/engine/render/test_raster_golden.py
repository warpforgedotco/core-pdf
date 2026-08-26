from __future__ import annotations

import hashlib
import json
import pathlib
import types
import zlib

import imagecodecs
import numpy
import pytest

from core_pdf.impl.engine.render.raster_image import RasterImage
from scripts import raster_golden
from scripts.raster_golden import (
    CANONICAL_SOURCE,
    RasterSnapshot,
    RasterSnapshotError,
    RasterTolerance,
    TolerantRasterSnapshot,
    compare_raster_to_reference,
    internal_scan_jpx_policy,
    is_canonical_regeneration_platform,
    load_reference_raster,
    load_snapshot,
    raster_snapshot_failure,
)


def internal_raster(samples: numpy.ndarray) -> RasterImage:
    height, width, channels = samples.shape
    return RasterImage(samples, width, height, channels)


def test_raster_difference_accepts_each_limit_at_its_boundary() -> None:
    reference = numpy.zeros((2, 2, 4), dtype=numpy.uint8)
    reference[:, :, 3] = 255
    actual = reference.copy()
    actual[0, 0, 0] = 1
    actual[1, 1, 2] = 2

    difference = compare_raster_to_reference(internal_raster(actual), reference)
    limits = RasterTolerance(
        max_rgb_channel_delta=2,
        max_changed_rgb_samples=2,
        max_total_rgb_delta=3,
    )

    assert difference.within(limits)
    assert difference.max_rgb_channel_delta == 2
    assert difference.changed_rgb_samples == 2
    assert difference.changed_rgb_pixels == 2
    assert difference.total_rgb_delta == 3
    assert not difference.within(RasterTolerance(1, 2, 3))
    assert not difference.within(RasterTolerance(2, 1, 3))
    assert not difference.within(RasterTolerance(2, 2, 2))


def test_raster_difference_requires_exact_alpha() -> None:
    reference = numpy.zeros((1, 1, 4), dtype=numpy.uint8)
    actual = reference.copy()
    actual[0, 0, 3] = 1

    difference = compare_raster_to_reference(internal_raster(actual), reference)

    assert difference.alpha_changed_samples == 1
    assert not difference.within(RasterTolerance(0, 0, 0))


def test_raster_difference_reports_shape_before_pixel_metrics() -> None:
    actual = internal_raster(numpy.zeros((1, 2, 4), dtype=numpy.uint8))
    reference = numpy.zeros((2, 1, 4), dtype=numpy.uint8)

    difference = compare_raster_to_reference(actual, reference)

    assert difference.actual_shape == (1, 2, 4)
    assert difference.reference_shape == (2, 1, 4)
    assert difference.max_rgb_channel_delta is None
    assert not difference.within(RasterTolerance(255, 100, 100))


@pytest.mark.parametrize(
    "reference",
    [
        numpy.zeros((1, 1, 4), dtype=numpy.uint16),
        numpy.zeros((1, 1, 3), dtype=numpy.uint8),
    ],
)
def test_raster_difference_rejects_invalid_reference_layout(reference: numpy.ndarray) -> None:
    actual = internal_raster(numpy.zeros((1, 1, 4), dtype=numpy.uint8))

    with pytest.raises(RasterSnapshotError):
        compare_raster_to_reference(actual, reference)


@pytest.mark.parametrize(
    ("system", "machine", "operating_system", "imagecodecs_version", "expected"),
    [
        ("Linux", "x86_64", "ubuntu-24.04", "2026.6.26", True),
        ("linux", "AMD64", "ubuntu-24.04", "2026.6.26", True),
        ("Linux", "aarch64", "ubuntu-24.04", "2026.6.26", False),
        ("Linux", "x86_64", "debian-13", "2026.6.26", False),
        ("Linux", "x86_64", "ubuntu-24.04", "2027.1.1", False),
        ("Darwin", "x86_64", "darwin", "2026.6.26", False),
    ],
)
def test_canonical_raster_regeneration_platform(
    system: str,
    machine: str,
    operating_system: str,
    imagecodecs_version: str,
    *,
    expected: bool,
) -> None:
    assert (
        is_canonical_regeneration_platform(
            system=system,
            machine=machine,
            operating_system=operating_system,
            imagecodecs_version=imagecodecs_version,
            jpeg2000_version="openjpeg 2.5.4",
        )
        is expected
    )


def test_jpx_policy_scans_filter_chains_nested_items_and_soft_masks() -> None:
    source = numpy.arange(256, dtype=numpy.uint8).reshape(16, 16)
    jpx = bytes(imagecodecs.jpeg2k_encode(source, reversible=False))
    compressed = zlib.compress(jpx)
    jpx_dictionary = {"Filter": ["FlateDecode", "JPXDecode"]}
    nested = {
        "raw_data": compressed,
        "dictionary": jpx_dictionary,
    }
    item = types.SimpleNamespace(
        data={
            "items": [nested],
            "dictionary": {
                "__soft_mask_raw_data__": compressed,
                "__soft_mask_dictionary__": jpx_dictionary,
            },
        }
    )

    scan = internal_scan_jpx_policy([item])

    assert scan.irreversible
    assert not scan.unclassified


def test_jpx_policy_fails_closed_for_an_unclassified_declared_stream() -> None:
    item = types.SimpleNamespace(
        data={"raw_data": b"invalid", "dictionary": {"Filter": "JPXDecode"}}
    )

    scan = internal_scan_jpx_policy([item])

    assert not scan.irreversible
    assert scan.unclassified == ("item 0",)


def test_snapshot_references_resolve_relative_to_a_custom_manifest(
    tmp_path: pathlib.Path,
) -> None:
    reference_directory = tmp_path / "first_page_scale1_refs"
    reference_directory.mkdir()
    reference = numpy.zeros((1, 2, 4), dtype=numpy.uint8)
    reference_path = reference_directory / "sample.png"
    reference_path.write_bytes(bytes(imagecodecs.png_encode(reference)))
    digest = hashlib.sha256(reference.tobytes()).hexdigest()
    manifest = tmp_path / "custom.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 2,
                "canonical_source": CANONICAL_SOURCE,
                "render": {
                    "page": 0,
                    "scale": 1.0,
                    "background": [255, 255, 255, 255],
                },
                "exact": {},
                "tolerant": {
                    "sample.pdf": {
                        "canonical_sha256": digest,
                        "reference": "first_page_scale1_refs/sample.png",
                        "limits": {
                            "max_rgb_channel_delta": 0,
                            "max_changed_rgb_samples": 0,
                            "max_total_rgb_delta": 0,
                        },
                    }
                },
            }
        )
    )

    snapshot = load_snapshot(manifest)

    entry = snapshot.tolerant["sample.pdf"]
    assert entry.reference_path == reference_path
    numpy.testing.assert_array_equal(load_reference_raster(entry), reference)


def test_tolerant_exact_hash_fast_path_still_requires_the_reference_shape(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = internal_raster(numpy.zeros((1, 2, 4), dtype=numpy.uint8))
    reference = numpy.zeros((2, 1, 4), dtype=numpy.uint8)
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(bytes(imagecodecs.png_encode(reference)))
    digest = hashlib.sha256(reference.tobytes()).hexdigest()
    entry = TolerantRasterSnapshot(
        canonical_sha256=digest,
        reference="first_page_scale1_refs/reference.png",
        reference_path=reference_path,
        limits=RasterTolerance(0, 0, 0),
    )
    snapshot = RasterSnapshot({}, {"sample.pdf": entry}, CANONICAL_SOURCE)

    def render_actual(_pdf: pathlib.Path) -> RasterImage:
        return actual

    monkeypatch.setattr(raster_golden, "render_first_page", render_actual)

    assert raster_snapshot_failure(pathlib.Path("sample.pdf"), snapshot) == (
        "shape changed: actual=(1, 2, 4), reference=(2, 1, 4)"
    )

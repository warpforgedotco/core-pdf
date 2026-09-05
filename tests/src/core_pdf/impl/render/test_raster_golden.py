from __future__ import annotations

import hashlib
import json
import pathlib
import types
import zlib

import imagecodecs
import numpy
import pytest

from core_pdf.impl.render.model import ImagePaintItem, RasterImage
from core_pdf.impl.spec.s_08_graphics.image_decode import ImageSource, SoftMask
from scripts.raster_cover import greedy_cover
from scripts.raster_golden import (
    EXPECTED_CODEC_STACK,
    PortableRasterSnapshot,
    RasterSnapshot,
    RasterSnapshotError,
    compare_raster_to_reference,
    internal_rgb_envelope,
    internal_scan_jpx_policy,
    internal_snapshot_json,
    load_reference_raster,
    load_snapshot,
    merge_observations,
)


def internal_raster(samples: numpy.ndarray) -> RasterImage:
    height, width, channels = samples.shape
    return RasterImage(samples, width, height, channels)


def internal_digest(samples: numpy.ndarray) -> str:
    return hashlib.sha256(samples.tobytes()).hexdigest()


def internal_write_observation(
    root: pathlib.Path,
    platform_id: str,
    portable: numpy.ndarray,
    *,
    revision: str = "revision",
    exact_digest: str = "a" * 64,
) -> pathlib.Path:
    directory = root / platform_id
    rasters = directory / "rasters"
    rasters.mkdir(parents=True)
    raster_path = rasters / "portable.png"
    raster_path.write_bytes(bytes(imagecodecs.png_encode(portable)))
    system, architecture = {
        "linux-x86_64": ("linux", "x86_64"),
        "macos-arm64": ("darwin", "arm64"),
    }[platform_id]
    data = {
        "version": 1,
        "platform_id": platform_id,
        "revision": revision,
        "runtime": {
            "system": system,
            "operating_system": system,
            "architecture": architecture,
        },
        "codec_stack": EXPECTED_CODEC_STACK,
        "render": {"page": 0, "scale": 1.0, "background": [255, 255, 255, 255]},
        "pages": {
            "exact.pdf": {
                "sha256": exact_digest,
                "width": 1,
                "height": 1,
                "irreversible_jpx": False,
                "unclassified_jpx": [],
                "jpx_stages": [],
            },
            "portable.pdf": {
                "sha256": internal_digest(portable),
                "width": int(portable.shape[1]),
                "height": int(portable.shape[0]),
                "irreversible_jpx": True,
                "unclassified_jpx": [],
                "jpx_stages": [
                    {
                        "location": "item 0",
                        "raw_sha256": "b" * 64,
                        "decoded_sha256": ("c" if platform_id == "linux-x86_64" else "d") * 64,
                        "shape": [1, 1, 3],
                    }
                ],
                "raster": "rasters/portable.png",
            },
        },
    }
    (directory / "observation.json").write_text(json.dumps(data))
    return directory


def test_raster_cover_seeds_portable_documents_without_choosing_them() -> None:
    line_a = ("render.py", 1)
    line_b = ("render.py", 2)
    line_c = ("render.py", 3)
    per_document = {
        "portable.pdf": {line_a},
        "first.pdf": {line_a, line_b},
        "second.pdf": {line_c},
    }

    chosen = greedy_cover(per_document, precovered_names=frozenset({"portable.pdf"}))

    assert chosen == ["first.pdf", "second.pdf"]


def test_sparse_rgb_envelope_accepts_only_calibrated_samples() -> None:
    reference = numpy.zeros((1, 2, 4), dtype=numpy.uint8)
    reference[:, :, 3] = 255
    actual = reference.copy()
    actual.reshape(-1)[0] = 2
    actual.reshape(-1)[6] = 4
    ranges = ((0, 0, 2), (6, 0, 4))

    difference = compare_raster_to_reference(internal_raster(actual), reference, ranges)

    assert difference.accepted
    escaped = actual.copy()
    escaped.reshape(-1)[1] = 1
    assert compare_raster_to_reference(
        internal_raster(escaped), reference, ranges
    ).unexpected_rgb_offsets == (1,)
    exceeded = actual.copy()
    exceeded.reshape(-1)[0] = 3
    assert compare_raster_to_reference(
        internal_raster(exceeded), reference, ranges
    ).out_of_range == ((0, 3, 0, 2),)


def test_sparse_rgb_envelope_requires_exact_alpha_and_shape() -> None:
    reference = numpy.zeros((1, 1, 4), dtype=numpy.uint8)
    alpha = reference.copy()
    alpha[0, 0, 3] = 1

    difference = compare_raster_to_reference(internal_raster(alpha), reference, ())

    assert difference.alpha_offsets == (3,)
    assert not difference.accepted
    shaped = compare_raster_to_reference(
        internal_raster(numpy.zeros((1, 2, 4), dtype=numpy.uint8)), reference, ()
    )
    assert shaped.actual_shape != shaped.reference_shape
    assert not shaped.accepted


@pytest.mark.parametrize(
    "reference",
    [
        numpy.zeros((1, 1, 4), dtype=numpy.uint16),
        numpy.zeros((1, 1, 3), dtype=numpy.uint8),
    ],
)
def test_raster_envelope_rejects_invalid_reference_layout(reference: numpy.ndarray) -> None:
    actual = internal_raster(numpy.zeros((1, 1, 4), dtype=numpy.uint8))
    with pytest.raises(RasterSnapshotError):
        compare_raster_to_reference(actual, reference, ())


def test_rgb_envelope_is_sparse_and_rejects_alpha_variation() -> None:
    first = numpy.zeros((1, 2, 4), dtype=numpy.uint8)
    second = first.copy()
    second.reshape(-1)[0] = 2
    second.reshape(-1)[6] = 7

    assert internal_rgb_envelope([first, second]) == ((0, 0, 2), (6, 0, 7))
    second.reshape(-1)[7] = 1
    with pytest.raises(RasterSnapshotError, match="alpha differs"):
        internal_rgb_envelope([first, second])


def test_merge_observations_builds_exact_and_portable_entries(tmp_path: pathlib.Path) -> None:
    linux = numpy.zeros((1, 2, 4), dtype=numpy.uint8)
    linux[:, :, 3] = 255
    macos = linux.copy()
    macos.reshape(-1)[0] = 2
    linux_dir = internal_write_observation(tmp_path, "linux-x86_64", linux)
    macos_dir = internal_write_observation(tmp_path, "macos-arm64", macos)
    snapshot_path = tmp_path / "snapshot" / "first_page_scale1.json"

    merged = merge_observations([macos_dir, linux_dir], snapshot_path=snapshot_path)

    assert merged.exact == {"exact.pdf": "a" * 64}
    assert merged.portable["portable.pdf"].rgb_ranges == ((0, 0, 2),)
    assert load_snapshot(snapshot_path) == merged


def test_merge_observations_rejects_non_jpx_platform_drift(tmp_path: pathlib.Path) -> None:
    raster = numpy.zeros((1, 1, 4), dtype=numpy.uint8)
    linux_dir = internal_write_observation(tmp_path, "linux-x86_64", raster)
    macos_dir = internal_write_observation(tmp_path, "macos-arm64", raster, exact_digest="e" * 64)

    with pytest.raises(RasterSnapshotError, match="non-JPX raster differs"):
        merge_observations(
            [linux_dir, macos_dir], snapshot_path=tmp_path / "snapshot" / "golden.json"
        )


def test_merge_observations_rejects_mismatched_revisions(tmp_path: pathlib.Path) -> None:
    raster = numpy.zeros((1, 1, 4), dtype=numpy.uint8)
    linux_dir = internal_write_observation(tmp_path, "linux-x86_64", raster)
    macos_dir = internal_write_observation(tmp_path, "macos-arm64", raster, revision="other")

    with pytest.raises(RasterSnapshotError, match="revisions differ"):
        merge_observations(
            [linux_dir, macos_dir], snapshot_path=tmp_path / "snapshot" / "golden.json"
        )


def test_jpx_policy_scans_filter_chains_nested_items_and_soft_masks() -> None:
    source = numpy.arange(256, dtype=numpy.uint8).reshape(16, 16)
    jpx = bytes(imagecodecs.jpeg2k_encode(source, reversible=False))
    compressed = zlib.compress(jpx)
    dictionary = {"Filter": ["FlateDecode", "JPXDecode"]}
    item = types.SimpleNamespace(
        data={
            "items": [{"raw_data": compressed, "dictionary": dictionary}],
            "dictionary": {},
            "soft_mask": SoftMask(compressed, dictionary),
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


def test_jpx_policy_scans_typed_image_sources() -> None:
    encoded = bytes(
        imagecodecs.jpeg2k_encode(
            numpy.arange(256, dtype=numpy.uint8).reshape(16, 16), reversible=False
        )
    )
    item = ImagePaintItem(
        paint_kind="image",
        seqno=0,
        bbox=None,
        source=ImageSource(encoded, {"Filter": "JPXDecode"}),
        quad=None,
        fill=None,
        fill_opacity=None,
        blend_mode=None,
        soft_mask_alpha=None,
        image_clip=None,
        source_metadata={},
    )
    assert internal_scan_jpx_policy([item]).irreversible


def test_snapshot_round_trip_uses_platform_neutral_sparse_envelope(
    tmp_path: pathlib.Path,
) -> None:
    reference_directory = tmp_path / "first_page_scale1_refs"
    reference_directory.mkdir()
    reference = numpy.zeros((1, 2, 4), dtype=numpy.uint8)
    reference[:, :, 3] = 255
    reference_path = reference_directory / "sample.png"
    reference_path.write_bytes(bytes(imagecodecs.png_encode(reference)))
    digest = internal_digest(reference)
    entry = PortableRasterSnapshot(
        digest,
        "first_page_scale1_refs/sample.png",
        reference_path,
        {"linux-x86_64": digest, "macos-arm64": "1" * 64},
        ((0, 0, 1),),
    )
    snapshot = RasterSnapshot({}, {"sample.pdf": entry}, dict(EXPECTED_CODEC_STACK))
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(internal_snapshot_json(snapshot))

    loaded = load_snapshot(manifest)

    assert loaded.codec_stack == EXPECTED_CODEC_STACK
    assert loaded.portable["sample.pdf"].rgb_ranges == ((0, 0, 1),)
    numpy.testing.assert_array_equal(
        load_reference_raster(loaded.portable["sample.pdf"]), reference
    )


@pytest.mark.parametrize(
    "ranges",
    [
        [[3, 0, 1]],
        [[0, 0, 1], [0, 0, 2]],
        [[0, 2, 1]],
        [[99, 0, 1]],
    ],
)
def test_snapshot_rejects_invalid_sparse_ranges(
    tmp_path: pathlib.Path, ranges: list[list[int]]
) -> None:
    reference_directory = tmp_path / "first_page_scale1_refs"
    reference_directory.mkdir()
    reference = numpy.zeros((1, 1, 4), dtype=numpy.uint8)
    reference_path = reference_directory / "sample.png"
    reference_path.write_bytes(bytes(imagecodecs.png_encode(reference)))
    digest = internal_digest(reference)
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 3,
                "codec_stack": EXPECTED_CODEC_STACK,
                "render": {"page": 0, "scale": 1.0, "background": [255, 255, 255, 255]},
                "exact": {},
                "portable": {
                    "sample.pdf": {
                        "reference_sha256": digest,
                        "reference": "first_page_scale1_refs/sample.png",
                        "variants": {"linux-x86_64": digest, "macos-arm64": digest},
                        "rgb_ranges": ranges,
                    }
                },
            }
        )
    )
    with pytest.raises(RasterSnapshotError):
        load_snapshot(manifest)

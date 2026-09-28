# SPDX-License-Identifier: AGPL-3.0-only

import gzip
import importlib
import pickle
import struct
from pathlib import Path

import numpy
import pytest

import core_pdf_cythonized
from core_pdf.impl.capture_glyphs import RunGeometry
from core_pdf.impl.graphics_stream_decoding import decode_jbig2
from core_pdf.impl.render_model import SoftMaskPlane

CYTHONIZED_TESTS = Path(__file__).parents[3] / "packages" / "core-pdf-cythonized" / "tests"
GLYPH_GEOMETRY_GOLDEN = pickle.loads(
    gzip.decompress((CYTHONIZED_TESTS / "glyph_geometry_golden.pkl.gz").read_bytes())
)

KERNEL_USES = [
    ("core_pdf.impl.render_target", "blend_normal_alpha_array_numpy"),
    ("core_pdf.impl.render_target", "composite_elementary_normal"),
    ("core_pdf.impl.render_target", "composite_knockout_group"),
    ("core_pdf.impl.render_target", "composite_masked_normal"),
    ("core_pdf.impl.render_target", "composite_normal_group"),
    ("core_pdf.impl.render_target", "fill_glyph_coverage_at"),
    ("core_pdf.impl.render_target", "fill_glyph_knockout_at"),
    ("core_pdf.impl.render_target", "fill_rect_coverage"),
    ("core_pdf.impl.render_target", "sample_opaque_pixels"),
    ("core_pdf.impl.render_target", "supersampled_coverage_plane"),
    ("core_pdf.impl.render_commands", "translated_outline_edges"),
    ("core_pdf.impl.capture_glyphs", "capture_horizontal_glyphs"),
    ("core_pdf.impl.fonts_cff_repair", "cell_distance_map"),
    ("core_pdf.impl.capture_records", "fill_edge_rows"),
    ("core_pdf.impl.fonts_raster_kernel", "glyph_bitmap_rows"),
    ("core_pdf.impl.render_blend", "blend_visible_rgba"),
    ("core_pdf.impl.fonts_program_cff", "type2_glyph_geometry"),
    ("core_pdf.impl.graphics_images", "interleave_soft_mask"),
    ("core_pdf.impl.graphics_stream_decoding", "decode_arithmetic_generic_template0"),
]

DELETED = [
    ("core_pdf.impl.render_paths", "signed_area_coverage"),
    ("core_pdf.impl.render_target", "signed_area_coverage"),
    ("core_pdf.impl.render_paths", "group_offsets"),
    ("core_pdf.impl.fonts_program_cff", "pure_cubic_sample_times"),
    ("core_pdf.impl.fonts_program_cff", "execute_type2_charstring"),
    ("core_pdf.impl.render_blend", "blend_normal_alpha_array_numpy"),
    ("core_pdf.impl.capture_recovery", "match_token"),
    ("core_pdf_spec.s_11_transparency.groups", "composite_knockout_element"),
]


@pytest.mark.parametrize(("module_name", "name"), KERNEL_USES)
def test_core_uses_the_kernel(module_name: str, name: str) -> None:
    module = importlib.import_module(module_name)
    assert getattr(module, name) is getattr(core_pdf_cythonized, name)


@pytest.mark.parametrize(("module_name", "name"), DELETED)
def test_the_python_a_kernel_replaced_stays_deleted(module_name: str, name: str) -> None:
    assert not hasattr(importlib.import_module(module_name), name)


def test_core_decodes_arithmetic_regions_with_the_kernel():
    page = struct.pack(">IBBBI", 1, 48, 0, 1, 19) + struct.pack(">IIIIBH", 8, 1, 0, 0, 0, 0)
    header = struct.pack(">IIiiBB", 8, 1, 0, 0, 0, 0) + bytes.fromhex("03 ff fd ff 02 fe fe fe")
    region = header + bytes.fromhex("ff ac")
    segment = struct.pack(">IBBBI", 2, 38, 0, 1, len(region)) + region
    end = struct.pack(">IBBBI", 3, 49, 0, 1, 0)
    assert decode_jbig2(page + segment + end, None) == b"\x00"


def test_a_plane_without_transfer_reads_as_its_float32_window():
    alpha = numpy.arange(256, dtype=numpy.uint8).reshape(16, 16)
    for table in (None, numpy.linspace(1, 0, 256, dtype=numpy.float32)):
        plane = SoftMaskPlane(alpha, table)
        window = (slice(2, 9), slice(3, 14))
        assert plane.values()[alpha[window]].tobytes() == plane[window].tobytes()


@pytest.mark.parametrize("index", range(0, len(GLYPH_GEOMETRY_GOLDEN), 3))
def test_the_run_unions_are_run_geometry_adds(index: int) -> None:
    kwargs, _ = GLYPH_GEOMETRY_GOLDEN[index]
    arguments = dict(kwargs)
    produced = core_pdf_cythonized.horizontal_glyph_geometry(
        arguments.pop("offsets"),
        arguments.pop("advances"),
        arguments.pop("glyph_boxes"),
        **arguments,
    )
    geometry = RunGeometry()
    for advance, ink in zip(produced[0], produced[3], strict=True):
        geometry.add(advance, ink, None)
    if not produced[0]:
        assert produced[6] is None
        assert produced[7] is None
        return
    assert repr(produced[6]) == repr(geometry.advance)
    assert repr(produced[7]) == repr(geometry.ink)

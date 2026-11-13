# SPDX-License-Identifier: AGPL-3.0-only

# Kernels load when first used (Python 3.15+).
from core_pdf_cythonized._cmap import scan_to_unicode_cmap
from core_pdf_cythonized._content import ContentScanner, NativeOperators
from core_pdf_cythonized._glyphs import (
    DECODED_GLYPH_FIELDS,
    OBSERVATION_FIELDS,
    SlotLayout,
    capture_horizontal_glyphs,
    horizontal_glyph_geometry,
    untrack_tuple,
)
from core_pdf_cythonized._markers import (
    markers_present,
    regions_with_markers,
    security_marker_present,
)
from core_pdf_cythonized._objects import ObjectScanner
lazy from core_pdf_cythonized._area_gaps import AreaGapQueue
lazy from core_pdf_cythonized._bezier import cubic_sample_times
lazy from core_pdf_cythonized._blend import blend_normal_alpha_array_numpy
lazy from core_pdf_cythonized._blend_visible import blend_visible_rgba
lazy from core_pdf_cythonized._chars import no_char, pdfminer_chars, plumber_char_dicts
lazy from core_pdf_cythonized._composite import (
    composite_elementary_normal,
    composite_masked_normal,
    composite_normal_group,
)
lazy from core_pdf_cythonized._coverage import (
    fill_glyph_coverage,
    fill_glyph_coverage_at,
    fill_glyph_knockout,
    fill_glyph_knockout_at,
    glyph_coverage_plane,
    signed_area_coverage,
)
lazy from core_pdf_cythonized._coverage_blend import blend_coverage_counts
lazy from core_pdf_cythonized._distinct_rows import (
    code_presence,
    distinct_uint16_rows,
    gather_uint8_rows,
)
lazy from core_pdf_cythonized._downsample import box_downsample_blocks
lazy from core_pdf_cythonized._features import cell_distance_map, glyph_feature_cells
lazy from core_pdf_cythonized._glyph_bitmap import glyph_bitmap_rows
lazy from core_pdf_cythonized._jbig2 import decode_arithmetic_generic_template0
lazy from core_pdf_cythonized._knockout import (
    composite_elementary_knockout,
    composite_knockout_element,
    composite_knockout_group,
)
lazy from core_pdf_cythonized._nonisolated import composite_nonisolated_blend
lazy from core_pdf_cythonized._outline import (
    outline_coordinate_arrays,
    outline_edges,
    translated_outline_edges,
)
lazy from core_pdf_cythonized._paths import fill_edge_rows, flatten_path_commands, path_bounds
lazy from core_pdf_cythonized._rect import fill_rect_coverage
lazy from core_pdf_cythonized._sample_blit import (
    alpha_channel,
    interleave_soft_mask,
    sample_opaque_pixels,
)
lazy from core_pdf_cythonized._segments import merge_collinear_rows
lazy from core_pdf_cythonized._serialize import ObjectWriter, collect_references
lazy from core_pdf_cythonized._shading import shading_blend, shading_t, shading_values
lazy from core_pdf_cythonized._source_plane import accumulate_source_plane
lazy from core_pdf_cythonized._stroke import stroke_polylines
lazy from core_pdf_cythonized._stroke_segment import stroke_segment_samples
lazy from core_pdf_cythonized._supersample import supersampled_coverage_plane
lazy from core_pdf_cythonized._text_array import RunAccumulator, show_text_array
lazy from core_pdf_cythonized._truetype import truetype_contours
lazy from core_pdf_cythonized._type1 import decrypt_type1, type1_glyph_bounds
lazy from core_pdf_cythonized._type2 import type2_glyph_geometry
lazy from core_pdf_cythonized._xref_headers import object_headers_match

__all__ = (
    "no_char",
    "pdfminer_chars",
    "plumber_char_dicts",
    "ObjectWriter",
    "collect_references",
    "AreaGapQueue",
    "cell_distance_map",
    "glyph_feature_cells",
    "merge_collinear_rows",
    "markers_present",
    "regions_with_markers",
    "security_marker_present",
    "DECODED_GLYPH_FIELDS",
    "OBSERVATION_FIELDS",
    "SlotLayout",
    "capture_horizontal_glyphs",
    "show_text_array",
    "RunAccumulator",
    "untrack_tuple",
    "ContentScanner",
    "NativeOperators",
    "ObjectScanner",
    "accumulate_source_plane",
    "alpha_channel",
    "blend_coverage_counts",
    "blend_normal_alpha_array_numpy",
    "blend_visible_rgba",
    "code_presence",
    "box_downsample_blocks",
    "composite_elementary_knockout",
    "composite_elementary_normal",
    "composite_knockout_element",
    "composite_knockout_group",
    "composite_masked_normal",
    "composite_nonisolated_blend",
    "composite_normal_group",
    "cubic_sample_times",
    "decode_arithmetic_generic_template0",
    "decrypt_type1",
    "type1_glyph_bounds",
    "distinct_uint16_rows",
    "fill_edge_rows",
    "fill_glyph_coverage",
    "fill_glyph_coverage_at",
    "fill_glyph_knockout",
    "fill_glyph_knockout_at",
    "fill_rect_coverage",
    "flatten_path_commands",
    "gather_uint8_rows",
    "glyph_bitmap_rows",
    "glyph_coverage_plane",
    "horizontal_glyph_geometry",
    "interleave_soft_mask",
    "object_headers_match",
    "outline_coordinate_arrays",
    "outline_edges",
    "path_bounds",
    "sample_opaque_pixels",
    "scan_to_unicode_cmap",
    "shading_blend",
    "shading_t",
    "shading_values",
    "signed_area_coverage",
    "stroke_polylines",
    "stroke_segment_samples",
    "supersampled_coverage_plane",
    "translated_outline_edges",
    "truetype_contours",
    "type2_glyph_geometry",
)

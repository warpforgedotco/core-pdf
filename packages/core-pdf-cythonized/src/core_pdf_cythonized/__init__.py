# SPDX-License-Identifier: AGPL-3.0-only


from core_pdf_cythonized._bezier import cubic_sample_times
from core_pdf_cythonized._blend import blend_normal_alpha_array_numpy
from core_pdf_cythonized._blend_visible import blend_visible_rgba
from core_pdf_cythonized._cmap import scan_to_unicode_cmap
from core_pdf_cythonized._composite import (
    composite_elementary_normal,
    composite_masked_normal,
    composite_normal_group,
)
from core_pdf_cythonized._content import ContentScanner
from core_pdf_cythonized._coverage import (
    fill_glyph_coverage,
    fill_glyph_knockout,
    glyph_coverage_plane,
    signed_area_coverage,
)
from core_pdf_cythonized._coverage_blend import blend_coverage_counts
from core_pdf_cythonized._distinct_rows import (
    code_presence,
    distinct_uint16_rows,
    gather_uint8_rows,
)
from core_pdf_cythonized._downsample import box_downsample_blocks
from core_pdf_cythonized._glyph_bitmap import glyph_bitmap_rows
from core_pdf_cythonized._glyphs import horizontal_glyph_geometry
from core_pdf_cythonized._jbig2 import decode_arithmetic_generic_template0
from core_pdf_cythonized._knockout import (
    composite_elementary_knockout,
    composite_knockout_element,
    composite_knockout_group,
)
from core_pdf_cythonized._objects import ObjectScanner
from core_pdf_cythonized._outline import outline_edges, translated_outline_edges
from core_pdf_cythonized._paths import fill_edge_rows, flatten_path_commands, path_bounds
from core_pdf_cythonized._rect import fill_rect_coverage
from core_pdf_cythonized._sample_blit import (
    alpha_channel,
    interleave_soft_mask,
    sample_opaque_pixels,
)
from core_pdf_cythonized._shading import shading_blend, shading_t, shading_values
from core_pdf_cythonized._source_plane import accumulate_source_plane
from core_pdf_cythonized._stroke import stroke_polylines
from core_pdf_cythonized._stroke_segment import stroke_segment_samples
from core_pdf_cythonized._supersample import supersampled_coverage_plane
from core_pdf_cythonized._truetype import truetype_contours
from core_pdf_cythonized._type1 import decrypt_type1
from core_pdf_cythonized._type2 import type2_glyph_geometry
from core_pdf_cythonized._xref_headers import object_headers_match

__all__ = (
    "ContentScanner",
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
    "composite_normal_group",
    "cubic_sample_times",
    "decode_arithmetic_generic_template0",
    "decrypt_type1",
    "distinct_uint16_rows",
    "fill_edge_rows",
    "fill_glyph_coverage",
    "fill_glyph_knockout",
    "fill_rect_coverage",
    "flatten_path_commands",
    "gather_uint8_rows",
    "glyph_bitmap_rows",
    "glyph_coverage_plane",
    "horizontal_glyph_geometry",
    "interleave_soft_mask",
    "object_headers_match",
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

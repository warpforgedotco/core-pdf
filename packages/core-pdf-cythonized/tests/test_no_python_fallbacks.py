# SPDX-License-Identifier: AGPL-3.0-only

"""The Python each kernel replaced stays deleted, so nothing can diverge from it.

The modules checked belong to core and spec. A freshly built wheel's tests run
with neither installed, so a case skips when its distribution is absent; once
the distribution imports, a missing module is a failure, not a skip.
"""

import importlib

import pytest

DELETED = [
    ("core_pdf.impl.render_paths", "signed_area_coverage"),
    # The device-space entry point is the one the golden vectors pin; core
    # reaches the shared core through the fused glyph front end instead.
    ("core_pdf.impl.render_target", "signed_area_coverage"),
    ("core_pdf.impl.render_paths", "group_offsets"),
    ("core_pdf.impl.fonts_font_program", "pure_cubic_sample_times"),
    ("core_pdf.impl.fonts_font_program", "execute_type2_charstring"),
    ("core_pdf.impl.render_blend", "blend_normal_alpha_array_numpy"),
    ("core_pdf.impl.capture_recovery", "match_token"),
    ("core_pdf_spec.s_11_transparency.groups", "composite_knockout_element"),
]


@pytest.mark.parametrize(("module_name", "name"), DELETED)
def test_the_python_a_kernel_replaced_stays_deleted(module_name: str, name: str) -> None:
    pytest.importorskip(module_name.partition(".")[0])
    assert not hasattr(importlib.import_module(module_name), name)

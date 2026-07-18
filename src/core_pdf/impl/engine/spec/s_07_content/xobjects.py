# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any

from core_layout.impl.layout.geometry import RectBox

from core_pdf.impl.engine.spec.s_07_content.capture import CapturedDrawing
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_08_graphics.matrix import (
    IDENTITY_MATRIX,
    Matrix,
)
from core_pdf.impl.objects import PdfReference, PdfStream


class XObjectMixin:
    """Image and form XObject methods for the concrete TextState class."""

    __slots__ = ()

    def op_Do(self: Any, o, d):
        if not o:
            return
        self.append_xobject(o[0], d)

    def append_xobject(self: Any, name_obj: Any, depth: int) -> None:
        name = self.resolve_name(name_obj)
        if not name:
            return
        xobjects = lookup_dict_key(self.resources, "XObject")
        raw_xobj = lookup_dict_key(xobjects, name) if isinstance(xobjects, dict) else None
        stream_key = (
            ("ref", raw_xobj.object_number, raw_xobj.generation_number)
            if isinstance(raw_xobj, PdfReference)
            else None
        )
        xobj = self.resolve(raw_xobj) if raw_xobj is not None else None
        if xobj is None:
            xobj = self.lookup_page_resource("XObject", name)
        if not isinstance(xobj, PdfStream):
            return
        xobj_dict = xobj.dictionary
        subtype = self.resolve_name(lookup_dict_key(xobj_dict, "Subtype"))
        if self.resolve_name(lookup_dict_key(xobj_dict, "Type")) == "ObjStm":
            return
        if subtype == "Image":
            if self.capture_graphics and self.is_graphics_visible():
                width = self.document.resolver.resolve_int(lookup_dict_key(xobj_dict, "Width")) or 0
                height = (
                    self.document.resolver.resolve_int(lookup_dict_key(xobj_dict, "Height")) or 0
                )
                bbox = None
                quad = None
                if width > 0 and height > 0:
                    points = (
                        self.transform_point(0.0, 0.0),
                        self.transform_point(1.0, 0.0),
                        self.transform_point(0.0, 1.0),
                        self.transform_point(1.0, 1.0),
                    )
                    quad = points
                    xs = [point[0] for point in points]
                    ys = [point[1] for point in points]
                    bbox = RectBox(
                        min(xs),
                        min(ys),
                        max(xs),
                        max(ys),
                    )
                smask_alpha = None
                soft_mask_raw_data = None
                soft_mask_dictionary = None
                smask = lookup_dict_key(xobj_dict, "SMask")
                if smask is not None:
                    smask_stream = self.document.resolver.resolve(smask)
                    if isinstance(smask_stream, PdfStream):
                        smask_dict = (
                            self.document.resolver.resolve_dict(smask_stream.dictionary) or {}
                        )
                        smask_data = getattr(smask_stream, "raw_data", b"")
                        soft_mask_raw_data = smask_data
                        soft_mask_dictionary = dict(smask_dict)
                        width = (
                            self.document.resolver.resolve_int(lookup_dict_key(smask_dict, "Width"))
                            or 0
                        )
                        height = (
                            self.document.resolver.resolve_int(
                                lookup_dict_key(smask_dict, "Height")
                            )
                            or 0
                        )
                        if width > 0 and height > 0 and smask_data:
                            total = min(len(smask_data), width * height)
                            if total > 0:
                                smask_alpha = sum(smask_data[:total]) / (255.0 * total)
                self.drawings.append(
                    CapturedDrawing(
                        seqno=self.sequence,
                        fill=None,
                        fill_opacity=None,
                        blend_mode=self.blend_mode,
                        dash_pattern=self.transformed_dash_pattern(),
                        soft_mask_alpha=smask_alpha,
                        kind="image",
                        items=[("quad", quad)] if quad is not None else [],
                        bbox=bbox,
                    )
                )
                self.drawings[-1].raw_data = getattr(xobj, "raw_data", b"")
                self.drawings[-1].dictionary = dict(xobj_dict)
                if soft_mask_raw_data is not None:
                    self.drawings[-1].dictionary["__soft_mask_raw_data__"] = soft_mask_raw_data
                    self.drawings[-1].dictionary["__soft_mask_dictionary__"] = (
                        soft_mask_dictionary or {}
                    )
            return
        if subtype != "Form":
            return
        group_alpha = None
        group = lookup_dict_key(xobj_dict, "Group")
        if group is not None:
            group_dict = self.document.resolver.resolve_dict(group)
            if (
                isinstance(group_dict, dict)
                and self.resolve_name(lookup_dict_key(group_dict, "S")) == "Transparency"
            ):
                group_alpha_val = self.document.resolver.resolve_float(
                    lookup_dict_key(group_dict, "ca"), default=None
                )
                if group_alpha_val is not None:
                    group_alpha = max(0.0, min(1.0, group_alpha_val))
        raw_resources = lookup_dict_key(xobj_dict, "Resources")
        resources = (
            raw_resources if isinstance(raw_resources, dict) else self.resolve_dict(raw_resources)
        ) or self.resources
        xobj_matrix = lookup_dict_key(xobj_dict, "Matrix")
        if isinstance(xobj_matrix, (list, tuple)) and len(xobj_matrix) > 6:
            xobj_matrix = xobj_matrix[:6]
        nested_ctm = (
            Matrix.from_operand(xobj_matrix) if xobj_matrix is not None else IDENTITY_MATRIX
        ).multiply(self.ctm)
        form_bbox = self.document.resolver.resolve_box(lookup_dict_key(xobj_dict, "BBox"))
        transformed_form_bbox = (
            transform_bbox(form_bbox, nested_ctm) if form_bbox is not None else None
        )
        self.queue_stream(
            xobj,
            resources,
            nested_ctm,
            depth + 1,
            clip_bbox=transformed_form_bbox,
            group_alpha=group_alpha,
            stream_key=stream_key,
            swallow_parse_errors=True,
        )


def transform_bbox(
    bbox: tuple[float, float, float, float], matrix: Matrix
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    a, b, c, d, e, f = matrix
    points = (
        (x0 * a + y0 * c + e, x0 * b + y0 * d + f),
        (x1 * a + y0 * c + e, x1 * b + y0 * d + f),
        (x0 * a + y1 * c + e, x0 * b + y1 * d + f),
        (x1 * a + y1 * c + e, x1 * b + y1 * d + f),
    )
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


__all__ = ("XObjectMixin",)

# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_document.protocols import LayersDocumentProtocol
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.objects import PdfReference


class LayersMixin:
    __slots__ = ()

    oc_layers: dict[str, bool] | None

    def ocg_key(self, ref: object, resolved: object) -> tuple[int, int] | int | None:
        if isinstance(ref, PdfReference):
            return (ref.object_number, ref.generation_number)
        if isinstance(resolved, dict):
            return id(resolved)
        return None

    def load_oc_layers(self: LayersDocumentProtocol) -> None:
        self.oc_layers = {}
        recover = bool(
            getattr(self, "xref_was_recovered", False)
            or getattr(self, "page_tree_was_recovered", False)
        )
        try:
            catalog = self.catalog()
        except ValueError:
            return
        oc = self.resolver.resolve(lookup_dict_key(catalog, "OCProperties"))
        if oc is None:
            return
        if not isinstance(oc, dict):
            if recover:
                return
            raise ValueError("invalid OCProperties dictionary")
        ocgs = self.resolver.resolve(lookup_dict_key(oc, "OCGs"))
        if ocgs is None:
            return
        if not isinstance(ocgs, list):
            if recover:
                return
            raise ValueError("invalid OCProperties OCGs array")

        on_layers: set[tuple[int, int] | int] = set()
        d = self.resolver.resolve(lookup_dict_key(oc, "D"))
        if d is not None:
            if not isinstance(d, dict):
                if recover:
                    d = None
                else:
                    raise ValueError("invalid OCProperties D dictionary")
        if d is not None:
            base_state_value = lookup_dict_key(d, "BaseState")
            base_state = (
                self.resolver.resolve_name(base_state_value)
                if base_state_value is not None
                else None
            )
            if base_state_value is not None and base_state is None:
                if recover:
                    base_state = None
                else:
                    raise ValueError("invalid OCProperties BaseState value")
            if base_state not in (None, "ON", "OFF", "Unchanged"):
                if recover:
                    base_state = None
                else:
                    raise ValueError("invalid OCProperties BaseState value")
            if base_state == "OFF":
                pass
            else:
                for ocg in ocgs:
                    ocg_resolved = self.resolver.resolve(ocg)
                    key = self.ocg_key(ocg, ocg_resolved)
                    if key is not None:
                        on_layers.add(key)
            on_refs = lookup_dict_key(d, "ON")
            if on_refs is not None and not isinstance(on_refs, list):
                on_refs = None
            for on_ref in on_refs or []:
                ocg_resolved = self.resolver.resolve(on_ref)
                if not isinstance(ocg_resolved, dict):
                    if recover:
                        continue
                    raise ValueError("invalid OCProperties ON entry")
                key = self.ocg_key(on_ref, ocg_resolved)
                if key is not None:
                    on_layers.add(key)
            off_refs = lookup_dict_key(d, "OFF")
            if off_refs is not None and not isinstance(off_refs, list):
                off_refs = None
            for off_ref in off_refs or []:
                ocg_resolved = self.resolver.resolve(off_ref)
                if not isinstance(ocg_resolved, dict):
                    if recover:
                        continue
                    raise ValueError("invalid OCProperties OFF entry")
                key = self.ocg_key(off_ref, ocg_resolved)
                if key is not None:
                    on_layers.discard(key)

        for ocg_ref in ocgs:
            ocg_resolved = self.resolver.resolve(ocg_ref)
            if not isinstance(ocg_resolved, dict):
                if recover:
                    continue
                raise ValueError("invalid OCProperties OCG entry")
            name = self.resolver.resolve_str(lookup_dict_key(ocg_resolved, "Name"))
            if not name:
                if recover:
                    continue
                raise ValueError("invalid OCProperties OCG name")
            key = self.ocg_key(ocg_ref, ocg_resolved)
            self.oc_layers[name] = key in on_layers if key is not None else False

    def oc_hidden_layers(self: LayersDocumentProtocol) -> frozenset[str]:
        if self.oc_layers is None:
            self.load_oc_layers()
        return frozenset(name for name, on in (self.oc_layers or {}).items() if not on)

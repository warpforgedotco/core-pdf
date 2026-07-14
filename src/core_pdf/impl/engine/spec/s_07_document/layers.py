# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_document.protocols import DocumentMixinProtocol


class LayersMixin:
    __slots__ = ()

    oc_layers: dict[str, bool] | None

    def load_oc_layers(self: DocumentMixinProtocol) -> None:
        self.oc_layers = {}
        oc_layers = self.oc_layers
        oc = self.resolver.resolve(self.catalog().get("OCProperties"))
        if oc is None:
            return
        if not isinstance(oc, dict):
            raise ValueError("invalid OCProperties dictionary")
        ocgs = oc.get("OCGs")
        if ocgs is None:
            return
        if not isinstance(ocgs, list):
            raise ValueError("invalid OCProperties OCGs array")

        on_layers: set[int] = set()
        d = oc.get("D")
        if d is not None:
            if not isinstance(d, dict):
                raise ValueError("invalid OCProperties D dictionary")
            base_state = d.get("BaseState")
            if base_state is not None and not isinstance(base_state, str):
                raise ValueError("invalid OCProperties BaseState value")
            if base_state not in (None, "ON", "OFF"):
                raise ValueError("invalid OCProperties BaseState value")
            if base_state == "OFF":
                pass
            else:
                for ocg in ocgs:
                    ocg_ref = self.resolver.resolve(ocg)
                    if isinstance(ocg_ref, dict):
                        on_layers.add(id(ocg_ref))
            on_refs = d.get("ON")
            if on_refs is not None and not isinstance(on_refs, list):
                raise ValueError("invalid OCProperties ON array")
            for on_ref in on_refs or []:
                ocg_ref = self.resolver.resolve(on_ref)
                if isinstance(ocg_ref, dict):
                    on_layers.add(id(ocg_ref))
            off_refs = d.get("OFF")
            if off_refs is not None and not isinstance(off_refs, list):
                raise ValueError("invalid OCProperties OFF array")
            for off_ref in off_refs or []:
                ocg_ref = self.resolver.resolve(off_ref)
                if isinstance(ocg_ref, dict):
                    on_layers.discard(id(ocg_ref))

        for ocg_ref in ocgs:
            ocg_ref = self.resolver.resolve(ocg_ref)
            if not isinstance(ocg_ref, dict):
                raise ValueError("invalid OCProperties OCG entry")
            name = self.resolver.resolve_str(ocg_ref.get("Name"))
            if not name:
                raise ValueError("invalid OCProperties OCG name")
            assert oc_layers is not None
            oc_layers[name] = id(ocg_ref) in on_layers

    def oc_hidden_layers(self: DocumentMixinProtocol) -> frozenset[str]:
        if self.oc_layers is None:
            self.load_oc_layers()
        return frozenset(name for name, on in (self.oc_layers or {}).items() if not on)

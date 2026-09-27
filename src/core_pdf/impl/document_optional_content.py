# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core_pdf.impl.document_contracts import DocumentState
from core_pdf.impl.types import PdfReference


class OptionalContent(DocumentState):
    __slots__ = ()

    @staticmethod
    def ocg_key(ref: object, resolved: object) -> tuple[int, int] | int | None:
        if isinstance(ref, PdfReference):
            return (ref.object_number, ref.generation_number)
        if isinstance(resolved, dict):
            return id(resolved)
        return None

    def oc_hidden_layers(self) -> frozenset[str]:
        return self.caches.get("hidden_layers", self.build_oc_hidden_layers)

    def build_oc_hidden_layers(self) -> frozenset[str]:
        malformed = self.recovery.malformed
        try:
            self.catalog()
        except ValueError:
            return frozenset()
        oc = self.catalog_dict("OCProperties", recoverable=True)
        if oc is None:
            return frozenset()
        ocgs = self.resolver.resolve(oc.get("OCGs"))
        if ocgs is None:
            return frozenset()
        if not isinstance(ocgs, list):
            malformed("invalid OCProperties OCGs array")
            return frozenset()

        on_layers: set[tuple[int, int] | int] = set()
        default_config = self.resolver.resolve(oc.get("D"))
        if default_config is not None and not isinstance(default_config, dict):
            malformed("invalid OCProperties D dictionary")
            default_config = None
        if default_config is not None:
            base_state_value = default_config.get("BaseState")
            base_state = (
                self.resolver.resolve_name(base_state_value)
                if base_state_value is not None
                else None
            )
            if (base_state_value is not None and base_state is None) or base_state not in (
                None,
                "ON",
                "OFF",
                "Unchanged",
            ):
                malformed("invalid OCProperties BaseState value")
                base_state = None
            if base_state != "OFF":
                for ocg in ocgs:
                    key = self.ocg_key(ocg, self.resolver.resolve(ocg))
                    if key is not None:
                        on_layers.add(key)

            for override_name, update in (("ON", on_layers.add), ("OFF", on_layers.discard)):
                refs = default_config.get(override_name)
                if not isinstance(refs, list):
                    continue
                for ref in refs:
                    ocg_resolved = self.resolver.as_dict(ref)
                    if ocg_resolved is None:
                        malformed(f"invalid OCProperties {override_name} entry")
                        continue
                    key = self.ocg_key(ref, ocg_resolved)
                    if key is not None:
                        update(key)

        hidden_layers: set[str] = set()
        for ocg_ref in ocgs:
            ocg_resolved = self.resolver.as_dict(ocg_ref)
            if ocg_resolved is None:
                malformed("invalid OCProperties OCG entry")
                continue
            name = self.resolver.resolve_str(ocg_resolved.get("Name"))
            if not name:
                malformed("invalid OCProperties OCG name")
                continue
            key = self.ocg_key(ocg_ref, ocg_resolved)
            if key is None or key not in on_layers:
                hidden_layers.add(name)
        return frozenset(hidden_layers)

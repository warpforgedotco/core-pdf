# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl.recovery_policy import Recovery
from core_pdf.impl.recovery_resolver import ObjectResolver
from core_pdf.impl.types import PdfReference
from core_pdf_spec.s_07_syntax.types import PdfDict


class OptionalContent:
    if not TYPE_CHECKING:
        __slots__ = ()

    resolver: ObjectResolver
    hidden_layers_cache: frozenset[str] | None

    if TYPE_CHECKING:

        def catalog(self) -> PdfDict: ...

        def catalog_dict(self, key: str, *, recoverable: bool = False) -> PdfDict | None: ...

        @property
        def recovery(self) -> Recovery: ...

    @staticmethod
    def ocg_key(ref: object, resolved: object) -> tuple[int, int] | int | None:
        if isinstance(ref, PdfReference):
            return (ref.object_number, ref.generation_number)
        if isinstance(resolved, dict):
            return id(resolved)
        return None

    def oc_hidden_layers(self) -> frozenset[str]:
        hidden = self.hidden_layers_cache
        if hidden is None:
            hidden = self.hidden_layers_cache = self.build_oc_hidden_layers()
        return hidden

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

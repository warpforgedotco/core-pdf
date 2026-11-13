# SPDX-License-Identifier: AGPL-3.0-only

"""Outlines, destinations and optional content.

The PdfDocument methods here are functions taking the document as self;
PdfDocument lists them as its methods.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
)

from core_pdf.impl.base.exceptions import (
    PdfParseError,
)
from core_pdf.impl.base.raw_records import (
    RawNamedDestination,
    RawOutlineItem,
)
from core_pdf.impl.document.pdf_page import goto_action_destination
from core_pdf.impl.spec.s_07_syntax.types import (
    PdfArray,
    PdfDict,
)
from core_pdf.impl.spec.standards import PdfVersion, SemanticContext
lazy from core_pdf.impl.recovery.trees import iter_name_tree_items

if TYPE_CHECKING:
    pass


from core_pdf.impl.document.page_tree import (
    PageLookup,
)

if TYPE_CHECKING:
    from core_pdf.impl.document.pdf_document import PageT, PdfDocument


def unresolved_destination(name: str) -> RawNamedDestination:
    return RawNamedDestination(page_index=None, type=None, args=[], raw=name)


def legacy_name_context(context: SemanticContext | None) -> bool:
    return context is not None and context.version in {PdfVersion(1, 0), PdfVersion(1, 1)}


def iter_outlines(self: PdfDocument[Any]) -> list[RawOutlineItem]:
    outlines = self.catalog_dict("Outlines")
    if outlines is None:
        return []
    first = self.resolver.resolve(outlines.get("First"))
    if first is None:
        return []
    return self.walk_outlines(first, 0)


def walk_outlines(
    self: PdfDocument[PageT],
    item: object,
    level: int,
    *,
    page_lookup: PageLookup[PageT] | None = None,
) -> list[RawOutlineItem]:
    if page_lookup is None:
        page_lookup = self.page_lookup
    recovery = self.recovery
    malformed = recovery.malformed
    if level > 200:
        raise ValueError("invalid outline depth")
    if not isinstance(item, dict):
        malformed("invalid outline item")
        return []
    result: list[RawOutlineItem] = []
    current: object | None = item
    seen: set[int] = set()
    while current is not None:
        current = self.resolver.as_dict(current)
        if current is None:
            malformed("invalid outline item")
            break
        marker = id(current)
        if marker in seen:
            malformed("outline cycle detected")
            break
        seen.add(marker)
        title = self.resolver.resolve_str(current.get("Title"))
        dest = current.get("Dest")
        if dest is None:
            dest = goto_action_destination(self.resolver, self.resolver.resolve(current.get("A")))
        try:
            result.append(
                RawOutlineItem(
                    title=title or "",
                    level=level,
                    dest=dest,
                    page_index=self.resolve_destination(dest, page_lookup=page_lookup),
                    count=self.extract_outline_count(current),
                )
            )
        except ValueError:
            if not recovery.enabled:
                raise
        first = current.get("First")
        if first is not None:
            first = self.resolver.as_dict(first)
            if first is None:
                malformed("invalid outline child")
                current = current.get("Next")
                continue
            result.extend(self.walk_outlines(first, level + 1, page_lookup=page_lookup))
        current = current.get("Next")
    return result


def extract_outline_count(self: PdfDocument[Any], current: PdfDict) -> int:
    raw_count = current.get("Count")
    if raw_count is None:
        return 0
    current_count = self.resolver.resolve_int(raw_count)
    if current_count is None:
        self.malformed("invalid outline count")
        return 0
    return current_count


def resolve_destination(
    self: PdfDocument[PageT], dest: object, *, page_lookup: PageLookup[PageT] | None = None
) -> int | None:
    if dest is None:
        return None
    normalized = self.normalize_destination_value(dest, page_lookup=page_lookup)
    if (
        normalized.raw is None
        and normalized.page_index is None
        and normalized.type is None
        and not normalized.args
    ):
        raise ValueError("invalid destination")
    return normalized.page_index


def resolve_named_destination(self: PdfDocument[Any], name: str) -> RawNamedDestination | None:
    return self.page_lookup.resolve_named_destination(name)


def destination_from_list(
    self: PdfDocument[PageT],
    resolved_list: PdfArray,
    *,
    page_lookup: PageLookup[PageT],
) -> RawNamedDestination:
    if not resolved_list:
        raise ValueError("invalid destination array")
    page_obj = self.resolver.resolve(resolved_list[0])
    if page_obj is None:
        raise ValueError("invalid destination page reference")
    page_index = page_lookup.page_index_for(page_obj)
    if page_index is None:
        raise ValueError("invalid destination page reference")
    dest_type = None
    args: PdfArray = []
    if len(resolved_list) >= 2:
        raw_type = resolved_list[1]
        dest_type = self.resolver.resolve_name_or_text(raw_type)
        if dest_type is None:
            raise ValueError("invalid destination type")
        args = list(resolved_list[2:])
    return RawNamedDestination(page_index=page_index, type=dest_type, args=args, raw=resolved_list)


def normalize_destination_value(
    self: PdfDocument[PageT],
    val: object,
    *,
    page_lookup: PageLookup[PageT] | None = None,
) -> RawNamedDestination:
    lookup = self.page_lookup if page_lookup is None else page_lookup
    return self.normalize_destination_entry(val, lookup.resolve_named_destination, lookup)


def normalize_destination_entry(
    self: PdfDocument[PageT],
    val: object,
    resolve_name: Callable[[str], RawNamedDestination | None],
    page_lookup: PageLookup[PageT],
) -> RawNamedDestination:
    seen: set[int] = set()
    resolved = self.resolver.resolve(val)
    while isinstance(resolved, dict):
        identity = id(resolved)
        if identity in seen:
            raise ValueError("cyclic destination dictionary")
        seen.add(identity)
        dest_value = resolved.get("D")
        if dest_value is None:
            break
        val = dest_value
        resolved = self.resolver.resolve(val)
    resolved_list = val if isinstance(val, list) else resolved
    if isinstance(resolved_list, tuple):
        resolved_list = list(resolved_list)
    if isinstance(resolved_list, list) and resolved_list:
        return self.destination_from_list(resolved_list, page_lookup=page_lookup)
    if isinstance(resolved_list, list):
        raise ValueError("invalid destination array")

    name = self.resolver.resolve_name_like_value(resolved)
    if name is not None:
        nested = resolve_name(name)
        if nested is not None:
            return nested
    raise ValueError("invalid destination")


def named_destinations(
    self: PdfDocument[PageT], *, page_lookup: PageLookup[PageT] | None = None
) -> dict[str, RawNamedDestination]:
    lookup = self.page_lookup if page_lookup is None else page_lookup
    targets: dict[str, object] = {}
    dests = self.resolver.dict_at(self.catalog(), "Dests")
    if dests is not None:
        for name, val in dests.items():
            resolved_name = self.resolver.resolve_name(name)
            if resolved_name is None:
                raise ValueError("invalid named destination key")
            targets[resolved_name] = self.resolver.resolve(val)
    names = self.resolver.dict_at(self.catalog(), "Names")
    if names is not None:
        dests_tree = self.resolver.dict_at(names, "Dests")
        if dests_tree is not None:
            targets.update(
                iter_name_tree_items(
                    dests_tree,
                    self.resolver.resolve,
                    self.resolver.resolve_str,
                    on_malformed=self.recovery.malformed,
                )
            )

    normalized: dict[str, RawNamedDestination] = {}
    resolving: set[str] = set()

    def normalize_name(name: str) -> RawNamedDestination:
        cached = normalized.get(name)
        if cached is not None:
            return cached
        if name in resolving:
            return unresolved_destination(name)
        resolving.add(name)
        try:
            target = targets.get(name)
            result = (
                unresolved_destination(name)
                if target is None
                else self.normalize_destination_entry(target, normalize_name, lookup)
            )
            normalized[name] = result
            return result
        finally:
            resolving.discard(name)

    for name in targets:
        try:
            normalize_name(name)
        except PdfParseError, ValueError:
            normalized[name] = unresolved_destination(name)
    return normalized


def oc_hidden_layers(self: PdfDocument[Any]) -> frozenset[str]:
    return self.caches.get("hidden_layers", self.build_oc_hidden_layers)


def build_oc_hidden_layers(self: PdfDocument[Any]) -> frozenset[str]:
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
            self.resolver.resolve_name(base_state_value) if base_state_value is not None else None
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

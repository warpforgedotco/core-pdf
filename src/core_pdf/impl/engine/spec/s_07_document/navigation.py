# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING, cast

from core_pdf.impl.engine.spec.s_07_document.name_trees import iter_name_tree_items
from core_pdf.impl.engine.spec.s_07_document.protocols import NavigationResolver
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.models import NamedDestination, OutlineItem
from core_pdf.impl.types import PdfArray, PdfDict, PdfObject


class NavigationMixin:
    __slots__ = ()

    named_destinations_cache: dict[str, NamedDestination] | None
    resolver: NavigationResolver
    xref_was_recovered: bool
    page_tree_was_recovered: bool

    if TYPE_CHECKING:

        def catalog(self) -> PdfDict: ...

        def page_index_for(self, page_obj: object) -> int | None: ...

    def iter_outlines(self) -> list[OutlineItem]:
        outlines = self.resolver.resolve(lookup_dict_key(self.catalog(), "Outlines"))
        if outlines is None:
            return []
        if not isinstance(outlines, dict):
            raise ValueError("invalid Outlines dictionary")
        first = self.resolver.resolve(lookup_dict_key(outlines, "First"))
        if first is None:
            return []
        return self.walk_outlines(first, 0)

    def walk_outlines(self, item: object, level: int) -> list[OutlineItem]:
        recover_outlines = self.xref_was_recovered or self.page_tree_was_recovered
        if level > 200:
            raise ValueError("invalid outline depth")
        if not isinstance(item, dict):
            if recover_outlines:
                return []
            raise ValueError("invalid outline item")
        result: list[OutlineItem] = []
        current: object | None = item
        seen: set[int] = set()
        while current is not None:
            current = self.resolver.resolve(current)
            if not isinstance(current, dict):
                if recover_outlines:
                    break
                raise ValueError("invalid outline item")
            marker = id(current)
            if marker in seen:
                if recover_outlines:
                    break
                raise ValueError("outline cycle detected")
            seen.add(marker)
            title = self.resolver.resolve_str(lookup_dict_key(current, "Title"))
            dest = lookup_dict_key(current, "Dest")
            if dest is None:
                action = lookup_dict_key(current, "A")
                if (
                    isinstance(action, dict)
                    and self.resolver.resolve_name(lookup_dict_key(action, "S")) == "GoTo"
                ):
                    dest = lookup_dict_key(action, "D")
            try:
                result.append(
                    OutlineItem(
                        title=title or "",
                        level=level,
                        dest=cast(PdfObject | str | None, dest),
                        page_index=self.resolve_destination(dest),
                        count=self.extract_outline_count(cast(PdfDict, current)),
                    )
                )
            except ValueError:
                if not recover_outlines:
                    raise
            first = lookup_dict_key(current, "First")
            if first is not None:
                first = self.resolver.resolve(first)
                if not isinstance(first, dict):
                    if recover_outlines:
                        current = lookup_dict_key(current, "Next")
                        continue
                    raise ValueError("invalid outline child")
                result.extend(self.walk_outlines(first, level + 1))
            current = lookup_dict_key(current, "Next")
        return result

    @staticmethod
    def validate_outline_count(value: object) -> int:
        if type(value) is not int:
            raise ValueError("invalid outline count")
        return value

    def extract_outline_count(self, current: PdfDict) -> int:
        raw_count = lookup_dict_key(current, "Count")
        if raw_count is None:
            return 0
        current_count = self.resolver.resolve_int(raw_count)
        if current_count is None:
            if self.xref_was_recovered or self.page_tree_was_recovered:
                return 0
            raise ValueError("invalid outline count")
        return self.validate_outline_count(current_count)

    def resolve_destination(self, dest: object, seen: set[str] | None = None) -> int | None:
        if dest is None:
            return None
        normalized = self.normalize_destination_value(dest, seen)
        if (
            normalized.raw is None
            and normalized.page_index is None
            and normalized.type is None
            and not normalized.args
        ):
            raise ValueError("invalid destination")
        return normalized.page_index

    def named_destinations(
        self,
    ) -> dict[str, NamedDestination]:
        if self.named_destinations_cache is None:
            self.populate_named_destinations()
        return dict(self.named_destinations_cache or {})

    def resolve_named_destination(
        self, name: str, seen: set[str] | None = None
    ) -> NamedDestination | None:
        if seen is None:
            seen = set()
        if name in seen:
            return None
        seen.add(name)
        if self.named_destinations_cache is None:
            self.populate_named_destinations()
        return (self.named_destinations_cache or {}).get(name)

    def resolve_named_destination_value(
        self, val: object, seen: set[str] | None = None
    ) -> NamedDestination:
        return self.normalize_destination_value(val, seen)

    def destination_from_list(self, resolved_list: PdfArray) -> NamedDestination:
        if not resolved_list:
            raise ValueError("invalid destination array")
        page_obj = self.resolver.resolve(resolved_list[0])
        if page_obj is None:
            raise ValueError("invalid destination page reference")
        page_index = self.page_index_for(page_obj)
        if page_index is None:
            raise ValueError("invalid destination page reference")
        dest_type = None
        args: PdfArray = []
        if len(resolved_list) >= 2:
            raw_type = resolved_list[1]
            dest_type = self.resolver.resolve_name(raw_type) or self.resolver.resolve_str(raw_type)
            if dest_type is None:
                raise ValueError("invalid destination type")
            args = list(resolved_list[2:]) if len(resolved_list) > 2 else []
        return NamedDestination(page_index=page_index, type=dest_type, args=args, raw=resolved_list)

    def normalize_destination_value(
        self,
        val: object,
        seen: set[str] | None = None,
        targets: dict[str, object] | None = None,
        normalized: dict[str, NamedDestination] | None = None,
        resolving: set[str] | None = None,
    ) -> NamedDestination:
        if seen is None:
            seen = set()
        resolved = self.resolver.resolve(val)
        if isinstance(resolved, dict):
            dest_value = lookup_dict_key(resolved, "D")
            if dest_value is not None:
                return self.normalize_destination_value(
                    dest_value,
                    seen,
                    targets=targets,
                    normalized=normalized,
                    resolving=resolving,
                )
        resolved_list = val if isinstance(val, list) else resolved
        if isinstance(resolved_list, tuple):
            resolved_list = list(resolved_list)
        if isinstance(resolved_list, list) and resolved_list:
            return self.destination_from_list(cast(PdfArray, resolved_list))
        if isinstance(resolved_list, list):
            raise ValueError("invalid destination array")

        name = self.resolver.resolve_name_like_value(resolved)
        if name is not None:
            if targets is not None and normalized is not None and resolving is not None:
                cached = normalized.get(name)
                if cached is not None:
                    return cached
                if name in resolving:
                    return NamedDestination(page_index=None, type=None, args=[], raw=name)
                resolving.add(name)
                try:
                    target = targets.get(name)
                    result = (
                        NamedDestination(page_index=None, type=None, args=[], raw=name)
                        if target is None
                        else self.normalize_destination_value(
                            target,
                            seen,
                            targets=targets,
                            normalized=normalized,
                            resolving=resolving,
                        )
                    )
                    normalized[name] = result
                    return result
                finally:
                    resolving.discard(name)

            if name in seen:
                return NamedDestination(page_index=None, type=None, args=[], raw=name)
            nested = self.resolve_named_destination(name, seen)
            if nested is not None:
                return nested
        raise ValueError("invalid destination")

    def populate_named_destinations(self) -> None:
        if self.named_destinations_cache is not None:
            return
        targets: dict[str, object] = {}
        dests = self.resolver.resolve(lookup_dict_key(self.catalog(), "Dests"))
        if isinstance(dests, dict):
            for name, val in dests.items():
                resolved_name = self.resolver.resolve_name(name)
                if resolved_name is None:
                    raise ValueError("invalid named destination key")
                targets[resolved_name] = self.resolver.resolve(val)
        names = self.resolver.resolve(lookup_dict_key(self.catalog(), "Names"))
        if isinstance(names, dict):
            dests_tree = self.resolver.resolve(lookup_dict_key(names, "Dests"))
            if isinstance(dests_tree, dict):
                for name, value in iter_name_tree_items(
                    dests_tree,
                    self.resolver.resolve,
                    self.resolver.resolve_str,
                    recover=self.xref_was_recovered or self.page_tree_was_recovered,
                ):
                    targets[name] = value

        normalized: dict[str, NamedDestination] = {}
        resolving: set[str] = set()

        for name in targets:
            self.normalize_destination_value(
                name,
                targets=targets,
                normalized=normalized,
                resolving=resolving,
            )

        object.__setattr__(self, "named_destinations_cache", normalized)

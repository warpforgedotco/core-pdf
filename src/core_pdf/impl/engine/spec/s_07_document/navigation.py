from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_document.models import NamedDestination, OutlineItem
from core_pdf.impl.engine.spec.s_07_document.protocols import DocumentMixinProtocol
from core_pdf.impl.engine.spec.s_07_objects.resolver import is_pdf_object
from core_pdf.impl.engine.spec.s_07_syntax.primitives import PdfDictLike, PdfObject, PdfReference


class NavigationMixin:
    __slots__ = ()

    named_destinations_cache: dict[str, NamedDestination] | None

    def iter_outlines(self: DocumentMixinProtocol) -> list[OutlineItem]:
        outlines = self.resolver.resolve(self.catalog().get("Outlines"))
        if outlines is None:
            return []
        if not isinstance(outlines, dict):
            raise ValueError("invalid Outlines dictionary")
        first = self.resolver.resolve(outlines.get("First"))
        if first is None:
            return []
        if not isinstance(first, dict):
            raise ValueError("invalid outline item")
        return self.walk_outlines(first, 0)

    def walk_outlines(
        self: DocumentMixinProtocol, item: PdfDictLike, level: int
    ) -> list[OutlineItem]:
        if level > 200:
            raise ValueError("invalid outline depth")
        if not isinstance(item, dict):
            raise ValueError("invalid outline item")
        result: list[OutlineItem] = []
        current: PdfDictLike | None = item
        seen: set[int] = set()
        while current is not None:
            marker = id(current)
            if marker in seen:
                raise ValueError("outline cycle detected")
            seen.add(marker)
            title = self.resolver.resolve_str(current.get("Title"))
            dest = current.get("Dest")
            if dest is None:
                action = current.get("A")
                if (
                    isinstance(action, dict)
                    and self.resolver.resolve_str(action.get("S")) == "GoTo"
                ):
                    dest = action.get("D")
            outline_dest = dest if isinstance(dest, (PdfReference, list, str)) else None
            result.append(
                OutlineItem(
                    title=title or "",
                    level=level,
                    dest=outline_dest,
                    page_index=self.resolve_destination(dest),
                    count=self.resolver.resolve_int(current.get("Count")) or 0,
                )
            )
            first = current.get("First")
            if first is not None:
                resolved_first = self.resolver.resolve_dict(first)
                if resolved_first is None:
                    raise ValueError("invalid outline child")
                result.extend(self.walk_outlines(resolved_first, level + 1))
            current = self.resolver.resolve_dict(current.get("Next"))
        return result

    def resolve_destination(
        self: DocumentMixinProtocol, dest: PdfObject, seen: set[str] | None = None
    ) -> int | None:
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

    def named_destinations(self: DocumentMixinProtocol) -> dict[str, NamedDestination]:
        if self.named_destinations_cache is None:
            self.populate_named_destinations()
        return dict(self.named_destinations_cache or {})

    def resolve_named_destination(
        self: DocumentMixinProtocol, name: str, seen: set[str] | None = None
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
        self: DocumentMixinProtocol, val: PdfObject, seen: set[str] | None = None
    ) -> NamedDestination:
        return self.normalize_destination_value(val, seen)

    def destination_from_list(
        self: DocumentMixinProtocol, resolved_list: list[PdfObject]
    ) -> NamedDestination:
        if not resolved_list:
            raise ValueError("invalid destination array")
        page_obj = self.resolver.resolve(resolved_list[0])
        if page_obj is None:
            raise ValueError("invalid destination page reference")
        page_index = self.page_index_for(page_obj)
        if page_index is None:
            raise ValueError("invalid destination page reference")
        dest_type = None
        args: list[PdfObject] = []
        if len(resolved_list) >= 2:
            raw_type = resolved_list[1]
            dest_type = self.resolver.resolve_name(raw_type) or self.resolver.resolve_str(raw_type)
            if dest_type is None:
                raise ValueError("invalid destination type")
            args = list(resolved_list[2:]) if len(resolved_list) > 2 else []
        return NamedDestination(page_index=page_index, type=dest_type, args=args, raw=resolved_list)

    def normalize_destination_value(
        self: DocumentMixinProtocol,
        val: PdfObject,
        seen: set[str] | None = None,
        targets: dict[str, PdfObject] | None = None,
        normalized: dict[str, NamedDestination] | None = None,
        resolving: set[str] | None = None,
    ) -> NamedDestination:
        if seen is None:
            seen = set()
        resolved_list = val if isinstance(val, list) else self.resolver.resolve(val)
        if isinstance(resolved_list, tuple):
            resolved_list = list(resolved_list)
        if isinstance(resolved_list, list) and resolved_list:
            return self.destination_from_list(resolved_list)
        if isinstance(resolved_list, list):
            raise ValueError("invalid destination array")

        resolved = self.resolver.resolve(val)
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

    def populate_named_destinations(self: DocumentMixinProtocol) -> None:
        if self.named_destinations_cache is not None:
            return
        targets: dict[str, PdfObject] = {}
        dests = self.resolver.resolve(self.catalog().get("Dests"))
        if isinstance(dests, dict):
            for name, val in dests.items():
                if not is_pdf_object(name):
                    continue
                resolved_name = self.resolver.resolve_str(name)
                if resolved_name is not None:
                    targets[resolved_name] = self.resolver.resolve(val)
        names = self.resolver.resolve(self.catalog().get("Names"))
        if isinstance(names, dict):
            dests_tree = self.resolver.resolve(names.get("Dests"))
            if isinstance(dests_tree, dict):
                self.walk_name_tree(dests_tree, targets)

        normalized: dict[str, NamedDestination] = {}
        resolving: set[str] = set()

        for name in targets:
            self.normalize_destination_value(
                name,
                targets=targets,
                normalized=normalized,
                resolving=resolving,
            )

        self.named_destinations_cache = normalized

    def walk_name_tree(
        self: DocumentMixinProtocol,
        node: PdfObject,
        results: dict[str, PdfObject],
        _depth: int = 0,
    ) -> None:
        if _depth > 100:
            raise ValueError("invalid name tree depth")
        node = self.resolver.resolve(node)
        if node is None:
            return
        if not isinstance(node, dict):
            raise ValueError("invalid name tree node")
        names = self.resolver.resolve(node.get("Names"))
        if names is None:
            pass
        elif not isinstance(names, list):
            raise ValueError("invalid name tree Names array")
        else:
            if len(names) % 2 != 0:
                raise ValueError("invalid name tree Names array")
            for i in range(0, len(names), 2):
                name_val = self.resolver.resolve_str(names[i])
                if name_val:
                    dest_val = names[i + 1]
                    results[name_val] = self.resolver.resolve(dest_val)
        kids = self.resolver.resolve(node.get("Kids"))
        if kids is None:
            return
        if not isinstance(kids, list):
            raise ValueError("invalid name tree Kids array")
        for kid in kids:
            self.walk_name_tree(kid, results, _depth + 1)

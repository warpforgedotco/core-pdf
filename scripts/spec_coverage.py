# SPDX-License-Identifier: AGPL-3.0-only

"""Write docs/spec-coverage.md: what each PDF version defines that core never reads.

The Arlington PDF Model (PDF Association, Apache-2.0) lists every object PDF defines, each
key with the version that introduced it, and the names a name-valued key can take. This
compares it with the keys core's source reads and the values it compares, found in the
AST of src/core_pdf (see KeyReads), and lists, by the version or extension that added
them, the keys core never reads and, for keys it reads, the values it never compares.

Reads are matched by key name across all of core, not attributed to the object read: no
reliable link from a .get("S") to the Arlington object it reads exists without types.
A key core never reads is never looked up by name; one it reads somewhere may still be
read only in some of the objects that have it, so this finds gaps, not proof of support.

uv run --all-packages python scripts/spec_coverage.py [path to an Arlington checkout]

Without a path it downloads the pinned commit below into .upstream-results/.
"""

from __future__ import annotations

import ast
import csv
import io
import re
import sys
import tarfile
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src" / "core_pdf"
REPORT = ROOT / "docs" / "spec-coverage.md"
ARLINGTON_COMMIT = "603c174b024d1d2bb15a33686af83800d5c41269"
ARLINGTON_URL = (
    "https://codeload.github.com/pdf-association/arlington-pdf-model/tar.gz/" + ARLINGTON_COMMIT
)
VERSIONS = ("2.0", "1.7", "1.6", "1.5", "1.4", "1.3", "1.2", "1.1", "1.0")
VERSION = re.compile(r"^[12]\.[0-9]$")
# fn:Extension(NAME,1.7), alone or in fn:Eval(fn:Extension(NAME,1.7) || 2.0).
EXTENSION = re.compile(r"fn:Extension\(([A-Za-z0-9_]+)")
LATER_VERSION = re.compile(r"\|\|\s*([12]\.[0-9])\)")
SINCE_VALUE = re.compile(r"fn:SinceVersion\(([12]\.[0-9]),\s*([^,()\]]+)\)")
SHOWN_OBJECTS = 6


def arlington_tables(argv: list[str]) -> Path:
    if argv:
        return Path(argv[0]) / "tsv" / "latest"
    target = ROOT / ".upstream-results" / f"arlington-{ARLINGTON_COMMIT}"
    tables = target / "tsv" / "latest"
    if not tables.is_dir():
        with urllib.request.urlopen(ARLINGTON_URL) as response:  # noqa: S310
            data = response.read()
        prefix = f"arlington-pdf-model-{ARLINGTON_COMMIT}/tsv/latest/"
        tables.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                if not (member.isfile() and member.name.startswith(prefix)):
                    continue
                if member.name.endswith(".tsv"):
                    extracted = archive.extractfile(member)
                    assert extracted is not None
                    (tables / Path(member.name).name).write_bytes(extracted.read())
    return tables


class Scope:
    """A function, class body or module, and the names it binds."""

    def __init__(self, node: ast.AST, parent: Scope | None) -> None:
        self.node = node
        self.parent = parent
        self.params: dict[str, ast.expr | None] = {}
        self.assigns: dict[str, list[ast.expr]] = defaultdict(list)
        # name -> (iterable, position in the loop target, () for the whole target)
        self.loops: dict[str, list[tuple[ast.expr, tuple[int, ...]]]] = defaultdict(list)
        self.owned: list[ast.AST] = []
        self.positional: list[str] = []
        self.offset = 0  # 1 for a method called through an attribute: self is not passed
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            positional = [*arguments.posonlyargs, *arguments.args]
            self.positional = [arg.arg for arg in positional]
            defaults: list[ast.expr | None] = [None] * (
                len(positional) - len(arguments.defaults)
            ) + list(arguments.defaults)
            for arg, default in zip(positional, defaults, strict=True):
                self.params[arg.arg] = default
            for arg, kw_default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
                self.params[arg.arg] = kw_default
            static = any(
                isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
                for decorator in node.decorator_list
            )
            if (
                isinstance(parent and parent.node, ast.ClassDef)
                and positional
                and positional[0].arg in ("self", "cls")
                and not static
            ):
                self.offset = 1


Atom: TypeAlias = tuple[int, str]  # (id of a function node, parameter name)
Bound: TypeAlias = tuple[list[tuple[ast.expr, Scope]], set[Atom]]
TABLE_BUILDERS = frozenset({"dict", "frozenset", "set", "tuple", "list", "MappingProxyType"})
NAME_WRAPPERS = frozenset({"PdfName", "of", "encode", "decode", "removeprefix"})
# A call x.append(...) on anything but self is taken for the builtin method, not one of
# core's functions of that name.
BUILTIN_METHODS = frozenset(
    name for kind in (dict, list, set, str, bytes, bytearray) for name in dir(kind)
)


class KeyReads:
    """The dictionary keys core reads, and the name values it compares, found in its AST.

    A literal counts as a key read where it is used as a dictionary key: x.get("K"),
    x["K"], "K" in x, a case {"K": ...} pattern, a comparison with the keys of a
    dictionary being iterated, or an argument that a function so uses (found by following
    parameters to a fixed point, so lookup_page_resource("Font", name) and inheritance
    helpers count). A literal counts as a value where it is compared (==, in, case), is a
    key of a table looked up by a value (BLEND_OPS[mode]), is an entry of a module- or
    class-level table (FILTER_DESCRIPTORS), or is passed to a parameter so used. A lookup
    in a constant table dispatches on a value, so is not a key read; a table of handlers
    (op_G) is keyed by operators, so iterating it yields no keys.
    """

    def __init__(self, source: Path) -> None:
        self.scopes: list[Scope] = []
        self.functions: dict[str, list[Scope]] = defaultdict(list)
        self.constants: dict[str, list[tuple[ast.expr, Scope]]] = defaultdict(list)
        self.keys: set[str] = set()
        self.values: set[str] = set()
        for path in sorted(source.rglob("*.py")):
            module = ast.parse(path.read_text(), str(path))
            self.collect(module, None)
        self.solve()

    # Collecting scopes.

    def collect(self, node: ast.AST, parent: Scope | None) -> None:
        scope = Scope(node, parent)
        self.scopes.append(scope)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.functions[node.name].append(scope)
        body: list[ast.AST] = []
        if isinstance(node, ast.Lambda):
            body = [node.body]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [*node.body, *node.args.defaults]
            body += [default for default in node.args.kw_defaults if default is not None]
        elif isinstance(node, (ast.Module, ast.ClassDef)):
            body = list(node.body)
        stack = list(reversed(body))
        while stack:
            child = stack.pop()
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.collect(child, scope)
                stack.extend(child.decorator_list)
                continue
            if isinstance(child, ast.Lambda):
                self.collect(child, scope)
                continue
            scope.owned.append(child)
            self.bind(scope, child)
            stack.extend(reversed(list(ast.iter_child_nodes(child))))

    def bind(self, scope: Scope, node: ast.AST) -> None:
        targets: Sequence[ast.expr] = ()
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)) and node.value:
            targets, value = [node.target], node.value
        for target in targets:
            if isinstance(target, ast.Name) and value is not None:
                scope.assigns[target.id].append(value)
                if isinstance(scope.node, (ast.Module, ast.ClassDef)):
                    self.constants[target.id].append((value, scope))
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            self.bind_loop(scope, node.target, node.iter, ())

    def bind_loop(
        self, scope: Scope, target: ast.expr, iterable: ast.expr, position: tuple[int, ...]
    ) -> None:
        if isinstance(target, ast.Name):
            scope.loops[target.id].append((iterable, position))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for index, element in enumerate(target.elts):
                self.bind_loop(scope, element, iterable, (*position, index))

    # Resolving what an expression may hold.

    def lookup(self, name: str, scope: Scope) -> Bound:
        """The expressions a name may hold, and the parameters it may be."""
        current: Scope | None = scope
        while current is not None:
            exprs: list[tuple[ast.expr, Scope]] = []
            params: set[Atom] = set()
            if name in current.params:
                params.add((id(current.node), name))
                default = current.params[name]
                if default is not None:
                    exprs.append((default, current))
            exprs += [(value, current) for value in current.assigns.get(name, ())]
            for iterable, position in current.loops.get(name, ()):
                exprs += self.loop_values(iterable, position, current)
            if name in current.params or name in current.assigns or name in current.loops:
                return exprs, params
            current = current.parent
            # Class bodies do not enclose their methods' names.
            while current is not None and isinstance(current.node, ast.ClassDef):
                current = current.parent
        return list(self.constants.get(name, ())), set()

    def loop_values(
        self, iterable: ast.expr, position: tuple[int, ...], scope: Scope
    ) -> list[tuple[ast.expr, Scope]]:
        if isinstance(iterable, ast.Call) and position:
            func = iterable.func
            if isinstance(func, ast.Attribute) and func.attr == "items" and len(position) == 1:
                pairs = self.dict_pairs(func.value, scope, set())
                return [pair[position[0]] for pair in pairs]
            if isinstance(func, ast.Name) and func.id == "enumerate" and iterable.args:
                if position[0] == 1:
                    return self.loop_values(iterable.args[0], position[1:], scope)
                return []
            if isinstance(func, ast.Name) and func.id == "zip":
                if position[0] < len(iterable.args):
                    return self.loop_values(iterable.args[position[0]], position[1:], scope)
                return []
        rows = self.members(iterable, scope, set())[0]
        for index in position:
            rows = [
                (row.elts[index], row_scope)
                for row, row_scope in rows
                if isinstance(row, (ast.Tuple, ast.List)) and index < len(row.elts)
            ]
        return rows

    def dict_pairs(
        self, expr: ast.expr, scope: Scope, seen: set[int]
    ) -> list[tuple[tuple[ast.expr, Scope], tuple[ast.expr, Scope]]]:
        if id(expr) in seen:
            return []
        seen.add(id(expr))
        if isinstance(expr, ast.Dict):
            # A table of handlers (op_G, lambda) is keyed by operators, not dictionary keys.
            return [
                ((key, scope), (value, scope))
                for key, value in zip(expr.keys, expr.values, strict=True)
                if key is not None and not isinstance(value, (ast.Attribute, ast.Lambda))
            ]
        pairs = []
        for value, value_scope in self.expressions(expr, scope):
            if value is not expr:
                pairs += self.dict_pairs(value, value_scope, seen)
        return pairs

    def expressions(self, expr: ast.expr, scope: Scope) -> list[tuple[ast.expr, Scope]]:
        """What a name or constant attribute refers to; the expression itself otherwise."""
        if isinstance(expr, ast.Name):
            return self.lookup(expr.id, scope)[0]
        if isinstance(expr, ast.Attribute) and expr.attr.isupper():
            return list(self.constants.get(expr.attr, ()))
        return [(expr, scope)]

    def scalar(self, expr: ast.expr, scope: Scope, seen: set[int]) -> tuple[set[str], set[Atom]]:
        """The literals an expression may be, and the parameters it may be."""
        if id(expr) in seen:
            return set(), set()
        seen.add(id(expr))
        names: set[str] = set()
        params: set[Atom] = set()
        if isinstance(expr, ast.Constant) and isinstance(expr.value, (str, bytes)):
            value = expr.value if isinstance(expr.value, str) else expr.value.decode("latin-1")
            names.add(value.removeprefix("/"))
        elif isinstance(expr, ast.Call) and isinstance(expr.func, (ast.Name, ast.Attribute)):
            func = expr.func
            name = func.id if isinstance(func, ast.Name) else func.attr
            if name in NAME_WRAPPERS:
                inner = expr.args[0] if expr.args and name in ("PdfName", "of") else None
                if inner is None and isinstance(func, ast.Attribute):
                    inner = func.value
                if inner is not None:
                    return self.scalar(inner, scope, seen)
        elif isinstance(expr, ast.Name):
            exprs, params = self.lookup(expr.id, scope)
            for value, value_scope in exprs:
                more_names, more_params = self.scalar(value, value_scope, seen)
                names |= more_names
                params |= more_params
        elif isinstance(expr, ast.Attribute) and expr.attr.isupper():
            for value, value_scope in self.constants.get(expr.attr, ()):
                names |= self.scalar(value, value_scope, seen)[0]
        elif isinstance(expr, (ast.IfExp, ast.BoolOp)):
            parts = [expr.body, expr.orelse] if isinstance(expr, ast.IfExp) else expr.values
            for part in parts:
                more_names, more_params = self.scalar(part, scope, seen)
                names |= more_names
                params |= more_params
        return names, params

    def members(self, expr: ast.expr, scope: Scope, seen: set[int]) -> Bound:
        """The element expressions of a collection, and the parameters it may be."""
        if id(expr) in seen:
            return [], set()
        seen.add(id(expr))
        found: list[tuple[ast.expr, Scope]] = []
        params: set[Atom] = set()
        parts: list[ast.expr] = []
        if isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
            for element in expr.elts:
                if isinstance(element, ast.Starred):
                    parts.append(element.value)
                else:
                    found.append((element, scope))
        elif isinstance(expr, ast.Dict):
            found += [(key, scope) for (key, _), _ in self.dict_pairs(expr, scope, set())]
        elif isinstance(expr, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            found.append((expr.elt, scope))
        elif isinstance(expr, ast.DictComp):
            found.append((expr.key, scope))
        elif isinstance(expr, ast.Call):
            func = expr.func
            if isinstance(func, ast.Name) and func.id in (*TABLE_BUILDERS, "sorted", "reversed"):
                parts += expr.args[:1]
            elif isinstance(func, ast.Attribute) and func.attr == "keys":
                parts.append(func.value)
        elif isinstance(expr, ast.BinOp) and isinstance(
            expr.op, (ast.BitOr, ast.Add, ast.Sub, ast.BitAnd)
        ):
            parts.append(expr.left)
            if not isinstance(expr.op, (ast.Sub, ast.BitAnd)):
                parts.append(expr.right)
        elif isinstance(expr, ast.IfExp):
            parts += [expr.body, expr.orelse]
        elif isinstance(expr, ast.Name):
            exprs, params = self.lookup(expr.id, scope)
            for value, value_scope in exprs:
                more, more_params = self.members(value, value_scope, seen)
                found += more
                params |= more_params
        elif isinstance(expr, ast.Attribute) and expr.attr.isupper():
            for value, value_scope in self.constants.get(expr.attr, ()):
                found += self.members(value, value_scope, seen)[0]
        for part in parts:
            more, more_params = self.members(part, scope, seen)
            found += more
            params |= more_params
        return found, params

    def is_table(self, expr: ast.expr, scope: Scope) -> bool:
        """Whether a lookup's base is a constant table, so the lookup dispatches on a value."""
        if isinstance(expr, (ast.Dict, ast.DictComp, ast.Set, ast.Tuple, ast.List)):
            return True
        if isinstance(expr, ast.Attribute):
            return expr.attr.isupper() and expr.attr in self.constants
        if not isinstance(expr, ast.Name):
            return False
        exprs, params = self.lookup(expr.id, scope)
        if params or not exprs:
            return False
        # A local sequence of literals is a table, as it is inline; one filled later
        # (keys = [], then appended to) may hold dictionary keys, so is not.
        return all(
            not isinstance(value_scope.node, (ast.FunctionDef, ast.AsyncFunctionDef))
            or isinstance(value, (ast.Dict, ast.DictComp, ast.Set, ast.SetComp))
            or (
                isinstance(value, (ast.Tuple, ast.List))
                and bool(value.elts)
                and all(isinstance(element, ast.Constant) for element in value.elts)
            )
            for value, value_scope in exprs
        )

    def dictionary_keys(self, expr: ast.expr, scope: Scope) -> bool:
        """Whether a name is bound to the keys of a dictionary being iterated."""
        if not isinstance(expr, ast.Name):
            return False
        current: Scope | None = scope
        while current is not None:
            for iterable, position in current.loops.get(expr.id, ()):
                if not (
                    isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Attribute)
                ):
                    continue
                method, base = iterable.func.attr, iterable.func.value
                if self.is_table(base, current):
                    continue
                if (method == "items" and position == (0,)) or (
                    method == "keys" and position == ()
                ):
                    return True
            if expr.id in current.loops or expr.id in current.params:
                return False
            current = current.parent
        return False

    # Finding the reads.

    def sinks(self) -> tuple[list[tuple[str, ast.expr, Scope]], list[tuple[ast.Call, Scope]]]:
        """Each expression used as a key or compared as a value, and every call."""
        sinks: list[tuple[str, ast.expr, Scope]] = []
        calls: list[tuple[ast.Call, Scope]] = []
        for scope in self.scopes:
            for node in scope.owned:
                sinks += self.node_sinks(node, scope)
                if isinstance(node, ast.Call):
                    calls.append((node, scope))
            if not isinstance(scope.node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                # Entries of module- and class-level tables: values core dispatches on.
                for name, values in scope.assigns.items():
                    if name == "__all__":
                        continue
                    for value in values:
                        if isinstance(value, (ast.Tuple, ast.List, ast.Set, ast.Dict)) or (
                            isinstance(value, ast.Call)
                            and isinstance(value.func, ast.Name)
                            and value.func.id in TABLE_BUILDERS
                        ):
                            for constant in ast.walk(value):
                                if isinstance(constant, ast.Constant):
                                    sinks.append(("value", constant, scope))
        return sinks, calls

    def node_sinks(self, node: ast.AST, scope: Scope) -> list[tuple[str, ast.expr, Scope]]:
        sinks: list[tuple[str, ast.expr, Scope]] = []
        lookup: tuple[ast.expr, ast.expr] | None = None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("get", "pop", "setdefault") and node.args:
                lookup = node.func.value, node.args[0]
        elif (
            isinstance(node, ast.Subscript)
            and not isinstance(node.ctx, ast.Store)
            and not isinstance(node.slice, ast.Slice)
        ):
            lookup = node.value, node.slice
        if lookup is not None:
            base, key = lookup
            if self.is_table(base, scope):
                # A dispatch on a value: the table's keys are the values handled.
                sinks += [("value", key, scope), ("values", base, scope)]
            else:
                sinks.append(("key", key, scope))
        elif isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for op, left, right in zip(node.ops, operands, operands[1:], strict=False):
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    sinks += [("value", left, scope), ("value", right, scope)]
                    for one, other in ((left, right), (right, left)):
                        if self.dictionary_keys(one, scope):
                            sinks.append(("key", other, scope))
                elif isinstance(op, (ast.In, ast.NotIn)):
                    if self.is_table(right, scope):
                        sinks += [("values", right, scope), ("value", left, scope)]
                        if self.dictionary_keys(left, scope):
                            sinks.append(("keys", right, scope))
                    else:
                        sinks += [("key", left, scope), ("values", right, scope)]
        elif isinstance(node, ast.Match):
            keys = self.dictionary_keys(node.subject, scope)
            sinks.append(("value", node.subject, scope))
            for case in node.cases:
                for pattern in ast.walk(case.pattern):
                    if isinstance(pattern, ast.MatchValue):
                        sinks.append(("key" if keys else "value", pattern.value, scope))
                        if keys:
                            sinks.append(("value", pattern.value, scope))
                    elif isinstance(pattern, ast.MatchMapping):
                        sinks += [("key", key, scope) for key in pattern.keys]
        return sinks

    def arguments(self, call: ast.Call, callee: Scope, param: str) -> list[ast.expr]:
        """The argument a call passes for a parameter of a function it may call."""
        found = [keyword.value for keyword in call.keywords if keyword.arg == param]
        if param not in callee.positional:
            return found
        offset = callee.offset if isinstance(call.func, ast.Attribute) else 0
        index = callee.positional.index(param) - offset
        if 0 <= index < len(call.args) and not any(
            isinstance(arg, ast.Starred) for arg in call.args[: index + 1]
        ):
            found.append(call.args[index])
        return found

    def solve(self) -> None:
        static, calls = self.sinks()
        # Parameters used as a key, a collection of keys, a value, a collection of values.
        params: dict[str, set[Atom]] = {kind: set() for kind in ("key", "keys", "value", "values")}
        while True:
            sinks = list(static)
            for call, scope in calls:
                func = call.func
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute) and (
                    func.attr not in BUILTIN_METHODS
                    or (isinstance(func.value, ast.Name) and func.value.id in ("self", "cls"))
                ):
                    name = func.attr
                else:
                    continue
                for callee in self.functions.get(name, ()):
                    for kind, atoms in params.items():
                        for param in callee.params:
                            if (id(callee.node), param) in atoms:
                                sinks += [
                                    (kind, arg, scope)
                                    for arg in self.arguments(call, callee, param)
                                ]
            before = sum(len(atoms) for atoms in params.values())
            for kind, expr, scope in sinks:
                target = self.keys if kind.startswith("key") else self.values
                scalar_kind = "key" if kind.startswith("key") else "value"
                if kind in ("key", "value"):
                    names, atoms = self.scalar(expr, scope, set())
                    target |= names
                    params[scalar_kind] |= atoms
                    continue
                elements, atoms = self.members(expr, scope, set())
                params[kind] |= atoms
                for element, element_scope in elements:
                    names, element_atoms = self.scalar(element, element_scope, set())
                    target |= names
                    params[scalar_kind] |= element_atoms
            if sum(len(atoms) for atoms in params.values()) == before:
                return


def folded(name: str) -> str:
    """A name as core may spell a value it normalizes: lower case, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def added_by(since: str) -> str:
    """The version, or extension, that added a key or value."""
    since = since.strip()
    if VERSION.match(since):
        return since
    later = LATER_VERSION.search(since)
    if later:
        return later.group(1)
    extension = EXTENSION.search(since)
    return f"extension {extension.group(1)}" if extension else "other"


def name_values(possible: str) -> list[tuple[str, str]]:
    """The names a key can take, each with the version that added it ("" for the key's)."""
    values: list[tuple[str, str]] = []
    for version, name in SINCE_VALUE.findall(possible):
        values.append((name.strip(), version))
    plain = SINCE_VALUE.sub("", possible)
    for group in re.findall(r"\[([^\]]*)\]", plain):
        for token in group.split(","):
            token = token.strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.#-]*", token) and not token.startswith("fn"):
                values.append((token, ""))
    return values


def objects_list(objects: set[str]) -> str:
    shown = sorted(objects)[:SHOWN_OBJECTS]
    rest = len(objects) - len(shown)
    return ", ".join(shown) + (f" and {rest} more" if rest else "")


def main(argv: list[str]) -> int:
    tables = arlington_tables(argv)
    reads = KeyReads(SOURCE)
    # Keys are read by their exact names; values are often normalized ("Multiply" as
    # "multiply"), so they are compared folded.
    core_values = {folded(name) for name in reads.values}
    # added by -> key -> objects; and added by -> (key, value) -> objects.
    keys: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    values: dict[str, dict[tuple[str, str], set[str]]] = defaultdict(lambda: defaultdict(set))
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for table in sorted(tables.glob("*.tsv")):
        obj = table.stem
        for row in csv.DictReader(table.open(newline=""), delimiter="\t"):
            key = row["Key"]
            if key.isdigit() or key == "*" or row.get("DeprecatedIn"):
                continue
            version = added_by(row["SinceVersion"])
            totals[version][0] += 1
            if key in reads.keys:
                totals[version][1] += 1
                if "name" in row["Type"].split(";"):
                    for value, value_version in name_values(row["PossibleValues"]):
                        if folded(value) not in core_values:
                            values[value_version or version][(key, value)].add(obj)
            else:
                keys[version][key].add(obj)
    order = [*VERSIONS, *sorted(version for version in totals if version not in VERSIONS)]
    lines = [
        "# Specification coverage",
        "",
        "Generated by `scripts/spec_coverage.py` from the [Arlington PDF Model]"
        f"(https://github.com/pdf-association/arlington-pdf-model/tree/{ARLINGTON_COMMIT})",
        f"(commit `{ARLINGTON_COMMIT[:8]}`), against the dictionary keys core's source reads.",
        "For each version or extension: the keys PDF defines that core never reads, and, for",
        "keys it reads, the name values it never compares.",
        "",
        "A key counts as read where a literal naming it is used as a dictionary key",
        '(`d.get("K")`, `d["K"]`, `"K" in d`, a `case {"K": ...}` pattern, a comparison with',
        "the keys of a dictionary being iterated) or is passed to a parameter a function so",
        'uses, followed to a fixed point, so `lookup_page_resource("Font", ...)`,',
        '`inheritable(..., "FT")` and `require_int("Columns")` count. Merely mentioning a',
        "name, as a value, a dictionary written or a message, does not. A value counts as",
        "handled where it is compared (`==`, `in`, `case`), is a key of a table looked up by",
        "value, or is an entry of a module-level table; values are compared ignoring case and",
        "punctuation, as core often normalizes them.",
        "",
        "Reads are matched by key name across all of core, not attributed to the object",
        "read: a key read in one object (`S` of an action) counts as read in every object",
        "with a key of that name. So *Read by core* is an upper bound, and a key listed as",
        "never read is never looked up by name (code that copies or walks a whole dictionary",
        "still passes it through). Deprecated keys are left out. See also the",
        "[specification index](spec-index.md).",
        "",
        "| Added by | Keys, by object | Read by core | Never read |",
        "| --- | --- | --- | --- |",
    ]
    for version in order:
        total, read = totals[version]
        label = f"PDF {version}" if VERSION.match(version) else version
        lines.append(f"| {label} | {total} | {read} | {total - read} |")
    for version in order:
        label = (
            f"PDF {version}"
            if VERSION.match(version)
            else version.replace("extension", "Extension")
        )
        missing_keys = keys.get(version, {})
        missing_values = values.get(version, {})
        if not missing_keys and not missing_values:
            continue
        lines += ["", f"## {label}"]
        if missing_keys:
            lines += ["", "Keys core never reads:", ""]
            lines += [
                f"- `{key}` in {objects_list(objects)}"
                for key, objects in sorted(missing_keys.items())
            ]
        if missing_values:
            lines += ["", "Values core never compares, of keys it reads:", ""]
            lines += [
                f"- `{key}` `{value}` in {objects_list(objects)}"
                for (key, value), objects in sorted(missing_values.items())
            ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

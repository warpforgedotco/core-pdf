# SPDX-License-Identifier: AGPL-3.0-only
"""``q``/``Q`` push and pop the same graphics-state fields, in the same order.

``op_q`` appends a positional tuple and ``op_Q`` unpacks it back into the same
attributes. Nothing in the type system couples the two lists: fourteen of the
slots are ``float`` and five are ``int``, so a field added to one and forgotten
in the other -- or two same-typed neighbours transposed -- restores every later
field into the wrong attribute, silently.

The pair is measurably the fastest form (a zip/setattr restore benchmarks ~13x
slower than the tuple unpack, on a path that runs hundreds of times per page),
so the duplication is deliberate and this test is what makes it safe.
"""

from __future__ import annotations

import ast
import pathlib

STATE_SOURCE = (
    pathlib.Path(__file__).resolve().parents[6] / "src/core_pdf/impl/spec/s_07_content/state.py"
)


def internal_method(name: str) -> ast.FunctionDef:
    module = ast.parse(STATE_SOURCE.read_text())
    text_state = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "TextState"
    )
    return next(
        node for node in text_state.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def internal_self_attributes(node: ast.Tuple) -> list[str]:
    return [
        element.attr
        for element in node.elts
        if isinstance(element, ast.Attribute)
        and isinstance(element.value, ast.Name)
        and element.value.id == "self"
    ]


def test_q_pushes_exactly_what_Q_pops() -> None:
    push_call = next(
        node
        for node in ast.walk(internal_method("op_q"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "stack"
    )
    pushed_tuple = push_call.args[0]
    assert isinstance(pushed_tuple, ast.Tuple)
    pushed = internal_self_attributes(pushed_tuple)

    pop_assign = next(
        node
        for node in ast.walk(internal_method("op_Q"))
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
    )
    popped_tuple = pop_assign.targets[0]
    assert isinstance(popped_tuple, ast.Tuple)
    popped = internal_self_attributes(popped_tuple)

    assert pushed, "op_q no longer pushes a tuple of self attributes"
    assert pushed == popped

import ast
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
SOURCE_ROOTS = (REPOSITORY / "src", *sorted((REPOSITORY / "packages").glob("*/src")))


def source_files() -> list[Path]:
    return [
        path for root in SOURCE_ROOTS for path in root.rglob("*.py") if "_vendor" not in path.parts
    ]


def is_type_checking_test(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING") or (
        isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING"
    )


def test_no_type_checking_block_is_empty() -> None:
    empty = [
        f"{path.relative_to(REPOSITORY)}:{node.lineno}"
        for path in source_files()
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.If)
        and is_type_checking_test(node.test)
        and all(isinstance(statement, ast.Pass) for statement in node.body)
        and not node.orelse
    ]
    assert not empty

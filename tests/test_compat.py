"""Guards that keep the package installable on the oldest Python we claim."""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCES = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))


def _has_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _uses_annotations(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns or any(a.annotation for a in node.args.args):
                return True
        if isinstance(node, ast.AnnAssign):
            return True
    return False


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_annotations_are_postponed(path: pathlib.Path) -> None:
    """`str | None` in a signature is evaluated at def time before Python 3.10.

    Tests count too: a module-level `def` in a test file is executed on import,
    so a missing future import breaks collection rather than a single test.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if not _uses_annotations(tree):
        return
    assert _has_future_annotations(tree), (
        f"{path.relative_to(ROOT)} uses annotations without "
        "'from __future__ import annotations'"
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_syntax_newer_than_python_38(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        assert not isinstance(node, ast.NamedExpr), (
            f"{path.relative_to(ROOT)}:{node.lineno} uses the walrus operator"
        )
        if hasattr(ast, "Match"):
            assert not isinstance(node, ast.Match), (
                f"{path.relative_to(ROOT)}:{node.lineno} uses a match statement (3.10+)"
            )

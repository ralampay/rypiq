"""Guard the operation boundaries documented in CODE_CONVENTIONS.md."""
import ast
from pathlib import Path

import pytest


OPERATIONS = Path(__file__).resolve().parents[3] / 'app' / 'operations'
MODULES = sorted(OPERATIONS.rglob('*.py'))


@pytest.mark.parametrize('path', MODULES, ids=lambda path: str(path.relative_to(OPERATIONS)))
def test_operations_stay_in_domain_modules_and_do_not_import_entrypoints(path):
    tree = ast.parse(path.read_text())
    if path.name == '__init__.py':
        assert all(isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                   and isinstance(node.value.value, str) for node in tree.body)
        return
    assert path.parent != OPERATIONS, 'Place actions in their owning domain package'
    forbidden = ('app.application', 'app.cli', 'app.controllers', 'app.api', 'app.worker')
    for node in ast.walk(tree):
        imports = []
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, 'Use explicit owning-module imports in operations'
            imports = [node.module or '']
            imports.extend(f'{node.module}.{alias.name}' for alias in node.names)
        for name in imports:
            assert not any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden), name
    for command in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        methods = {node.name: node for node in command.body if isinstance(node, ast.FunctionDef)}
        assert {name for name in methods if not name.startswith('_')} == {'execute'}
        assert methods['execute'].returns is not None
        constructor = methods['__init__']
        assert [arg.arg for arg in constructor.args.args] == ['self']
        assert not constructor.args.posonlyargs
        assert constructor.args.vararg is None
        assert constructor.args.kwarg is None

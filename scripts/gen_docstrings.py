"""

Generates function docstring templates for all python files.

"""

import os
import ast
from astor import to_source

_DIR = '../solvent'
_EXT = '.py'
_EXCLUDE = ['__init__.py']


def gen_doc() -> str:
    triple_quote = '"""'
    s = f"""\n    {triple_quote}
    TODO
    
    Args:
        TODO (TODO): TODO
          
    Returns:
         TODO (TODO): TODO

    {triple_quote}"""
    return s
    # return ast.Expr(value=ast.Str(s))


class Visitor(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if ast.get_docstring(node) is None:
            new_func = ast.FunctionDef()
            new_func.decorator_list = node.decorator_list
            new_func.name = node.name
            new_func.args = node.args
            new_func.body = node.body
            new_func.returns = node.returns
            docstring = gen_doc()
            new_func.body.insert(0, docstring) # type: ignore
            ast.NodeVisitor.generic_visit(self, new_func)
            return new_func
        return node

def write_func(node_visitor: Visitor, func: ast.FunctionDef) -> str:
    return to_source(node_visitor.visit(func)).replace('->', '-> ')

def main() -> None:
    node_visitor = Visitor()
    for subdir, dirs, files in os.walk(_DIR):
        for file in files:
            if file.endswith(_EXT) and file not in _EXCLUDE:
                path = os.path.join(subdir, file)
                with open(path, 'r') as f:
                    code = f.read()
                    tree = ast.parse(code)
                    for c in ast.walk(tree):
                        if isinstance(c, ast.ClassDef):
                            continue
                        elif isinstance(c, ast.FunctionDef):
                            s = write_func(node_visitor, c)
                            print(s)

if __name__ == '__main__':
    main()

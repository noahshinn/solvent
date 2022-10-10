"""

Generates function docstring templates for all python files.

"""

import sys
import ast
import subprocess
from astor import to_source

assert len(sys.argv) == 2
FILE = sys.argv[1]


def _gen_doc(args: ast.arguments, returns: ast.Return) -> str:
    l = len(args.args)
    arg_s = ''
    for i, a in enumerate(args.args):
        if a.annotation is None:
            continue
        p = to_source(a.annotation).replace("\n", '')
        arg_s += f'            {a.arg} ({p}): TODO' # type: ignore
        if i != l - 1:
            arg_s += '\n'
    r = to_source(returns).replace('\n', '')
    triple_quote = '"""'
    s = f"""        {triple_quote}
        TODO
    
        Args:
{arg_s}
          
        Returns:
            TODO ({r}): TODO

        {triple_quote}"""
    return s

def gen_doc(file: str) -> None:
    with open(file, 'r') as f:
        code = f.read()
        tree = ast.parse(code)
        for c in ast.walk(tree):
            if isinstance(c, ast.FunctionDef):
                docstring = _gen_doc(c.args, c.returns) # type: ignore
                # subprocess.run("pbcopy", universal_newlines=True, input=docstring)
                print(docstring)
                

if __name__ == '__main__':
    gen_doc(FILE)

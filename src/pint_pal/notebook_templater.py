import nbformat
import re
import argparse
import sys

assignment = re.compile(r"^(\w+)\s*=\s*(.*)$")

def transform_notebook(nb, transformations, verbose=False):
    """Change variable assignments in a loaded notebook.
    
    This looks for lines in the code that look like
    
    >>> variable = value
    
    If variable is in transformations, value is replaced with the string that
    transformations maps variable to. This is done only for non-indented lines,
    and only once, so as not to catch keyword arguments.
    
    Parameters
    ----------
    nb : nbformat.NotebookNode (dict-like object coming from nbformat.read() on a notebook)
        The notebook whose code cells should be transformed.
    transformations : dict
        A dictionary mapping variable names to string representations of their values.
    """
    subs = 0
    transformed = {k: False for k in transformations}
    for cell in nb["cells"]:
        if cell["cell_type"]!="code":
            continue
        lines = []
        for i, l in enumerate(cell["source"].split("\n")):
            m = assignment.match(l)
            if not m:
                lines.append(l)
                continue
            k = m.group(1)
            if k in transformations and not transformed[k]:
                val = transformations[k]
                transformed[k] = True
                new_line = f"{m.group(1)} = {val}"
                if verbose:
                    print(f"replacing line {repr(l)} by {repr(new_line)}")
                lines.append(new_line)
                subs += 1
            else:
                lines.append(l)
        cell["source"] = "\n".join(lines)
    return subs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Substitute variables on a Jupyter notebook.
    
    This script allows you to substitute values into variable assignments in a notebook.
    The idea is that it allows you to have a variable, say 'write_results' that controls
    the notebook's behaviour. Then in a template notebook there will be a line that says
    'write_results = False'; this script lets you replace this with 'write_results = True'.
    
    Variable assignments will be replaced only once, and only if they are not indented, to avoid
    confusion with keyword arguments.
    """)
    def parse(s):
        key, value = s.split("=")
        return key.strip(), value.strip()
    parser.add_argument("-s", "--set-variable", type=parse, action="append", 
                        help="The variable to replace and its value, separated by an equal sign. "
                        "May require quotes to protect it from your shell; the value is "
                        "python source code (so strings need quotes in addition to whatever "
                        "the shell requires. Can be specified multiple times to have multiple "
                        "substitutions.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Describe all substitutions.")
    parser.add_argument("infile", help="Input file to transform")
    parser.add_argument("outfile", help="Output file to write")
    args = parser.parse_args()
    
    if not args.set_variable:
        print("No substitutions requested, no action will be taken.", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)
    transformations = {k: v for (k,v) in args.set_variable}
    with open(args.infile) as f:
        nb = nbformat.read(f, as_version=4)
    if not transform_notebook(nb, transformations, verbose=args.verbose):
        print(f"No substitutions performed; requested were {transformations}", file=sys.stderr)
    with open(args.outfile, "w") as f:
        nb = nbformat.write(nb, f)
    
    
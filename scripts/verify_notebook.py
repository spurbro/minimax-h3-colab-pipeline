#!/usr/bin/env python3
"""
Validate all Jupyter Notebooks (.ipynb) in the repository for JSON schema & Python AST syntax integrity.
"""

import ast
import glob
import json
import os
import sys


def verify_notebook_file(nb_path: str) -> bool:
    print("=" * 68)
    print(f"[VERIFY] Checking notebook: {nb_path}")
    print("=" * 68)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb.get("nbformat") == 4, "nbformat must be 4"
    cells = nb.get("cells", [])
    assert len(cells) > 0, "Notebook contains 0 cells"

    md_count = 0
    code_count = 0

    for idx, cell in enumerate(cells):
        ctype = cell.get("cell_type")
        source_lines = cell.get("source", [])
        source_text = "".join(source_lines) if isinstance(source_lines, list) else str(source_lines)

        if ctype == "markdown":
            md_count += 1
        elif ctype == "code":
            code_count += 1
            # Strip IPython magic lines (! or %) if any before AST parsing
            clean_lines = []
            for line in source_text.splitlines():
                stripped = line.lstrip()
                if stripped.startswith("!") or stripped.startswith("%"):
                    clean_lines.append("pass")
                else:
                    clean_lines.append(line)
            clean_code = "\n".join(clean_lines)
            try:
                ast.parse(clean_code)
            except SyntaxError as e:
                print(f"[ERROR] SyntaxError in {nb_path} cell #{idx + 1}: {e}")
                return False

    print(f"[OK] Validated {len(cells)} cells ({md_count} markdown, {code_count} code) with 0 syntax errors.")
    return True


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nb_files = sorted(glob.glob(os.path.join(repo_root, "notebooks", "*.ipynb")))
    if not nb_files:
        print("[ERROR] No .ipynb files found in notebooks/")
        sys.exit(1)

    all_ok = True
    for nb_file in nb_files:
        if not verify_notebook_file(nb_file):
            all_ok = False

    if not all_ok:
        sys.exit(1)
    print("\n[SUCCESS] All repository notebooks passed 100% AST & structure verification!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fix all torch import variants to use fake_torch pattern
"""

import re
import sys
from pathlib import Path

def fix_torch_imports(file_path):
    """Fix all torch import patterns in a file"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    lines = content.split('\n')
    new_lines = []

    # Track if we've added the fake_torch import
    fake_torch_added = False
    # Track torch imports to convert
    torch_imports = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip if already using fake_torch
        if 'fake_torch' in line:
            new_lines.append(line)
            i += 1
            continue

        # Pattern 1: import torch.nn as nn
        match = re.match(r'^(\s*)import torch\.(\S+)(\s+as\s+(\w+))?', line)
        if match:
            indent = match.group(1)
            module_path = match.group(2)
            alias = match.group(4) if match.group(4) else module_path.split('.')[-1]

            if not fake_torch_added:
                new_lines.append(f"{indent}from apt_model.utils.fake_torch import get_torch")
                new_lines.append(f"{indent}torch = get_torch()")
                fake_torch_added = True

            # Add the module assignment
            new_lines.append(f"{indent}{alias} = torch.{module_path}")
            i += 1
            continue

        # Pattern 2: from torch.X import Y, Z
        match = re.match(r'^(\s*)from torch\.(\S+) import (.+)', line)
        if match:
            indent = match.group(1)
            module_path = match.group(2)
            imports = match.group(3)

            if not fake_torch_added:
                new_lines.append(f"{indent}from apt_model.utils.fake_torch import get_torch")
                new_lines.append(f"{indent}torch = get_torch()")
                fake_torch_added = True

            # Handle multiple imports (Y, Z)
            import_items = [item.strip() for item in imports.split(',')]
            for item in import_items:
                # Handle "import X as Y" pattern
                if ' as ' in item:
                    orig_name, alias = item.split(' as ')
                    orig_name = orig_name.strip()
                    alias = alias.strip()
                    new_lines.append(f"{indent}{alias} = torch.{module_path}.{orig_name}")
                else:
                    new_lines.append(f"{indent}{item} = torch.{module_path}.{item}")
            i += 1
            continue

        # Pattern 3: plain "import torch" (should already be fixed, but double check)
        if re.match(r'^(\s*)import torch\s*$', line):
            indent = re.match(r'^(\s*)', line).group(1)
            if not fake_torch_added:
                new_lines.append(f"{indent}from apt_model.utils.fake_torch import get_torch")
                new_lines.append(f"{indent}torch = get_torch()")
                fake_torch_added = True
            i += 1
            continue

        # Keep other lines as-is
        new_lines.append(line)
        i += 1

    new_content = '\n'.join(new_lines)

    # Only write if content changed
    if new_content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

def main():
    # Find all Python files in apt/core
    apt_core = Path('/home/user/APT-Transformer/apt/core')

    if not apt_core.exists():
        print(f"Error: {apt_core} not found")
        return 1

    files_fixed = 0
    for py_file in apt_core.rglob('*.py'):
        if fix_torch_imports(py_file):
            print(f"Fixed: {py_file}")
            files_fixed += 1

    print(f"\n✅ Fixed {files_fixed} files")
    return 0

if __name__ == '__main__':
    sys.exit(main())

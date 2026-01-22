#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APX Templates

YAML generation and file writing utilities for APX packaging.
"""

import textwrap
from pathlib import Path
from typing import Dict, List, Optional


def write_text(path: Path, content: str) -> None:
    """
    Write text content to file, creating parent directories if needed.

    Args:
        path: File path to write to
        content: Text content to write
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def yaml_dump_block(d: Dict[str, str], indent: int = 2) -> str:
    """
    Dump dictionary as YAML block.

    Args:
        d: Dictionary to dump
        indent: Indentation level (spaces)

    Returns:
        YAML formatted string
    """
    lines = []
    for k, v in d.items():
        lines.append(" " * indent + f"{k}: {v}")
    return "\n".join(lines)


def make_apx_yaml(
    name: str,
    version: str,
    entry_model: str,
    entry_tokenizer: str,
    artifacts: Dict[str, str],
    prefers: str = "builtin",
    capabilities: Optional[List[str]] = None,
    compose_kv: Optional[Dict[str, str]] = None,
) -> str:
    """
    Generate apx.yaml manifest content.

    Args:
        name: Model name
        version: Model version
        entry_model: Model adapter entry point (path:class)
        entry_tokenizer: Tokenizer adapter entry point (path:class)
        artifacts: Dictionary mapping artifact names to paths
        prefers: Preference for builtin vs plugin ("builtin" or "plugin")
        capabilities: Optional list of capability names
        compose_kv: Optional compose key-value pairs

    Returns:
        YAML formatted string

    Example:
        >>> yaml = make_apx_yaml(
        ...     name="my-model",
        ...     version="1.0.0",
        ...     entry_model="model/adapters/hf_adapter.py:HFAdapter",
        ...     entry_tokenizer="model/adapters/tokenizer_adapter.py:HFTokenizerAdapter",
        ...     artifacts={"config": "artifacts/config.json", "weights": "artifacts/model.safetensors"},
        ...     capabilities=["moe", "tva"],
        ...     compose_kv={"router": "observe_only"}
        ... )
    """
    capabilities = capabilities or []
    compose_kv = compose_kv or {}

    # Build capabilities section
    if capabilities:
        cap_items = "\n".join([f"    - {c}" for c in capabilities])
        cap_str = (
            f"capabilities:\n"
            f"  provides:\n"
            f"{cap_items}\n"
            f"  prefers:\n"
            f"    - {prefers}\n"
        )
    else:
        cap_str = f"capabilities:\n  prefers:\n    - {prefers}\n"

    # Build compose section
    compose_str = ""
    if compose_kv:
        compose_lines = "\n".join([f"  {k}: {v}" for k, v in compose_kv.items()])
        compose_str = f"compose:\n{compose_lines}\n"

    # Build artifacts section
    art_lines = "\n".join([f"  {k}: {v}" for k, v in artifacts.items()])

    # Generate complete YAML
    yaml_content = textwrap.dedent(
        f"""\
    apx_version: 1
    name: {name}
    version: {version}
    type: model
    entrypoints:
      model_adapter: {entry_model}
      tokenizer_adapter: {entry_tokenizer}
    artifacts:
{art_lines}
{cap_str}{compose_str}"""
    )

    return yaml_content

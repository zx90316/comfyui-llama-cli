import os
import json
import tempfile
from typing import Optional


def validate_binary_dir(path: str) -> tuple:
    """Validate that the directory exists and contains llama binaries.

    Returns (is_valid, available_binaries, error_message).
    """
    if not path or not os.path.isdir(path):
        return False, [], f"Directory not found: {path}"

    expected = ["llama-cli", "llama-embedding"]
    found = []
    for name in expected:
        exe_name = name + ".exe" if os.name == "nt" else name
        full_path = os.path.join(path, exe_name)
        if os.path.isfile(full_path):
            found.append(name)

    if not found:
        return False, [], f"No llama.cpp binaries found in: {path}"

    return True, found, ""


def resolve_binary_path(binary_dir: str, binary_name: str) -> str:
    """Resolve the full path of a binary, adding .exe on Windows."""
    binary = os.path.join(binary_dir, binary_name)
    if os.name == "nt" and not binary.endswith(".exe"):
        binary += ".exe"
    return binary


def parse_json_field(value: str, default=None):
    """Safely parse a JSON string, returning default on failure."""
    if not value or not value.strip():
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def write_temp_grammar(grammar_str: str, temp_dir: Optional[str] = None) -> str:
    """Write a BNF grammar string to a temp file for --grammar-file."""
    fd, path = tempfile.mkstemp(suffix=".gbnf", dir=temp_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(grammar_str)
    return path

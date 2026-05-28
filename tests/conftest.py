import sys
import os
import types
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = "comfyui_llama_cli"

# Mock folder_paths (ComfyUI-only module)
if "folder_paths" not in sys.modules:
    fp = types.ModuleType("folder_paths")
    fp.models_dir = os.path.join(ROOT, "models")
    fp.get_filename_list = lambda x: []
    fp.get_full_path_or_raise = lambda folder, name: name
    fp.add_model_folder_path = lambda *args: None
    sys.modules["folder_paths"] = fp


def _register_package(name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


def _import_module(parent_pkg, mod_name, file_path):
    full_name = f"{parent_pkg}.{mod_name}"
    spec = importlib.util.spec_from_file_location(full_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = parent_pkg
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Build the package tree so relative imports inside nodes/ and utils/ resolve
_register_package(PKG, ROOT)
_register_package(f"{PKG}.nodes", os.path.join(ROOT, "nodes"))
_register_package(f"{PKG}.utils", os.path.join(ROOT, "utils"))

_import_module(PKG, "process_manager", os.path.join(ROOT, "process_manager.py"))

for name in ["command_builder", "output_parser", "param_helpers"]:
    _import_module(f"{PKG}.utils", name, os.path.join(ROOT, "utils", f"{name}.py"))

for name in [
    "cli_config_node", "generate_node", "chat_node", "embedding_node",
    "infill_node", "image_encoder_node", "process_monitor_node",
]:
    _import_module(f"{PKG}.nodes", name, os.path.join(ROOT, "nodes", f"{name}.py"))

# Pre-register the root __init__ so pytest does not try to exec it from disk
# (pytest walks up directories and attempts to import __init__.py when it
# finds one, which fails because the dir name contains hyphens).
sys.modules["__init__"] = types.ModuleType("__init__")

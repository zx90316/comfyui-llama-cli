import os
import shlex
import tempfile
from typing import Optional


def build_base_cmd(config: dict, binary_name: str = "llama-cli") -> list:
    """Build the base command from CLI_CONFIG for any llama.cpp binary."""
    binary = os.path.join(config["binary_dir"], binary_name)
    if os.name == "nt" and not binary.endswith(".exe"):
        binary += ".exe"

    if not os.path.isfile(binary):
        raise FileNotFoundError(f"Binary not found: {binary}")

    cmd = [binary, "-m", config["model_path"]]

    cmd.extend(["-ngl", str(config.get("n_gpu_layers", 99))])
    cmd.extend(["-c", str(config.get("context_size", 4096))])

    if config.get("flash_attention", True):
        cmd.extend(["-fa", "on"])
    else:
        cmd.extend(["-fa", "off"])

    split_mode = config.get("split_mode", "none")
    cmd.extend(["--split-mode", split_mode])

    cmd.extend(["--main-gpu", str(config.get("main_gpu", 0))])

    threads = config.get("threads", 0)
    if threads > 0:
        cmd.extend(["-t", str(threads)])

    n_parallel = config.get("n_parallel", 1)
    if n_parallel > 1:
        cmd.extend(["-np", str(n_parallel)])

    cache_type_k = config.get("cache_type_k", "f16")
    if cache_type_k != "f16":
        cmd.extend(["--cache-type-k", cache_type_k])

    cache_type_v = config.get("cache_type_v", "f16")
    if cache_type_v != "f16":
        cmd.extend(["--cache-type-v", cache_type_v])

    # --jinja is NOT added here; it's chat-specific and added by chat_node only

    mmproj_path = config.get("mmproj_path")
    if mmproj_path and mmproj_path != "none":
        cmd.extend(["--mmproj", mmproj_path])
        if not config.get("mmproj_offload", True):
            cmd.append("--no-mmproj-offload")

    spec_type = config.get("spec_type", "none")
    if spec_type != "none":
        cmd.extend(["--spec-type", spec_type])

    spec_draft_n_max = config.get("spec_draft_n_max", 0)
    if spec_draft_n_max > 0:
        cmd.extend(["--spec-draft-n-max", str(spec_draft_n_max)])

    spec_draft_model_path = config.get("spec_draft_model_path")
    if spec_draft_model_path:
        cmd.extend(["--model-draft", spec_draft_model_path])

    extra_args = config.get("extra_args", "").strip()
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    return cmd


def add_sampling_args(cmd: list, params: dict) -> list:
    """Append sampling-related CLI arguments."""
    mapping = {
        "temperature": "--temp",
        "top_k": "--top-k",
        "top_p": "--top-p",
        "min_p": "--min-p",
        "seed": "-s",
        "repeat_penalty": "--repeat-penalty",
        "repeat_last_n": "--repeat-last-n",
        "presence_penalty": "--presence-penalty",
        "frequency_penalty": "--frequency-penalty",
        "typical_p": "--typical",
        "dynatemp_range": "--dynatemp-range",
        "dynatemp_exponent": "--dynatemp-exp",
        "mirostat": "--mirostat",
        "mirostat_tau": "--mirostat-tau",
        "mirostat_eta": "--mirostat-eta",
        "dry_multiplier": "--dry-multiplier",
        "dry_base": "--dry-base",
        "dry_allowed_length": "--dry-allowed-length",
        "dry_penalty_last_n": "--dry-penalty-last-n",
    }

    for param_key, cli_flag in mapping.items():
        value = params.get(param_key)
        if value is not None:
            cmd.extend([cli_flag, str(value)])

    return cmd


def add_generation_args(cmd: list, params: dict, temp_dir: Optional[str] = None) -> list:
    """Append generation control CLI arguments (grammar, stop, predict, etc.)."""
    n_predict = params.get("n_predict")
    if n_predict is not None:
        cmd.extend(["-n", str(n_predict)])

    grammar = params.get("grammar", "").strip()
    if grammar:
        grammar_file = _write_temp_file(grammar, suffix=".gbnf", temp_dir=temp_dir)
        cmd.extend(["--grammar-file", grammar_file])

    json_schema = params.get("json_schema", "").strip()
    if json_schema:
        cmd.extend(["--json-schema", json_schema])

    stop_sequences = params.get("stop_sequences", "").strip()
    if stop_sequences and stop_sequences != "[]":
        import json
        try:
            stops = json.loads(stop_sequences)
            if isinstance(stops, list):
                for s in stops:
                    cmd.extend(["--reverse-prompt", str(s)])
        except (json.JSONDecodeError, TypeError):
            pass

    image_path = params.get("image_path", "").strip()
    if image_path:
        cmd.extend(["--image", image_path])

    if params.get("log_disable", True):
        cmd.append("--log-disable")

    cmd.append("--no-display-prompt")

    return cmd


def _write_temp_file(content: str, suffix: str = ".txt", temp_dir: Optional[str] = None) -> str:
    """Write content to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix, dir=temp_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path

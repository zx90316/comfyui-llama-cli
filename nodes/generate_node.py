from ..process_manager import LlamaProcessManager
from ..utils.command_builder import build_base_cmd, add_sampling_args, add_generation_args
from ..utils.output_parser import parse_generation_output, parse_chat_response


class LlamaGenerateNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cli_config": ("CLI_CONFIG", {
                    "tooltip": "Configuration from Llama CLI Config node",
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "The prompt text for generation",
                }),
            },
            "optional": {
                "n_predict": ("INT", {
                    "default": 256, "min": -1, "max": 100000,
                    "tooltip": "-n: Max tokens to generate (-1 = infinite)",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "--temp: Sampling temperature",
                }),
                "top_k": ("INT", {
                    "default": 40, "min": 0, "max": 1000,
                    "tooltip": "--top-k: Top-K sampling",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "--top-p: Nucleus sampling",
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "--min-p: Min-P sampling",
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 2**31 - 1,
                    "tooltip": "-s: Random seed (-1 = random)",
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.1, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "--repeat-penalty: Repetition penalty",
                }),
                "repeat_last_n": ("INT", {
                    "default": 64, "min": -1, "max": 4096,
                    "tooltip": "--repeat-last-n: Lookback window for repeat penalty",
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": "--presence-penalty",
                }),
                "frequency_penalty": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": "--frequency-penalty",
                }),
                "typical_p": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "--typical: Locally typical sampling",
                }),
                "dynatemp_range": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "--dynatemp-range: Dynamic temperature range",
                }),
                "dynatemp_exponent": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01,
                    "tooltip": "--dynatemp-exp: Dynamic temperature exponent",
                }),
                "mirostat": ("INT", {
                    "default": 0, "min": 0, "max": 2,
                    "tooltip": "--mirostat: Mirostat mode (0=off, 1=v1, 2=v2)",
                }),
                "mirostat_tau": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "--mirostat-tau: Target entropy",
                }),
                "mirostat_eta": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "--mirostat-eta: Learning rate",
                }),
                "dry_multiplier": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "--dry-multiplier: DRY sampling multiplier",
                }),
                "dry_base": ("FLOAT", {
                    "default": 1.75, "min": 1.0, "max": 5.0, "step": 0.01,
                    "tooltip": "--dry-base: DRY base value",
                }),
                "dry_allowed_length": ("INT", {
                    "default": 2, "min": 1, "max": 100,
                    "tooltip": "--dry-allowed-length",
                }),
                "dry_penalty_last_n": ("INT", {
                    "default": -1, "min": -1, "max": 4096,
                    "tooltip": "--dry-penalty-last-n",
                }),
                "grammar": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "BNF grammar for constrained generation (written to temp file)",
                }),
                "json_schema": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "--json-schema: JSON Schema constraint",
                }),
                "stop_sequences": ("STRING", {
                    "default": "[]", "multiline": False,
                    "tooltip": "JSON array of stop strings (each becomes --reverse-prompt)",
                }),
                "image_path": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "--image: Path to image file for multimodal (from LlamaImageEncoder)",
                }),
                "log_disable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "--log-disable: Suppress llama.cpp logs in stderr",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "stderr_log", "exit_code")
    FUNCTION = "generate"
    CATEGORY = "AI/LlamaCpp"

    def generate(self, cli_config: dict, prompt: str, **kwargs):
        if not prompt.strip():
            return ("", "Error: empty prompt", -1)

        image_path = kwargs.get("image_path", "").strip()
        has_vision = bool(image_path) and bool(cli_config.get("mmproj_path"))

        mgr = LlamaProcessManager.get_instance()

        if has_vision:
            return self._generate_vision(mgr, cli_config, prompt, kwargs)

        cmd = build_base_cmd(cli_config, binary_name="llama-completion")
        cmd.append("-no-cnv")
        cmd.extend(["-p", prompt])

        sampling_params = {
            k: kwargs[k] for k in [
                "temperature", "top_k", "top_p", "min_p", "seed",
                "repeat_penalty", "repeat_last_n", "presence_penalty",
                "frequency_penalty", "typical_p", "dynatemp_range",
                "dynatemp_exponent", "mirostat", "mirostat_tau", "mirostat_eta",
                "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
            ] if k in kwargs
        }
        add_sampling_args(cmd, sampling_params)

        gen_params = {
            k: kwargs[k] for k in [
                "n_predict", "grammar", "json_schema", "stop_sequences",
            ] if k in kwargs
        }
        gen_params["log_disable"] = False
        add_generation_args(cmd, gen_params)

        if cli_config.get("keep_alive", False):
            try:
                interactive_cmd = build_base_cmd(cli_config, binary_name="llama-completion")
                interactive_cmd.append("--no-display-prompt")
                add_sampling_args(interactive_cmd, sampling_params)
                proc = mgr.get_interactive(interactive_cmd, cli_config)
                text, error, code = proc.send_prompt(prompt, timeout=cli_config.get("timeout", 600))
                text = parse_generation_output(text)
                return (text, error, code)
            except Exception as e:
                return ("", f"Interactive mode error: {e}", -1)
        else:
            stdout, stderr, code = mgr.run_oneshot(cmd, timeout=cli_config.get("timeout", 600))
            text = parse_generation_output(stdout, stderr)
            return (text, stderr, code)

    def _generate_vision(self, mgr, cli_config: dict, prompt: str, kwargs: dict):
        """Vision generation via llama-cli -cnv (only llama-cli supports --mmproj)."""
        cmd = build_base_cmd(cli_config, binary_name="llama-cli")
        if cli_config.get("jinja", True):
            cmd.append("--jinja")
        cmd.append("-cnv")
        cmd.append("--single-turn")
        cmd.extend(["-p", prompt])

        image_path = kwargs.get("image_path", "").strip()
        if image_path:
            cmd.extend(["--image", image_path])

        sampling_params = {
            k: kwargs[k] for k in [
                "temperature", "top_k", "top_p", "min_p", "seed",
                "repeat_penalty",
            ] if k in kwargs
        }
        add_sampling_args(cmd, sampling_params)

        gen_params = {
            "n_predict": kwargs.get("n_predict", 256),
            "log_disable": kwargs.get("log_disable", True),
        }
        add_generation_args(cmd, gen_params)

        timeout = cli_config.get("timeout", 600)
        stdout, stderr, code = mgr.run_oneshot(cmd, timeout=timeout)
        text = parse_chat_response(stdout)
        return (text, stderr, code)

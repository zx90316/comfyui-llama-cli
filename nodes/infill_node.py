from ..process_manager import LlamaProcessManager
from ..utils.command_builder import build_base_cmd, add_sampling_args, add_generation_args
from ..utils.output_parser import parse_generation_output


class LlamaInfillNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cli_config": ("CLI_CONFIG", {
                    "tooltip": "Configuration from Llama CLI Config node",
                }),
                "input_prefix": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "--in-prefix: Code before the fill point",
                }),
                "input_suffix": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "--in-suffix: Code after the fill point",
                }),
            },
            "optional": {
                "n_predict": ("INT", {
                    "default": 128, "min": -1, "max": 100000,
                    "tooltip": "-n: Max tokens to generate",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "--temp",
                }),
                "top_k": ("INT", {
                    "default": 40, "min": 0, "max": 1000,
                    "tooltip": "--top-k",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "--top-p",
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "--min-p",
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 2**31 - 1,
                    "tooltip": "-s: Random seed",
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.1, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "--repeat-penalty",
                }),
                "log_disable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "--log-disable",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("infilled_text", "stderr_log", "exit_code")
    FUNCTION = "infill"
    CATEGORY = "AI/LlamaCpp"

    def infill(self, cli_config: dict, input_prefix: str, input_suffix: str, **kwargs):
        if not input_prefix.strip() and not input_suffix.strip():
            return ("", "Error: both prefix and suffix are empty", -1)

        mgr = LlamaProcessManager.get_instance()
        cmd = build_base_cmd(cli_config, binary_name="llama-completion")
        cmd.append("-no-cnv")

        cmd.append("--infill")
        cmd.extend(["--in-prefix", input_prefix])
        cmd.extend(["--in-suffix", input_suffix])

        sampling = {
            k: kwargs[k] for k in [
                "temperature", "top_k", "top_p", "min_p", "seed", "repeat_penalty",
            ] if k in kwargs
        }
        add_sampling_args(cmd, sampling)

        gen_params = {
            "n_predict": kwargs.get("n_predict", 128),
            "log_disable": False,
        }
        add_generation_args(cmd, gen_params)

        stdout, stderr, code = mgr.run_oneshot(cmd, timeout=cli_config.get("timeout", 600))
        text = parse_generation_output(stdout, stderr)
        return (text, stderr, code)

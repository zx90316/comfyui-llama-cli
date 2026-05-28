from ..process_manager import LlamaProcessManager
from ..utils.command_builder import build_base_cmd
from ..utils.output_parser import parse_embedding_output


class LlamaEmbeddingNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cli_config": ("CLI_CONFIG", {
                    "tooltip": "Configuration from Llama CLI Config node",
                }),
                "input_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text to compute embeddings for",
                }),
            },
            "optional": {
                "embd_normalize": ("INT", {
                    "default": 2, "min": -1, "max": 10,
                    "tooltip": "--embd-normalize: Embedding normalization type",
                }),
                "embd_separator": ("STRING", {
                    "default": "\\n",
                    "multiline": False,
                    "tooltip": "--embd-separator: Separator for multiple inputs",
                }),
                "log_disable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "--log-disable: Suppress logs",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("embedding", "stderr_log", "exit_code")
    FUNCTION = "embed"
    CATEGORY = "AI/LlamaCpp"

    def embed(self, cli_config: dict, input_text: str, **kwargs):
        if not input_text.strip():
            return ("[]", "Error: empty input text", -1)

        mgr = LlamaProcessManager.get_instance()

        cmd = build_base_cmd(cli_config, binary_name="llama-embedding")
        cmd.extend(["-p", input_text])

        embd_normalize = kwargs.get("embd_normalize", 2)
        cmd.extend(["--embd-normalize", str(embd_normalize)])

        embd_separator = kwargs.get("embd_separator", "\\n")
        if embd_separator and embd_separator != "\\n":
            cmd.extend(["--embd-separator", embd_separator])

        if kwargs.get("log_disable", True):
            cmd.append("--log-disable")

        stdout, stderr, code = mgr.run_oneshot(cmd, timeout=cli_config.get("timeout", 600))

        embedding = parse_embedding_output(stdout)
        return (embedding, stderr, code)

import json
import tempfile
import os

from ..process_manager import LlamaProcessManager
from ..utils.command_builder import build_base_cmd, add_sampling_args, add_generation_args
from ..utils.output_parser import parse_chat_response


class LlamaChatNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cli_config": ("CLI_CONFIG", {
                    "tooltip": "Configuration from Llama CLI Config node",
                }),
                "user_message": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "User message for the current turn",
                }),
            },
            "optional": {
                "system_message": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "System prompt to set model behavior",
                }),
                "chat_history": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": 'JSON array of prior messages, e.g. [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"}]',
                }),
                "chat_template": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "--chat-template: Custom chat template name",
                }),
                "max_tokens": ("INT", {
                    "default": 1024, "min": -1, "max": 100000,
                    "tooltip": "-n: Max tokens in response",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "--temp: Sampling temperature",
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
                "json_schema": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "--json-schema: Constrain output to JSON schema",
                }),
                "image_path": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "--image: Image file for multimodal chat",
                }),
                "log_disable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "--log-disable",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("response", "updated_history", "stderr_log", "exit_code")
    FUNCTION = "chat"
    CATEGORY = "AI/LlamaCpp"

    def chat(self, cli_config: dict, user_message: str, **kwargs):
        if not user_message.strip():
            return ("", "[]", "Error: empty user message", -1)

        history = []
        raw_history = kwargs.get("chat_history", "[]")
        if raw_history and raw_history.strip() and raw_history.strip() != "[]":
            try:
                history = json.loads(raw_history)
                if not isinstance(history, list):
                    history = []
            except (json.JSONDecodeError, TypeError):
                history = []

        system_message = kwargs.get("system_message", "").strip()

        full_prompt = self._build_prompt_file(system_message, history, user_message)

        mgr = LlamaProcessManager.get_instance()
        cmd = build_base_cmd(cli_config)

        if cli_config.get("jinja", True):
            cmd.append("--jinja")

        prompt_file = self._write_temp_prompt(full_prompt)
        cmd.extend(["-f", prompt_file])
        cmd.append("-cnv")
        cmd.append("--single-turn")

        chat_template = kwargs.get("chat_template", "").strip()
        if chat_template:
            cmd.extend(["--chat-template", chat_template])

        sampling = {
            k: kwargs[k] for k in [
                "temperature", "top_k", "top_p", "min_p", "seed", "repeat_penalty",
            ] if k in kwargs
        }
        add_sampling_args(cmd, sampling)

        gen_params = {
            "n_predict": kwargs.get("max_tokens", 1024),
            "json_schema": kwargs.get("json_schema", ""),
            "image_path": kwargs.get("image_path", ""),
            "log_disable": kwargs.get("log_disable", True),
        }
        add_generation_args(cmd, gen_params)

        timeout = cli_config.get("timeout", 600)

        if cli_config.get("keep_alive", False):
            try:
                interactive_cmd = build_base_cmd(cli_config)
                if cli_config.get("jinja", True):
                    interactive_cmd.append("--jinja")
                interactive_cmd.append("-cnv")
                interactive_cmd.append("--no-display-prompt")
                if chat_template:
                    interactive_cmd.extend(["--chat-template", chat_template])
                add_sampling_args(interactive_cmd, sampling)

                proc = mgr.get_interactive(interactive_cmd, cli_config)
                text, error, code = proc.send_prompt(user_message, timeout=timeout)
                response = parse_chat_response(text)
            except Exception as e:
                response = ""
                error = f"Interactive mode error: {e}"
                code = -1
        else:
            stdout, stderr, code = mgr.run_oneshot(cmd, timeout=timeout)
            response = parse_chat_response(stdout)
            error = stderr

        try:
            os.unlink(prompt_file)
        except OSError:
            pass

        new_history = list(history)
        if system_message and not any(m.get("role") == "system" for m in new_history):
            new_history.insert(0, {"role": "system", "content": system_message})
        new_history.append({"role": "user", "content": user_message})
        if response:
            new_history.append({"role": "assistant", "content": response})

        return (response, json.dumps(new_history, ensure_ascii=False), error, code)

    def _build_prompt_file(self, system_message: str, history: list, user_message: str) -> str:
        """Build the full conversation as a JSON messages array for -f input."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return json.dumps(messages, ensure_ascii=False)

    def _write_temp_prompt(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".json", prefix="llama_chat_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        return path

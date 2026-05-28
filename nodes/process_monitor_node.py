import json

from ..process_manager import LlamaProcessManager


class LlamaProcessMonitorNode:
    """View and manage keep-alive llama-cli processes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["status", "release_all", "release_by_model"], {
                    "default": "status",
                    "tooltip": "Action: view status, release all, or release by model name",
                }),
            },
            "optional": {
                "model_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Model name filter (for release_by_model action)",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("info", "released")
    FUNCTION = "monitor"
    CATEGORY = "AI/LlamaCpp"
    OUTPUT_NODE = True

    def monitor(self, action: str, **kwargs):
        mgr = LlamaProcessManager.get_instance()

        if action == "status":
            status = mgr.get_status()
            info = json.dumps(status, indent=2, ensure_ascii=False)
            return (info, False)

        elif action == "release_all":
            mgr.shutdown_all()
            return ('{"action": "release_all", "result": "all processes terminated"}', True)

        elif action == "release_by_model":
            model_filter = kwargs.get("model_filter", "").strip()
            if not model_filter:
                return ('{"error": "model_filter is required for release_by_model"}', False)
            count = mgr.release_by_model(model_filter)
            result = {"action": "release_by_model", "model": model_filter, "released_count": count}
            return (json.dumps(result, ensure_ascii=False), count > 0)

        return ('{"error": "unknown action"}', False)

import os

try:
    import folder_paths
except ImportError:
    folder_paths = None

CACHE_TYPE_OPTIONS = ["f16", "f32", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
SPLIT_MODE_OPTIONS = ["none", "layer", "row"]
SPEC_TYPE_OPTIONS = [
    "none", "draft-simple", "draft-eagle3", "draft-mtp",
    "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache",
]


def _get_gguf_list():
    if folder_paths:
        try:
            return folder_paths.get_filename_list("llama_gguf")
        except Exception:
            pass
    return []


def _get_mmproj_list():
    if folder_paths:
        try:
            return ["none"] + folder_paths.get_filename_list("llama_mmproj")
        except Exception:
            pass
    return ["none"]


class LlamaCliConfigNode:

    @classmethod
    def INPUT_TYPES(cls):
        gguf_list = _get_gguf_list()
        mmproj_list = _get_mmproj_list()
        draft_list = ["none"] + list(gguf_list)

        return {
            "required": {
                "binary_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory containing llama.cpp binaries (llama-cli, llama-embedding, etc.)",
                }),
                "model_name": (gguf_list if gguf_list else ["(no models found)"], {
                    "tooltip": "GGUF model file from ComfyUI/models/llama_gguf/",
                }),
            },
            "optional": {
                "n_gpu_layers": ("INT", {
                    "default": 99, "min": -1, "max": 9999,
                    "tooltip": "-ngl: Number of layers to offload to GPU (99 = all)",
                }),
                "context_size": ("INT", {
                    "default": 4096, "min": 128, "max": 1048576,
                    "tooltip": "-c: Context window size",
                }),
                "flash_attention": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "-fa: Enable Flash Attention",
                }),
                "split_mode": (SPLIT_MODE_OPTIONS, {
                    "default": "none",
                    "tooltip": "--split-mode: GPU split mode",
                }),
                "main_gpu": ("INT", {
                    "default": 0, "min": 0, "max": 15,
                    "tooltip": "--main-gpu: Primary GPU index",
                }),
                "threads": ("INT", {
                    "default": 0, "min": 0, "max": 256,
                    "tooltip": "-t: CPU threads (0 = auto)",
                }),
                "n_parallel": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "-np: Number of parallel sequences",
                }),
                "cache_type_k": (CACHE_TYPE_OPTIONS, {
                    "default": "f16",
                    "tooltip": "--cache-type-k: KV cache data type for K",
                }),
                "cache_type_v": (CACHE_TYPE_OPTIONS, {
                    "default": "f16",
                    "tooltip": "--cache-type-v: KV cache data type for V",
                }),
                "jinja": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "--jinja: Enable Jinja chat template engine",
                }),
                "mmproj_name": (mmproj_list, {
                    "default": "none",
                    "tooltip": "--mmproj: Multimodal projector GGUF from ComfyUI/models/llama_mmproj/",
                }),
                "mmproj_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "--mmproj-offload: GPU offloading for multimodal projector",
                }),
                "spec_type": (SPEC_TYPE_OPTIONS, {
                    "default": "none",
                    "tooltip": "--spec-type: Speculative decoding type",
                }),
                "spec_draft_n_max": ("INT", {
                    "default": 0, "min": 0, "max": 64,
                    "tooltip": "--spec-draft-n-max: Max speculative draft tokens (0 = disabled)",
                }),
                "spec_draft_model": (draft_list if draft_list else ["none"], {
                    "default": "none",
                    "tooltip": "--model-draft: Draft model for speculative decoding",
                }),
                "keep_alive": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep llama-cli process alive between runs (reuse VRAM)",
                }),
                "timeout": ("INT", {
                    "default": 600, "min": 10, "max": 7200,
                    "tooltip": "Execution timeout in seconds",
                }),
                "extra_args": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Additional CLI arguments (e.g. --tensor-split 1.6,1 --mlock)",
                }),
            },
        }

    RETURN_TYPES = ("CLI_CONFIG",)
    RETURN_NAMES = ("cli_config",)
    FUNCTION = "build_config"
    CATEGORY = "AI/LlamaCpp"

    def build_config(self, binary_dir: str, model_name: str, **kwargs):
        if not binary_dir or not os.path.isdir(binary_dir):
            raise ValueError(f"Invalid binary directory: {binary_dir}")

        if folder_paths:
            model_path = folder_paths.get_full_path_or_raise("llama_gguf", model_name)
        else:
            model_path = model_name

        mmproj_name = kwargs.get("mmproj_name", "none")
        mmproj_path = None
        if mmproj_name and mmproj_name != "none" and folder_paths:
            mmproj_path = folder_paths.get_full_path_or_raise("llama_mmproj", mmproj_name)

        spec_draft_model = kwargs.get("spec_draft_model", "none")
        spec_draft_model_path = None
        if spec_draft_model and spec_draft_model != "none" and folder_paths:
            spec_draft_model_path = folder_paths.get_full_path_or_raise("llama_gguf", spec_draft_model)

        config = {
            "binary_dir": binary_dir,
            "model_path": model_path,
            "model_name": model_name,
            "n_gpu_layers": kwargs.get("n_gpu_layers", 99),
            "context_size": kwargs.get("context_size", 4096),
            "flash_attention": kwargs.get("flash_attention", True),
            "split_mode": kwargs.get("split_mode", "none"),
            "main_gpu": kwargs.get("main_gpu", 0),
            "threads": kwargs.get("threads", 0),
            "n_parallel": kwargs.get("n_parallel", 1),
            "cache_type_k": kwargs.get("cache_type_k", "f16"),
            "cache_type_v": kwargs.get("cache_type_v", "f16"),
            "jinja": kwargs.get("jinja", True),
            "mmproj_path": mmproj_path,
            "mmproj_offload": kwargs.get("mmproj_offload", True),
            "spec_type": kwargs.get("spec_type", "none"),
            "spec_draft_n_max": kwargs.get("spec_draft_n_max", 0),
            "spec_draft_model_path": spec_draft_model_path,
            "keep_alive": kwargs.get("keep_alive", False),
            "timeout": kwargs.get("timeout", 600),
            "extra_args": kwargs.get("extra_args", ""),
        }

        return (config,)

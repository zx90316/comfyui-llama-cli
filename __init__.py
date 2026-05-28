import os

try:
    import folder_paths

    folder_paths.add_model_folder_path(
        "llama_gguf",
        os.path.join(folder_paths.models_dir, "llama_gguf"),
    )
    folder_paths.add_model_folder_path(
        "llama_mmproj",
        os.path.join(folder_paths.models_dir, "llama_mmproj"),
    )
except ImportError:
    pass

from .nodes.cli_config_node import LlamaCliConfigNode
from .nodes.generate_node import LlamaGenerateNode
from .nodes.chat_node import LlamaChatNode
from .nodes.embedding_node import LlamaEmbeddingNode
from .nodes.infill_node import LlamaInfillNode
from .nodes.image_encoder_node import LlamaImageEncoderNode
from .nodes.process_monitor_node import LlamaProcessMonitorNode

NODE_CLASS_MAPPINGS = {
    "LlamaCliConfig": LlamaCliConfigNode,
    "LlamaGenerate": LlamaGenerateNode,
    "LlamaChat": LlamaChatNode,
    "LlamaEmbedding": LlamaEmbeddingNode,
    "LlamaInfill": LlamaInfillNode,
    "LlamaImageEncoder": LlamaImageEncoderNode,
    "LlamaProcessMonitor": LlamaProcessMonitorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCliConfig": "Llama CLI Config",
    "LlamaGenerate": "Llama Generate",
    "LlamaChat": "Llama Chat",
    "LlamaEmbedding": "Llama Embedding",
    "LlamaInfill": "Llama Infill",
    "LlamaImageEncoder": "Llama Image Encoder",
    "LlamaProcessMonitor": "Llama Process Monitor",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

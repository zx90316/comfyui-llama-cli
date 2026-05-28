# ComfyUI Llama.cpp CLI Node

ComfyUI custom nodes that directly control **llama.cpp** command-line binaries (`llama-cli`, `llama-embedding`) via subprocess. No server required — the node spawns the process, captures output, and optionally manages VRAM lifecycle.

## Features

- **Direct CLI control** — No HTTP server; calls `llama-cli` / `llama-embedding` as subprocesses
- **Two VRAM modes** — One-shot (auto-release after each run) or Keep-alive (process stays resident)
- **7 specialized nodes** — Config, Generate, Chat, Embedding, Infill, Image Encoder, Process Monitor
- **GGUF model dropdown** — Integrated with ComfyUI's model manager
- **Full sampling parameters** — Temperature, Top-K/P, Min-P, Mirostat, DRY, Dynamic Temperature, etc.
- **Structured output** — BNF grammar and JSON Schema constraints
- **Multimodal vision** — mmproj projector + image input via `LlamaImageEncoder`
- **Speculative decoding** — draft-mtp, draft-eagle3, ngram variants
- **KV cache optimization** — Configurable cache-type-k/v (q8_0, q4_0, etc.)
- **Process monitoring** — View and release keep-alive processes

## Requirements

- [llama.cpp](https://github.com/ggml-org/llama.cpp) compiled binaries (`llama-cli`, `llama-embedding`)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) >= 1.0.0
- Python packages: `pillow`, `numpy` (typically already installed with ComfyUI)

## Installation

1. Clone or copy this repository into `ComfyUI/custom_nodes/`:

```
cd ComfyUI/custom_nodes/
git clone <this-repo-url> comfyui-llama-cli
```

2. Create model directories:

```
mkdir ComfyUI/models/llama_gguf
mkdir ComfyUI/models/llama_mmproj
```

3. Place your `.gguf` model files in `ComfyUI/models/llama_gguf/`
4. (Optional) Place mmproj files in `ComfyUI/models/llama_mmproj/`

## Nodes

### Llama CLI Config

Central configuration node. Set binary directory, model, GPU settings, and all launch parameters.

| Parameter | CLI Flag | Description |
|-----------|----------|-------------|
| binary_dir | — | Directory containing llama.cpp executables |
| model_name | -m | GGUF model (dropdown) |
| n_gpu_layers | -ngl | GPU layers (99 = all) |
| context_size | -c | Context window size |
| flash_attention | -fa | Flash Attention |
| split_mode | --split-mode | GPU split: none/layer/row |
| main_gpu | --main-gpu | Primary GPU index |
| threads | -t | CPU threads (0 = auto) |
| n_parallel | -np | Parallel sequences |
| cache_type_k | --cache-type-k | K cache type (f16/q8_0/q4_0/...) |
| cache_type_v | --cache-type-v | V cache type |
| jinja | --jinja | Jinja template engine |
| mmproj_name | --mmproj | Vision projector (dropdown) |
| spec_type | --spec-type | Speculative decoding type |
| spec_draft_n_max | --spec-draft-n-max | Max draft tokens |
| spec_draft_model | --model-draft | Draft model |
| keep_alive | — | Keep process alive between runs |
| timeout | — | Execution timeout (seconds) |
| extra_args | — | Additional CLI arguments |

### Llama Generate

Text completion via `llama-cli -p "prompt"`.

- **Input:** CLI_CONFIG + prompt + sampling parameters
- **Output:** generated text, stderr log, exit code

### Llama Chat

Multi-turn conversation via `llama-cli -cnv --jinja`.

- **Input:** CLI_CONFIG + user message + system message + chat history
- **Output:** response text, updated history JSON, stderr log, exit code

### Llama Embedding

Text embeddings via `llama-embedding`.

- **Input:** CLI_CONFIG + input text
- **Output:** embedding JSON array, stderr log, exit code

### Llama Infill

Code fill-in-the-middle via `llama-cli --infill`.

- **Input:** CLI_CONFIG + prefix + suffix
- **Output:** infilled text, stderr log, exit code

### Llama Image Encoder

Converts ComfyUI IMAGE tensor to a temp image file for `--image` flag.

- **Input:** IMAGE tensor + format (png/jpg)
- **Output:** image file path (connect to Generate/Chat `image_path`)

### Llama Process Monitor

View and manage keep-alive processes.

- **Actions:** status, release_all, release_by_model
- **Output:** process info JSON, released status

## VRAM Management

### One-shot Mode (default: `keep_alive=False`)

Each node execution spawns a new `llama-cli` process. When generation completes, the process exits and VRAM is automatically freed.

### Keep-alive Mode (`keep_alive=True`)

The first execution starts `llama-cli` in interactive mode. Subsequent executions reuse the same process (model stays loaded in VRAM). Use the **Llama Process Monitor** node to manually release processes.

## Example Workflow

```
[Llama CLI Config] → CLI_CONFIG → [Llama Chat] → response text
                                         ↑
[Llama Image Encoder] → image_path ──────┘
```

### Example Config

Equivalent to:
```bash
llama-cli -m model.gguf --jinja -ngl 99 -c 80000 -fa \
    -np 1 --split-mode none --main-gpu 0 \
    --cache-type-k q8_0 --spec-type draft-mtp --spec-draft-n-max 6
```

Set in the Config node:
- `n_gpu_layers` = 99
- `context_size` = 80000
- `flash_attention` = True
- `n_parallel` = 1
- `split_mode` = none
- `main_gpu` = 0
- `cache_type_k` = q8_0
- `spec_type` = draft-mtp
- `spec_draft_n_max` = 6

## Model Directories

| Directory | Purpose |
|-----------|---------|
| `ComfyUI/models/llama_gguf/` | Main GGUF model files |
| `ComfyUI/models/llama_mmproj/` | Vision projector GGUF files |

## License

MIT

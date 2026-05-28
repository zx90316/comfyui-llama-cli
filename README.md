# ComfyUI Llama.cpp CLI Node

[繁體中文](README_zh-TW.md)

ComfyUI custom nodes that directly control **llama.cpp** command-line binaries via subprocess. No server required — the node spawns the process, captures output, and optionally manages VRAM lifecycle.

## Architecture

```
ComfyUI Node  ──→  subprocess  ──→  llama.cpp binary  ──→  parsed output
                    (one-shot / keep-alive)
```

| Task | Binary | Mode |
|------|--------|------|
| Text completion | `llama-completion` | `-no-cnv` |
| Chat / Conversation | `llama-cli` | `-cnv --single-turn --jinja` |
| Vision (multimodal) | `llama-cli` | `-cnv --single-turn --mmproj` |
| Embedding | `llama-embedding` | — |
| Code infill | `llama-completion` | `--infill` |

## Features

- **Direct CLI control** — No HTTP server; calls llama.cpp binaries as subprocesses
- **Two VRAM modes** — One-shot (auto-release) or Keep-alive (process stays resident)
- **7 specialized nodes** — Config, Generate, Chat, Embedding, Infill, Image Encoder, Process Monitor
- **GGUF model dropdown** — Integrated with ComfyUI's model manager (`folder_paths`)
- **Full sampling parameters** — Temperature, Top-K/P, Min-P, Mirostat, DRY, Dynamic Temperature, etc.
- **Structured output** — BNF grammar and JSON Schema constraints
- **Multimodal vision** — mmproj projector + image input via `LlamaImageEncoder`
- **Speculative decoding** — draft-mtp, draft-eagle3, ngram variants
- **KV cache optimization** — Configurable cache-type-k/v (f16, q8_0, q4_0, etc.)
- **Process monitoring** — View and release keep-alive processes

## Requirements

- [llama.cpp](https://github.com/ggml-org/llama.cpp) compiled binaries (`llama-cli`, `llama-completion`, `llama-embedding`)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) >= 1.0.0
- Python packages: `pillow`, `numpy` (typically already installed with ComfyUI)

## Installation

1. Clone into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/zx90316/comfyui-llama-cli.git
```

2. Create model directories:

```bash
mkdir ComfyUI/models/llama_gguf
mkdir ComfyUI/models/llama_mmproj
```

3. Place `.gguf` model files in `ComfyUI/models/llama_gguf/`
4. (Optional) Place mmproj vision projector files in `ComfyUI/models/llama_mmproj/`

## Nodes

### Llama CLI Config

Central configuration node. All other nodes receive a `CLI_CONFIG` dict from this node.

| Parameter | CLI Flag | Description |
|-----------|----------|-------------|
| binary_dir | — | Directory containing llama.cpp executables |
| model_name | `-m` | GGUF model (dropdown from `llama_gguf`) |
| n_gpu_layers | `-ngl` | GPU layers to offload (99 = all) |
| context_size | `-c` | Context window size |
| flash_attention | `-fa` | Flash Attention (on/off) |
| split_mode | `--split-mode` | GPU split strategy: none / layer / row |
| main_gpu | `--main-gpu` | Primary GPU index |
| threads | `-t` | CPU threads (0 = auto) |
| n_parallel | `-np` | Parallel sequences |
| cache_type_k | `--cache-type-k` | K cache quantization (f16 / q8_0 / q4_0 / ...) |
| cache_type_v | `--cache-type-v` | V cache quantization |
| jinja | `--jinja` | Jinja chat template engine (chat mode) |
| mmproj_name | `--mmproj` | Vision projector file (dropdown from `llama_mmproj`) |
| mmproj_offload | `--no-mmproj-offload` | Whether to offload mmproj to GPU |
| spec_type | `--spec-type` | Speculative decoding type |
| spec_draft_n_max | `--spec-draft-n-max` | Max draft tokens |
| spec_draft_model | `--model-draft` | Draft model for speculative decoding |
| keep_alive | — | Keep process alive between runs |
| timeout | — | Execution timeout in seconds |
| extra_args | — | Additional raw CLI arguments |

### Llama Generate

Text completion via `llama-completion`. Automatically switches to `llama-cli -cnv` when vision input (image + mmproj) is detected.

- **Input:** `CLI_CONFIG` + prompt + sampling parameters + optional `image_path`
- **Output:** generated text, stderr log, exit code

### Llama Chat

Multi-turn conversation via `llama-cli -cnv --single-turn --jinja`.

- **Input:** `CLI_CONFIG` + user message + system message + chat history JSON
- **Output:** response text, updated history JSON, stderr log, exit code

### Llama Embedding

Text embeddings via `llama-embedding`.

- **Input:** `CLI_CONFIG` + input text
- **Output:** embedding JSON array, stderr log, exit code

### Llama Infill

Code fill-in-the-middle via `llama-completion --infill`.

- **Input:** `CLI_CONFIG` + prefix + suffix
- **Output:** infilled text, stderr log, exit code

### Llama Image Encoder

Converts ComfyUI `IMAGE` tensor to a temporary image file for the `--image` flag.

- **Input:** IMAGE tensor + format (png / jpg)
- **Output:** image file path (connect to Generate or Chat `image_path`)

### Llama Process Monitor

View and manage keep-alive processes.

- **Actions:** `status`, `release_all`, `release_by_model`
- **Output:** process info JSON, released count

## VRAM Management

### One-shot Mode (default)

`keep_alive = False` — Each node execution spawns a new process. When generation completes, the process exits and VRAM is freed automatically.

### Keep-alive Mode

`keep_alive = True` — The first execution starts the binary in interactive mode. Subsequent runs reuse the same process (model stays loaded in VRAM). Use **Llama Process Monitor** to manually release processes.

## Example Workflow

```
[Llama CLI Config] ──→ CLI_CONFIG ──→ [Llama Chat] ──→ response text
                                            ↑
[Llama Image Encoder] ──→ image_path ───────┘
```

### Equivalent CLI Command

The Config node with these settings:

| Setting | Value |
|---------|-------|
| n_gpu_layers | 99 |
| context_size | 80000 |
| flash_attention | True |
| split_mode | none |
| main_gpu | 0 |
| cache_type_k | q8_0 |
| spec_type | draft-mtp |
| spec_draft_n_max | 6 |

Produces a command equivalent to:

```bash
llama-cli -m model.gguf --jinja -ngl 99 -c 80000 -fa on \
    --split-mode none --main-gpu 0 \
    --cache-type-k q8_0 --spec-type draft-mtp --spec-draft-n-max 6
```

## Model Directories

| Directory | Purpose |
|-----------|---------|
| `ComfyUI/models/llama_gguf/` | Main GGUF model files |
| `ComfyUI/models/llama_mmproj/` | Vision projector (mmproj) GGUF files |

## License

MIT

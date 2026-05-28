# ComfyUI Llama.cpp CLI 節點

[English](README.md)

ComfyUI 自訂節點，透過子進程直接控制 **llama.cpp** 命令列工具。不需要啟動伺服器 — 節點直接呼叫二進位程式、擷取輸出，並可選擇性管理 VRAM 生命週期。

## 架構

```
ComfyUI 節點  ──→  subprocess  ──→  llama.cpp 二進位  ──→  解析後輸出
                   (一次性 / 常駐)
```

| 任務 | 二進位程式 | 模式 |
|------|-----------|------|
| 文字補全 | `llama-completion` | `-no-cnv` |
| 對話 | `llama-cli` | `-cnv --single-turn --jinja` |
| 視覺（多模態） | `llama-cli` | `-cnv --single-turn --mmproj` |
| 嵌入向量 | `llama-embedding` | — |
| 程式碼填充 | `llama-completion` | `--infill` |

## 功能特色

- **直接 CLI 控制** — 不需要 HTTP 伺服器，直接以子進程呼叫 llama.cpp
- **兩種 VRAM 模式** — 一次性（自動釋放）或常駐（進程保持載入）
- **7 個專用節點** — Config、Generate、Chat、Embedding、Infill、Image Encoder、Process Monitor
- **GGUF 模型下拉選單** — 整合 ComfyUI 的模型管理器（`folder_paths`）
- **完整取樣參數** — Temperature、Top-K/P、Min-P、Mirostat、DRY、Dynamic Temperature 等
- **結構化輸出** — BNF 文法與 JSON Schema 約束
- **多模態視覺** — mmproj 投影器 + 圖片輸入（透過 `LlamaImageEncoder`）
- **推測解碼** — draft-mtp、draft-eagle3、ngram 等變體
- **KV 快取最佳化** — 可設定 cache-type-k/v（f16、q8_0、q4_0 等）
- **進程監控** — 檢視並釋放常駐進程

## 系統需求

- [llama.cpp](https://github.com/ggml-org/llama.cpp) 編譯後的二進位程式（`llama-cli`、`llama-completion`、`llama-embedding`）
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) >= 1.0.0
- Python 套件：`pillow`、`numpy`（ComfyUI 通常已安裝）

## 安裝

1. 複製到 `ComfyUI/custom_nodes/`：

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/zx90316/comfyui-llama-cli.git
```

2. 建立模型目錄：

```bash
mkdir ComfyUI/models/llama_gguf
mkdir ComfyUI/models/llama_mmproj
```

3. 將 `.gguf` 模型檔放入 `ComfyUI/models/llama_gguf/`
4. （選用）將 mmproj 視覺投影器檔案放入 `ComfyUI/models/llama_mmproj/`

## 節點說明

### Llama CLI Config（設定節點）

中央設定節點。所有其他節點透過 `CLI_CONFIG` 字典接收此節點的設定。

| 參數 | CLI 旗標 | 說明 |
|------|---------|------|
| binary_dir | — | llama.cpp 執行檔所在目錄 |
| model_name | `-m` | GGUF 模型（下拉選單，來自 `llama_gguf`） |
| n_gpu_layers | `-ngl` | 卸載至 GPU 的層數（99 = 全部） |
| context_size | `-c` | 上下文視窗大小 |
| flash_attention | `-fa` | Flash Attention（開/關） |
| split_mode | `--split-mode` | GPU 分割策略：none / layer / row |
| main_gpu | `--main-gpu` | 主要 GPU 索引 |
| threads | `-t` | CPU 執行緒數（0 = 自動） |
| n_parallel | `-np` | 平行序列數 |
| cache_type_k | `--cache-type-k` | K 快取量化類型（f16 / q8_0 / q4_0 / ...） |
| cache_type_v | `--cache-type-v` | V 快取量化類型 |
| jinja | `--jinja` | Jinja 聊天模板引擎（對話模式） |
| mmproj_name | `--mmproj` | 視覺投影器檔案（下拉選單，來自 `llama_mmproj`） |
| mmproj_offload | `--no-mmproj-offload` | 是否將 mmproj 卸載至 GPU |
| spec_type | `--spec-type` | 推測解碼類型 |
| spec_draft_n_max | `--spec-draft-n-max` | 最大草稿 token 數 |
| spec_draft_model | `--model-draft` | 推測解碼用的草稿模型 |
| keep_alive | — | 執行間保持進程常駐 |
| timeout | — | 執行逾時（秒） |
| extra_args | — | 額外的原始 CLI 參數 |

### Llama Generate（文字生成）

透過 `llama-completion` 進行文字補全。偵測到視覺輸入（圖片 + mmproj）時，自動切換至 `llama-cli -cnv`。

- **輸入：** `CLI_CONFIG` + 提示詞 + 取樣參數 + 選用 `image_path`
- **輸出：** 生成文字、stderr 日誌、退出碼

### Llama Chat（對話）

透過 `llama-cli -cnv --single-turn --jinja` 進行多輪對話。

- **輸入：** `CLI_CONFIG` + 使用者訊息 + 系統訊息 + 對話歷史 JSON
- **輸出：** 回應文字、更新後的歷史 JSON、stderr 日誌、退出碼

### Llama Embedding（嵌入向量）

透過 `llama-embedding` 生成文字嵌入向量。

- **輸入：** `CLI_CONFIG` + 輸入文字
- **輸出：** 嵌入向量 JSON 陣列、stderr 日誌、退出碼

### Llama Infill（程式碼填充）

透過 `llama-completion --infill` 進行程式碼中間填充（Fill-in-the-Middle）。

- **輸入：** `CLI_CONFIG` + 前綴 + 後綴
- **輸出：** 填充後的文字、stderr 日誌、退出碼

### Llama Image Encoder（圖片編碼器）

將 ComfyUI 的 `IMAGE` 張量轉換為暫存圖片檔案，供 `--image` 旗標使用。

- **輸入：** IMAGE 張量 + 格式（png / jpg）
- **輸出：** 圖片檔案路徑（連接至 Generate 或 Chat 的 `image_path`）

### Llama Process Monitor（進程監控）

檢視與管理常駐進程。

- **動作：** `status`（狀態）、`release_all`（全部釋放）、`release_by_model`（依模型釋放）
- **輸出：** 進程資訊 JSON、已釋放數量

## VRAM 管理

### 一次性模式（預設）

`keep_alive = False` — 每次節點執行時產生新的子進程。生成完成後進程結束，VRAM 自動釋放。

### 常駐模式

`keep_alive = True` — 首次執行時以互動模式啟動二進位程式。後續執行重複使用同一進程（模型保持載入於 VRAM）。使用 **Llama Process Monitor** 手動釋放進程。

## 工作流程範例

```
[Llama CLI Config] ──→ CLI_CONFIG ──→ [Llama Chat] ──→ 回應文字
                                            ↑
[Llama Image Encoder] ──→ image_path ───────┘
```

### 對應的 CLI 命令

Config 節點使用以下設定：

| 設定 | 值 |
|------|-----|
| n_gpu_layers | 99 |
| context_size | 80000 |
| flash_attention | True |
| split_mode | none |
| main_gpu | 0 |
| cache_type_k | q8_0 |
| spec_type | draft-mtp |
| spec_draft_n_max | 6 |

產生的命令等同於：

```bash
llama-cli -m model.gguf --jinja -ngl 99 -c 80000 -fa on \
    --split-mode none --main-gpu 0 \
    --cache-type-k q8_0 --spec-type draft-mtp --spec-draft-n-max 6
```

## 模型目錄

| 目錄 | 用途 |
|------|------|
| `ComfyUI/models/llama_gguf/` | 主要 GGUF 模型檔案 |
| `ComfyUI/models/llama_mmproj/` | 視覺投影器（mmproj）GGUF 檔案 |

## 授權

MIT

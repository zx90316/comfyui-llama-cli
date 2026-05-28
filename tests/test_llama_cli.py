# Comprehensive pytest suite for comfyui-llama-cli.
#
# Usage:
#   pytest tests/test_llama_cli.py -v                  # all tests
#   pytest tests/test_llama_cli.py -v -m unit          # unit tests only (fast, no GPU)
#   pytest tests/test_llama_cli.py -v -m integration   # integration tests (needs GPU + model)

import os
import sys
import json
import tempfile

import pytest
import numpy as np
from PIL import Image

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from comfyui_llama_cli.process_manager import LlamaProcessManager, InteractiveProcess
from comfyui_llama_cli.utils.command_builder import (
    build_base_cmd, add_sampling_args, add_generation_args,
)
from comfyui_llama_cli.utils.output_parser import (
    parse_generation_output,
    parse_chat_response,
    parse_embedding_output,
    parse_timing_stats,
)
from comfyui_llama_cli.utils.param_helpers import (
    validate_binary_dir,
    resolve_binary_path,
    parse_json_field,
    write_temp_grammar,
)
from comfyui_llama_cli.nodes.cli_config_node import LlamaCliConfigNode
from comfyui_llama_cli.nodes.generate_node import LlamaGenerateNode
from comfyui_llama_cli.nodes.chat_node import LlamaChatNode
from comfyui_llama_cli.nodes.embedding_node import LlamaEmbeddingNode
from comfyui_llama_cli.nodes.infill_node import LlamaInfillNode
from comfyui_llama_cli.nodes.process_monitor_node import LlamaProcessMonitorNode
from comfyui_llama_cli.nodes.image_encoder_node import LlamaImageEncoderNode

# ---------------------------------------------------------------------------
# Paths — small Gemma 4B model for fast tests
# ---------------------------------------------------------------------------

BINARY_DIR = r"C:\Users\zx020\project\llama-cpp-turboquant\build\bin\Release"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models",
    "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf",
)
MMPROJ_PATH = os.path.join(
    PROJECT_ROOT, "models",
    "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf",
)
MODEL_NAME = "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config():
    """Minimal CLI_CONFIG dict for unit-testing command builders."""
    return {
        "binary_dir": BINARY_DIR,
        "model_path": MODEL_PATH,
        "model_name": MODEL_NAME,
        "n_gpu_layers": 99,
        "context_size": 512,
        "flash_attention": True,
        "split_mode": "none",
        "main_gpu": 0,
        "threads": 0,
        "n_parallel": 1,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "jinja": True,
        "mmproj_path": None,
        "mmproj_offload": True,
        "spec_type": "none",
        "spec_draft_n_max": 0,
        "spec_draft_model_path": None,
        "keep_alive": False,
        "timeout": 120,
        "extra_args": "",
    }


@pytest.fixture
def integration_config(base_config):
    """Config tuned for fast integration tests with the small 4B model."""
    base_config["context_size"] = 512
    base_config["timeout"] = 120
    return base_config


@pytest.fixture
def vision_config(integration_config):
    """Config with mmproj for vision/multimodal tests."""
    integration_config["mmproj_path"] = MMPROJ_PATH
    return integration_config


@pytest.fixture
def test_image_path():
    """Create a small test image and return its path. Cleaned up after test."""
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    fd, path = tempfile.mkstemp(suffix=".png", prefix="test_img_")
    os.close(fd)
    img.save(path, "PNG")
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def dummy_image_tensor():
    """Create a fake ComfyUI-style IMAGE tensor (batch, H, W, C) as float32 [0,1]."""
    img = np.zeros((1, 64, 64, 3), dtype=np.float32)
    img[0, :32, :, 0] = 1.0  # top half red
    img[0, 32:, :, 2] = 1.0  # bottom half blue

    class FakeTensor:
        """Minimal tensor-like object with .cpu().numpy() support."""
        def __init__(self, arr):
            self._arr = arr
        def cpu(self):
            return self
        def numpy(self):
            return self._arr
        def __getitem__(self, idx):
            return FakeTensor(self._arr[idx])

    return FakeTensor(img)


# ===========================================================================
# UNIT TESTS — output_parser
# ===========================================================================

class TestOutputParser:

    @pytest.mark.unit
    def test_parse_generation_basic(self):
        assert parse_generation_output("Hello world") == "Hello world"

    @pytest.mark.unit
    def test_parse_generation_strips_whitespace(self):
        assert parse_generation_output("  Hello world  \n") == "Hello world"

    @pytest.mark.unit
    def test_parse_generation_strips_eot_im_end(self):
        assert parse_generation_output("Hello<|im_end|>") == "Hello"

    @pytest.mark.unit
    def test_parse_generation_strips_eot_eos(self):
        assert parse_generation_output("Result</s>") == "Result"

    @pytest.mark.unit
    def test_parse_generation_strips_eot_endoftext(self):
        assert parse_generation_output("Text<|endoftext|>") == "Text"

    @pytest.mark.unit
    def test_parse_generation_strips_eot_id(self):
        assert parse_generation_output("Answer<|eot_id|>") == "Answer"

    @pytest.mark.unit
    def test_parse_generation_empty(self):
        assert parse_generation_output("") == ""

    @pytest.mark.unit
    def test_parse_chat_response_basic(self):
        assert parse_chat_response("  Hi there!  ") == "Hi there!"

    @pytest.mark.unit
    def test_parse_chat_response_strips_eot(self):
        assert parse_chat_response("Response<|im_end|>") == "Response"

    @pytest.mark.unit
    def test_parse_chat_response_strips_llama_cli_noise(self):
        raw = (
            "Loading model... \n\n"
            "build      : b9418\nmodel      : test.gguf\n\n"
            "> [{\"role\": \"user\", \"content\": \"Hi\"}]\n\n"
            "Hello!\n\n"
            "[ Prompt: 100.0 t/s | Generation: 50.0 t/s ]\n\n"
            "Exiting...\n\x1b[0m"
        )
        assert parse_chat_response(raw) == "Hello!"

    @pytest.mark.unit
    def test_parse_embedding_output_floats(self):
        result = parse_embedding_output("0.1 0.2 0.3")
        parsed = json.loads(result)
        assert parsed == [0.1, 0.2, 0.3]

    @pytest.mark.unit
    def test_parse_embedding_output_empty(self):
        assert parse_embedding_output("") == "[]"

    @pytest.mark.unit
    def test_parse_embedding_output_non_numeric(self):
        result = parse_embedding_output("not a number")
        assert result == "not a number"

    @pytest.mark.unit
    def test_parse_timing_stats_full(self):
        stderr = (
            "llama_perf_context_print:        load time =   1234.56 ms\n"
            "llama_perf_context_print: prompt eval time =   500.00 ms /    10 tokens (   50.00 ms per token,    20.00 tokens per second)\n"
            "llama_perf_context_print:        eval time =  2000.00 ms /    50 tokens (   40.00 ms per token,    25.00 tokens per second)\n"
        )
        stats = parse_timing_stats(stderr)
        assert stats is not None
        assert stats["load_time_ms"] == 1234.56
        assert stats["prompt_eval_tokens"] == 10
        assert stats["prompt_eval_tokens_per_second"] == 20.0
        assert stats["eval_tokens"] == 50
        assert stats["eval_tokens_per_second"] == 25.0

    @pytest.mark.unit
    def test_parse_timing_stats_none(self):
        assert parse_timing_stats("random log output") is None


# ===========================================================================
# UNIT TESTS — param_helpers
# ===========================================================================

class TestParamHelpers:

    @pytest.mark.unit
    def test_validate_binary_dir_valid(self):
        ok, found, err = validate_binary_dir(BINARY_DIR)
        assert ok is True
        assert "llama-cli" in found
        assert err == ""

    @pytest.mark.unit
    def test_validate_binary_dir_invalid(self):
        ok, found, err = validate_binary_dir(r"C:\nonexistent\path")
        assert ok is False
        assert found == []

    @pytest.mark.unit
    def test_validate_binary_dir_empty(self):
        ok, found, err = validate_binary_dir("")
        assert ok is False

    @pytest.mark.unit
    def test_resolve_binary_path(self):
        result = resolve_binary_path(BINARY_DIR, "llama-cli")
        assert result.endswith("llama-cli.exe")
        assert BINARY_DIR in result

    @pytest.mark.unit
    def test_parse_json_field_valid(self):
        assert parse_json_field('["a", "b"]') == ["a", "b"]

    @pytest.mark.unit
    def test_parse_json_field_invalid(self):
        assert parse_json_field("not json", default=[]) == []

    @pytest.mark.unit
    def test_parse_json_field_empty(self):
        assert parse_json_field("", default=None) is None

    @pytest.mark.unit
    def test_parse_json_field_dict(self):
        assert parse_json_field('{"k": 1}') == {"k": 1}

    @pytest.mark.unit
    def test_write_temp_grammar(self):
        grammar = 'root ::= "hello"'
        path = write_temp_grammar(grammar)
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as f:
            assert f.read() == grammar
        os.unlink(path)


# ===========================================================================
# UNIT TESTS — command_builder
# ===========================================================================

class TestCommandBuilder:

    @pytest.mark.unit
    def test_build_base_cmd_minimal(self, base_config):
        cmd = build_base_cmd(base_config)
        assert cmd[0].endswith("llama-cli.exe")
        assert "-m" in cmd
        assert base_config["model_path"] in cmd
        assert "-ngl" in cmd
        assert "99" in cmd
        assert "-c" in cmd
        assert "512" in cmd
        idx_fa = cmd.index("-fa")
        assert cmd[idx_fa + 1] == "on"
        assert "--split-mode" in cmd
        assert "--main-gpu" in cmd

    @pytest.mark.unit
    def test_build_base_cmd_custom_binary(self, base_config):
        cmd = build_base_cmd(base_config, binary_name="llama-embedding")
        assert cmd[0].endswith("llama-embedding.exe")

    @pytest.mark.unit
    def test_build_base_cmd_no_flash_attention(self, base_config):
        base_config["flash_attention"] = False
        cmd = build_base_cmd(base_config)
        idx_fa = cmd.index("-fa")
        assert cmd[idx_fa + 1] == "off"

    @pytest.mark.unit
    def test_build_base_cmd_cache_type(self, base_config):
        base_config["cache_type_k"] = "q8_0"
        base_config["cache_type_v"] = "q4_0"
        cmd = build_base_cmd(base_config)
        idx_k = cmd.index("--cache-type-k")
        assert cmd[idx_k + 1] == "q8_0"
        idx_v = cmd.index("--cache-type-v")
        assert cmd[idx_v + 1] == "q4_0"

    @pytest.mark.unit
    def test_build_base_cmd_cache_type_default_omitted(self, base_config):
        cmd = build_base_cmd(base_config)
        assert "--cache-type-k" not in cmd
        assert "--cache-type-v" not in cmd

    @pytest.mark.unit
    def test_build_base_cmd_spec_type(self, base_config):
        base_config["spec_type"] = "draft-mtp"
        base_config["spec_draft_n_max"] = 6
        cmd = build_base_cmd(base_config)
        idx = cmd.index("--spec-type")
        assert cmd[idx + 1] == "draft-mtp"
        idx2 = cmd.index("--spec-draft-n-max")
        assert cmd[idx2 + 1] == "6"

    @pytest.mark.unit
    def test_build_base_cmd_n_parallel(self, base_config):
        base_config["n_parallel"] = 4
        cmd = build_base_cmd(base_config)
        idx = cmd.index("-np")
        assert cmd[idx + 1] == "4"

    @pytest.mark.unit
    def test_build_base_cmd_n_parallel_1_omitted(self, base_config):
        cmd = build_base_cmd(base_config)
        assert "-np" not in cmd

    @pytest.mark.unit
    def test_build_base_cmd_threads(self, base_config):
        base_config["threads"] = 8
        cmd = build_base_cmd(base_config)
        idx = cmd.index("-t")
        assert cmd[idx + 1] == "8"

    @pytest.mark.unit
    def test_build_base_cmd_threads_auto_omitted(self, base_config):
        cmd = build_base_cmd(base_config)
        assert "-t" not in cmd

    @pytest.mark.unit
    def test_build_base_cmd_extra_args(self, base_config):
        base_config["extra_args"] = "--mlock --reasoning-format none"
        cmd = build_base_cmd(base_config)
        assert "--mlock" in cmd
        assert "--reasoning-format" in cmd

    @pytest.mark.unit
    def test_build_base_cmd_mmproj(self, base_config):
        base_config["mmproj_path"] = MMPROJ_PATH
        base_config["mmproj_offload"] = False
        cmd = build_base_cmd(base_config)
        assert "--mmproj" in cmd
        assert MMPROJ_PATH in cmd
        assert "--no-mmproj-offload" in cmd

    @pytest.mark.unit
    def test_build_base_cmd_mmproj_offload_default(self, base_config):
        base_config["mmproj_path"] = MMPROJ_PATH
        cmd = build_base_cmd(base_config)
        assert "--mmproj" in cmd
        assert "--no-mmproj-offload" not in cmd

    @pytest.mark.unit
    def test_build_base_cmd_draft_model(self, base_config):
        base_config["spec_draft_model_path"] = r"C:\fake\draft.gguf"
        cmd = build_base_cmd(base_config)
        idx = cmd.index("--model-draft")
        assert cmd[idx + 1] == r"C:\fake\draft.gguf"

    @pytest.mark.unit
    def test_add_sampling_args(self):
        cmd = []
        params = {"temperature": 0.5, "top_k": 30, "seed": 42}
        add_sampling_args(cmd, params)
        assert "--temp" in cmd
        assert "0.5" in cmd
        assert "--top-k" in cmd
        assert "30" in cmd
        assert "-s" in cmd
        assert "42" in cmd

    @pytest.mark.unit
    def test_add_sampling_args_empty(self):
        cmd = []
        add_sampling_args(cmd, {})
        assert cmd == []

    @pytest.mark.unit
    def test_add_sampling_args_all_params(self):
        cmd = []
        params = {
            "temperature": 0.7, "top_k": 40, "top_p": 0.9, "min_p": 0.1,
            "seed": 123, "repeat_penalty": 1.2, "repeat_last_n": 32,
            "presence_penalty": 0.5, "frequency_penalty": 0.3,
            "typical_p": 0.95, "dynatemp_range": 0.5, "dynatemp_exponent": 1.5,
            "mirostat": 2, "mirostat_tau": 5.0, "mirostat_eta": 0.1,
            "dry_multiplier": 0.8, "dry_base": 1.75,
            "dry_allowed_length": 2, "dry_penalty_last_n": 64,
        }
        add_sampling_args(cmd, params)
        assert len(cmd) == 38  # 19 params * 2 (flag + value)

    @pytest.mark.unit
    def test_add_generation_args_basic(self):
        cmd = []
        params = {"n_predict": 100, "log_disable": True}
        add_generation_args(cmd, params)
        assert "-n" in cmd
        assert "100" in cmd
        assert "--log-disable" in cmd
        assert "--no-display-prompt" in cmd

    @pytest.mark.unit
    def test_add_generation_args_stop_sequences(self):
        cmd = []
        params = {"stop_sequences": '["STOP", "END"]'}
        add_generation_args(cmd, params)
        assert cmd.count("--reverse-prompt") == 2

    @pytest.mark.unit
    def test_add_generation_args_json_schema(self):
        cmd = []
        schema = '{"type": "object"}'
        params = {"json_schema": schema}
        add_generation_args(cmd, params)
        idx = cmd.index("--json-schema")
        assert cmd[idx + 1] == schema

    @pytest.mark.unit
    def test_add_generation_args_grammar(self):
        cmd = []
        params = {"grammar": 'root ::= "hello"'}
        add_generation_args(cmd, params)
        assert "--grammar-file" in cmd
        grammar_path = cmd[cmd.index("--grammar-file") + 1]
        assert os.path.isfile(grammar_path)
        os.unlink(grammar_path)

    @pytest.mark.unit
    def test_add_generation_args_image_path(self):
        cmd = []
        params = {"image_path": r"C:\test\image.png"}
        add_generation_args(cmd, params)
        assert "--image" in cmd


# ===========================================================================
# UNIT TESTS — process_manager
# ===========================================================================

class TestProcessManager:

    @pytest.mark.unit
    def test_singleton(self):
        mgr1 = LlamaProcessManager.get_instance()
        mgr2 = LlamaProcessManager.get_instance()
        assert mgr1 is mgr2

    @pytest.mark.unit
    def test_hash_config_deterministic(self):
        cfg = {"a": 1, "b": "test"}
        h1 = LlamaProcessManager.hash_config(cfg)
        h2 = LlamaProcessManager.hash_config(cfg)
        assert h1 == h2
        assert len(h1) == 16

    @pytest.mark.unit
    def test_hash_config_order_independent(self):
        h1 = LlamaProcessManager.hash_config({"a": 1, "b": 2})
        h2 = LlamaProcessManager.hash_config({"b": 2, "a": 1})
        assert h1 == h2

    @pytest.mark.unit
    def test_hash_config_different(self):
        h1 = LlamaProcessManager.hash_config({"a": 1})
        h2 = LlamaProcessManager.hash_config({"a": 2})
        assert h1 != h2

    @pytest.mark.unit
    def test_get_status_empty(self):
        mgr = LlamaProcessManager.get_instance()
        mgr.shutdown_all()
        status = mgr.get_status()
        assert status == []

    @pytest.mark.unit
    def test_release_nonexistent(self):
        mgr = LlamaProcessManager.get_instance()
        assert mgr.release("nonexistent_hash") is False

    @pytest.mark.unit
    def test_release_by_model_none(self):
        mgr = LlamaProcessManager.get_instance()
        assert mgr.release_by_model("no_such_model") == 0


# ===========================================================================
# UNIT TESTS — config node
# ===========================================================================

class TestCliConfigNode:

    @pytest.mark.unit
    def test_input_types_structure(self):
        types = LlamaCliConfigNode.INPUT_TYPES()
        assert "required" in types
        assert "optional" in types
        assert "binary_dir" in types["required"]
        assert "model_name" in types["required"]
        assert "n_gpu_layers" in types["optional"]
        assert "cache_type_k" in types["optional"]
        assert "spec_type" in types["optional"]
        assert "keep_alive" in types["optional"]
        assert "extra_args" in types["optional"]
        assert "mmproj_name" in types["optional"]

    @pytest.mark.unit
    def test_return_types(self):
        assert LlamaCliConfigNode.RETURN_TYPES == ("CLI_CONFIG",)
        assert LlamaCliConfigNode.FUNCTION == "build_config"
        assert LlamaCliConfigNode.CATEGORY == "AI/LlamaCpp"

    @pytest.mark.unit
    def test_build_config_basic(self):
        node = LlamaCliConfigNode()
        (config,) = node.build_config(
            binary_dir=BINARY_DIR,
            model_name=MODEL_PATH,
            n_gpu_layers=99,
            context_size=512,
        )
        assert config["binary_dir"] == BINARY_DIR
        assert config["model_path"] == MODEL_PATH
        assert config["n_gpu_layers"] == 99
        assert config["context_size"] == 512
        assert config["flash_attention"] is True
        assert config["jinja"] is True
        assert config["keep_alive"] is False

    @pytest.mark.unit
    def test_build_config_full_params(self):
        node = LlamaCliConfigNode()
        (config,) = node.build_config(
            binary_dir=BINARY_DIR,
            model_name=MODEL_PATH,
            n_gpu_layers=40,
            context_size=2048,
            flash_attention=True,
            split_mode="layer",
            main_gpu=1,
            threads=8,
            n_parallel=2,
            cache_type_k="q8_0",
            cache_type_v="q4_0",
            jinja=True,
            spec_type="draft-mtp",
            spec_draft_n_max=6,
            keep_alive=True,
            timeout=120,
            extra_args="--mlock",
        )
        assert config["split_mode"] == "layer"
        assert config["main_gpu"] == 1
        assert config["threads"] == 8
        assert config["n_parallel"] == 2
        assert config["cache_type_k"] == "q8_0"
        assert config["spec_type"] == "draft-mtp"
        assert config["spec_draft_n_max"] == 6
        assert config["keep_alive"] is True
        assert config["extra_args"] == "--mlock"

    @pytest.mark.unit
    def test_build_config_invalid_dir(self):
        node = LlamaCliConfigNode()
        with pytest.raises(ValueError, match="Invalid binary directory"):
            node.build_config(binary_dir=r"C:\nonexistent", model_name="test.gguf")


# ===========================================================================
# UNIT TESTS — node structure validation
# ===========================================================================

class TestNodeStructure:

    @pytest.mark.unit
    @pytest.mark.parametrize("node_cls,func_name", [
        (LlamaGenerateNode, "generate"),
        (LlamaChatNode, "chat"),
        (LlamaEmbeddingNode, "embed"),
        (LlamaInfillNode, "infill"),
        (LlamaProcessMonitorNode, "monitor"),
        (LlamaImageEncoderNode, "encode"),
    ])
    def test_node_has_required_attrs(self, node_cls, func_name):
        assert hasattr(node_cls, "INPUT_TYPES")
        assert hasattr(node_cls, "RETURN_TYPES")
        assert hasattr(node_cls, "FUNCTION")
        assert hasattr(node_cls, "CATEGORY")
        assert node_cls.FUNCTION == func_name
        assert node_cls.CATEGORY == "AI/LlamaCpp"

    @pytest.mark.unit
    @pytest.mark.parametrize("node_cls", [
        LlamaGenerateNode, LlamaChatNode, LlamaEmbeddingNode,
        LlamaInfillNode, LlamaProcessMonitorNode, LlamaImageEncoderNode,
    ])
    def test_input_types_has_required(self, node_cls):
        types = node_cls.INPUT_TYPES()
        assert "required" in types

    @pytest.mark.unit
    def test_generate_node_has_cli_config_input(self):
        types = LlamaGenerateNode.INPUT_TYPES()
        assert "cli_config" in types["required"]

    @pytest.mark.unit
    def test_chat_node_has_user_message_input(self):
        types = LlamaChatNode.INPUT_TYPES()
        assert "user_message" in types["required"]

    @pytest.mark.unit
    def test_embedding_node_has_input_text(self):
        types = LlamaEmbeddingNode.INPUT_TYPES()
        assert "input_text" in types["required"]

    @pytest.mark.unit
    def test_infill_node_has_prefix_suffix(self):
        types = LlamaInfillNode.INPUT_TYPES()
        assert "input_prefix" in types["required"]
        assert "input_suffix" in types["required"]

    @pytest.mark.unit
    def test_image_encoder_has_image_input(self):
        types = LlamaImageEncoderNode.INPUT_TYPES()
        assert "image" in types["required"]

    @pytest.mark.unit
    def test_generate_empty_prompt(self):
        node = LlamaGenerateNode()
        text, err, code = node.generate(cli_config={}, prompt="")
        assert code == -1
        assert "empty" in err.lower()

    @pytest.mark.unit
    def test_chat_empty_message(self):
        node = LlamaChatNode()
        resp, hist, err, code = node.chat(cli_config={}, user_message="")
        assert code == -1
        assert "empty" in err.lower()

    @pytest.mark.unit
    def test_embedding_empty_text(self):
        node = LlamaEmbeddingNode()
        emb, err, code = node.embed(cli_config={}, input_text="")
        assert code == -1

    @pytest.mark.unit
    def test_infill_empty_both(self):
        node = LlamaInfillNode()
        text, err, code = node.infill(cli_config={}, input_prefix="", input_suffix="")
        assert code == -1


# ===========================================================================
# UNIT TESTS — image encoder node
# ===========================================================================

class TestImageEncoderUnit:

    @pytest.mark.unit
    def test_encode_png(self, dummy_image_tensor):
        node = LlamaImageEncoderNode()
        (path,) = node.encode(image=dummy_image_tensor, format="png")
        assert os.path.isfile(path)
        assert path.endswith(".png")
        with Image.open(path) as img:
            assert img.size == (64, 64)
        os.unlink(path)

    @pytest.mark.unit
    def test_encode_jpg(self, dummy_image_tensor):
        node = LlamaImageEncoderNode()
        (path,) = node.encode(image=dummy_image_tensor, format="jpg")
        assert os.path.isfile(path)
        assert path.endswith(".jpg")
        with Image.open(path) as img:
            assert img.size == (64, 64)
        os.unlink(path)


# ===========================================================================
# UNIT TESTS — process monitor node
# ===========================================================================

class TestProcessMonitorNode:

    @pytest.mark.unit
    def test_status_action(self):
        LlamaProcessManager.get_instance().shutdown_all()
        node = LlamaProcessMonitorNode()
        info, released = node.monitor(action="status")
        parsed = json.loads(info)
        assert isinstance(parsed, list)
        assert released is False

    @pytest.mark.unit
    def test_release_all_action(self):
        node = LlamaProcessMonitorNode()
        info, released = node.monitor(action="release_all")
        assert released is True
        assert "release_all" in info

    @pytest.mark.unit
    def test_release_by_model_no_filter(self):
        node = LlamaProcessMonitorNode()
        info, released = node.monitor(action="release_by_model")
        assert released is False
        assert "error" in info.lower()

    @pytest.mark.unit
    def test_release_by_model_with_filter(self):
        node = LlamaProcessMonitorNode()
        info, released = node.monitor(
            action="release_by_model", model_filter="nonexistent.gguf"
        )
        parsed = json.loads(info)
        assert parsed["released_count"] == 0


# ===========================================================================
# UNIT TESTS — chat node history management
# ===========================================================================

class TestChatHistory:

    @pytest.mark.unit
    def test_build_prompt_file_basic(self):
        node = LlamaChatNode()
        result = node._build_prompt_file("You are helpful", [], "Hello")
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["role"] == "system"
        assert parsed[1]["role"] == "user"
        assert parsed[1]["content"] == "Hello"

    @pytest.mark.unit
    def test_build_prompt_file_with_history(self):
        node = LlamaChatNode()
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = node._build_prompt_file("System", history, "How are you?")
        parsed = json.loads(result)
        assert len(parsed) == 4
        assert parsed[0]["role"] == "system"
        assert parsed[1]["role"] == "user"
        assert parsed[2]["role"] == "assistant"
        assert parsed[3]["content"] == "How are you?"

    @pytest.mark.unit
    def test_build_prompt_file_no_system(self):
        node = LlamaChatNode()
        result = node._build_prompt_file("", [], "Hello")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["role"] == "user"

    @pytest.mark.unit
    def test_write_temp_prompt(self):
        node = LlamaChatNode()
        path = node._write_temp_prompt('{"test": true}')
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as f:
            assert json.loads(f.read()) == {"test": True}
        os.unlink(path)


# ===========================================================================
# INTEGRATION TESTS — text generation (Gemma 4B)
# ===========================================================================

@pytest.mark.integration
class TestGenerateIntegration:

    def test_basic_generation(self, integration_config):
        """Basic text completion with short output."""
        node = LlamaGenerateNode()
        text, stderr, code = node.generate(
            cli_config=integration_config,
            prompt="The capital of France is",
            n_predict=10,
            temperature=0.1,
            seed=42,
        )
        print(f"\n[Generate] text={text!r}, code={code}")
        assert code == 0, f"llama-cli failed: {stderr[:500]}"
        assert len(text) > 0

    def test_generation_with_seed_determinism(self, integration_config):
        """Same seed should produce same output."""
        node = LlamaGenerateNode()
        results = []
        for _ in range(2):
            text, stderr, code = node.generate(
                cli_config=integration_config,
                prompt="Count: 1, 2, 3,",
                n_predict=10,
                temperature=0.0,
                seed=12345,
            )
            assert code == 0, f"llama-cli failed: {stderr[:500]}"
            results.append(text)
        print(f"\n[Seed test] r1={results[0]!r}, r2={results[1]!r}")
        assert results[0] == results[1], "Same seed should produce identical output"

    def test_generation_with_json_schema(self, integration_config):
        """Constrained generation with JSON schema."""
        schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        })
        node = LlamaGenerateNode()
        text, stderr, code = node.generate(
            cli_config=integration_config,
            prompt="Generate JSON for a person named Alice age 30:",
            n_predict=30,
            temperature=0.1,
            json_schema=schema,
        )
        print(f"\n[JSON Schema] text={text!r}, code={code}")
        assert code == 0, f"llama-cli failed: {stderr[:500]}"
        parsed = json.loads(text)
        assert "name" in parsed
        assert "age" in parsed

    def test_generation_stderr_has_timing(self, integration_config):
        """stderr should contain timing stats when log_disable=False."""
        node = LlamaGenerateNode()
        text, stderr, code = node.generate(
            cli_config=integration_config,
            prompt="Hello",
            n_predict=5,
            temperature=0.1,
            log_disable=False,
        )
        assert code == 0, f"llama-cli failed: {stderr[:500]}"
        stats = parse_timing_stats(stderr)
        print(f"\n[Timing] stats={stats}")
        assert stats is not None, f"No timing stats in stderr: {stderr[:300]}"


# ===========================================================================
# INTEGRATION TESTS — chat (Gemma 4B)
# ===========================================================================

@pytest.mark.integration
class TestChatIntegration:

    def test_basic_chat(self, integration_config):
        """Single-turn chat."""
        node = LlamaChatNode()
        response, history, stderr, code = node.chat(
            cli_config=integration_config,
            user_message="Say hello in exactly 3 words.",
            system_message="You are a helpful assistant. Reply concisely.",
            max_tokens=15,
            temperature=0.1,
        )
        print(f"\n[Chat] response={response!r}, code={code}")
        assert code == 0, f"llama-cli failed: {stderr[:500]}"
        assert len(response) > 0

    def test_chat_history_output(self, integration_config):
        """Chat should output valid updated history JSON."""
        node = LlamaChatNode()
        response, history_json, stderr, code = node.chat(
            cli_config=integration_config,
            user_message="What is 2+2?",
            system_message="Answer with just the number.",
            max_tokens=5,
            temperature=0.0,
        )
        assert code == 0, f"llama-cli failed: {stderr[:500]}"
        history = json.loads(history_json)
        assert isinstance(history, list)
        assert any(m["role"] == "system" for m in history)
        assert any(m["role"] == "user" for m in history)
        if response:
            assert any(m["role"] == "assistant" for m in history)
        print(f"\n[Chat History] {json.dumps(history, indent=2, ensure_ascii=False)}")

    def test_chat_with_prior_history(self, integration_config):
        """Chat with existing conversation history."""
        node = LlamaChatNode()
        prior_history = json.dumps([
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice!"},
        ])
        response, history_json, stderr, code = node.chat(
            cli_config=integration_config,
            user_message="What is my name?",
            chat_history=prior_history,
            max_tokens=10,
            temperature=0.1,
        )
        print(f"\n[Chat w/ History] response={response!r}, code={code}")
        assert code == 0, f"llama-cli failed: {stderr[:500]}"
        history = json.loads(history_json)
        assert len(history) >= 4


# ===========================================================================
# INTEGRATION TESTS — vision / multimodal (Gemma 4B + mmproj)
# ===========================================================================

@pytest.mark.integration
class TestVisionIntegration:

    def test_generate_with_image(self, vision_config, test_image_path):
        """Text generation with an image input (multimodal)."""
        node = LlamaGenerateNode()
        text, stderr, code = node.generate(
            cli_config=vision_config,
            prompt="Describe this image briefly.",
            n_predict=20,
            temperature=0.1,
            image_path=test_image_path,
        )
        print(f"\n[Vision Generate] text={text!r}, code={code}")
        assert code == 0, f"llama-cli failed: {stderr[:500]}"
        assert len(text) > 0

    def test_chat_with_image(self, vision_config, test_image_path):
        """Chat with an image input (multimodal chat)."""
        node = LlamaChatNode()
        response, history, stderr, code = node.chat(
            cli_config=vision_config,
            user_message="What do you see in this image?",
            system_message="Describe images concisely.",
            max_tokens=20,
            temperature=0.1,
            image_path=test_image_path,
        )
        print(f"\n[Vision Chat] response={response!r}, code={code}")
        assert code == 0, f"llama-cli failed: {stderr[:500]}"
        assert len(response) > 0

    def test_image_encoder_to_generate_pipeline(self, vision_config, dummy_image_tensor):
        """Full pipeline: ImageEncoder → Generate with image."""
        enc_node = LlamaImageEncoderNode()
        (image_path,) = enc_node.encode(image=dummy_image_tensor, format="png")
        assert os.path.isfile(image_path)

        gen_node = LlamaGenerateNode()
        text, stderr, code = gen_node.generate(
            cli_config=vision_config,
            prompt="What colors do you see?",
            n_predict=15,
            temperature=0.1,
            image_path=image_path,
        )
        print(f"\n[Encoder→Generate] text={text!r}, code={code}")
        assert code == 0, f"llama-cli failed: {stderr[:500]}"
        assert len(text) > 0

        try:
            os.unlink(image_path)
        except OSError:
            pass

    def test_mmproj_in_command(self, vision_config):
        """Verify mmproj flag appears in the built command."""
        cmd = build_base_cmd(vision_config)
        assert "--mmproj" in cmd
        assert MMPROJ_PATH in cmd


# ===========================================================================
# INTEGRATION TESTS — process manager
# ===========================================================================

@pytest.mark.integration
class TestProcessManagerIntegration:

    def test_oneshot_execution(self, integration_config):
        """ProcessManager.run_oneshot runs and exits cleanly."""
        mgr = LlamaProcessManager.get_instance()
        cmd = build_base_cmd(integration_config, binary_name="llama-completion")
        cmd.extend(["-no-cnv", "-p", "Hi", "-n", "5", "--no-display-prompt"])
        stdout, stderr, code = mgr.run_oneshot(cmd, timeout=integration_config["timeout"])
        print(f"\n[Oneshot] stdout={stdout!r}, code={code}")
        assert code == 0, f"One-shot failed: {stderr[:500]}"
        assert len(stdout.strip()) > 0

    def test_oneshot_bad_binary(self):
        """run_oneshot with non-existent binary returns error gracefully."""
        mgr = LlamaProcessManager.get_instance()
        stdout, stderr, code = mgr.run_oneshot(
            [r"C:\nonexistent\llama-cli.exe", "-p", "test"], timeout=10
        )
        assert code == -1
        assert "not found" in stderr.lower() or "failed" in stderr.lower()


# ===========================================================================
# INTEGRATION TEST — end-to-end pipelines
# ===========================================================================

@pytest.mark.integration
class TestEndToEnd:

    def test_config_to_generate_pipeline(self):
        """Full pipeline: ConfigNode → GenerateNode."""
        config_node = LlamaCliConfigNode()
        (config,) = config_node.build_config(
            binary_dir=BINARY_DIR,
            model_name=MODEL_PATH,
            n_gpu_layers=99,
            context_size=512,
            flash_attention=True,
            jinja=True,
            timeout=120,
        )

        gen_node = LlamaGenerateNode()
        text, stderr, code = gen_node.generate(
            cli_config=config,
            prompt="Write one word:",
            n_predict=5,
            temperature=0.7,
            seed=42,
        )
        print(f"\n[E2E Generate] text={text!r}, code={code}")
        assert code == 0, f"Pipeline failed: {stderr[:500]}"
        assert len(text) > 0

    def test_config_to_chat_pipeline(self):
        """Full pipeline: ConfigNode → ChatNode."""
        config_node = LlamaCliConfigNode()
        (config,) = config_node.build_config(
            binary_dir=BINARY_DIR,
            model_name=MODEL_PATH,
            n_gpu_layers=99,
            context_size=512,
            flash_attention=True,
            jinja=True,
            timeout=120,
        )

        chat_node = LlamaChatNode()
        response, history, stderr, code = chat_node.chat(
            cli_config=config,
            user_message="Say hi.",
            system_message="Be concise.",
            max_tokens=10,
            temperature=0.3,
        )
        print(f"\n[E2E Chat] response={response!r}, code={code}")
        assert code == 0, f"Pipeline failed: {stderr[:500]}"
        assert len(response) > 0
        parsed_hist = json.loads(history)
        assert len(parsed_hist) >= 2

    def test_config_to_vision_chat_pipeline(self, test_image_path):
        """Full pipeline: ConfigNode (with mmproj) → ChatNode with image."""
        config_node = LlamaCliConfigNode()
        (config,) = config_node.build_config(
            binary_dir=BINARY_DIR,
            model_name=MODEL_PATH,
            n_gpu_layers=99,
            context_size=512,
            flash_attention=True,
            jinja=True,
            timeout=120,
        )
        config["mmproj_path"] = MMPROJ_PATH

        chat_node = LlamaChatNode()
        response, history, stderr, code = chat_node.chat(
            cli_config=config,
            user_message="What is in this image?",
            max_tokens=15,
            temperature=0.1,
            image_path=test_image_path,
        )
        print(f"\n[E2E Vision] response={response!r}, code={code}")
        assert code == 0, f"Vision pipeline failed: {stderr[:500]}"
        assert len(response) > 0

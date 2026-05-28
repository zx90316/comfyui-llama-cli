import re
import json
from typing import Optional


def parse_generation_output(stdout: str, stderr: str = "") -> str:
    """Extract generated text from llama-completion stdout.

    With --no-display-prompt, stdout should be purely generated text.
    We strip leading/trailing whitespace and any trailing EOT markers.
    """
    text = stdout
    text = re.sub(r"\s*\[end of text\]\s*$", "", text)
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)

    eot_markers = ["<|eot_id|>", "<|end|>", "</s>", "<|im_end|>", "<|endoftext|>"]
    for marker in eot_markers:
        if text.endswith(marker):
            text = text[: -len(marker)]

    return text.strip()


def parse_chat_response(stdout: str) -> str:
    """Extract assistant response from llama-cli -cnv conversation-mode output.

    llama-cli -cnv outputs a header (ASCII art, model info, commands),
    then echoes the prompt after '> ', followed by the actual response,
    then timing stats and 'Exiting...'. We extract only the response.
    """
    text = stdout

    prompt_marker = re.search(r"\n> .+\n", text)
    if prompt_marker:
        text = text[prompt_marker.end():]

    text = re.sub(r"\n\[ Prompt:.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\nExiting\.\.\.\s*$", "", text)
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)

    eot_markers = ["<|eot_id|>", "<|end|>", "</s>", "<|im_end|>", "<|endoftext|>"]
    for marker in eot_markers:
        if text.endswith(marker):
            text = text[: -len(marker)]

    return text.strip()


def parse_embedding_output(stdout: str) -> str:
    """Parse embedding vector output from llama-embedding.

    llama-embedding outputs the embedding as space-separated floats.
    We convert to a JSON array for easier downstream use.
    """
    text = stdout.strip()
    if not text:
        return "[]"

    try:
        values = [float(x) for x in text.split()]
        return json.dumps(values)
    except ValueError:
        return text


def parse_timing_stats(stderr: str) -> Optional[dict]:
    """Extract timing stats from llama-cli stderr output.

    Looks for lines like:
        llama_perf_sampler_print:    sampling time =   ...
        llama_perf_context_print:        load time =   ...
        llama_perf_context_print: prompt eval time =   ...
        llama_perf_context_print:        eval time =   ...
    """
    stats = {}

    prompt_eval = re.search(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
        stderr,
    )
    if prompt_eval:
        stats["prompt_eval_time_ms"] = float(prompt_eval.group(1))
        stats["prompt_eval_tokens"] = int(prompt_eval.group(2))
        stats["prompt_eval_tokens_per_second"] = float(prompt_eval.group(4))

    eval_match = re.search(
        r"(?<!prompt\s)eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
        stderr,
    )
    if eval_match:
        stats["eval_time_ms"] = float(eval_match.group(1))
        stats["eval_tokens"] = int(eval_match.group(2))
        stats["eval_tokens_per_second"] = float(eval_match.group(4))

    load_match = re.search(r"load time\s*=\s*([\d.]+)\s*ms", stderr)
    if load_match:
        stats["load_time_ms"] = float(load_match.group(1))

    return stats if stats else None

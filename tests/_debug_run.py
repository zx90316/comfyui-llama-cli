import subprocess
import time
import sys
import os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BINARY = r"C:\Users\zx020\project\llama-cpp-turboquant\build\bin\Release\llama-cli.exe"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(
    PROJECT_ROOT, "models",
    "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf",
)
MMPROJ = os.path.join(
    PROJECT_ROOT, "models",
    "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf",
)

tests = [
    ("basic-generate", [
        BINARY, "-m", MODEL, "-ngl", "99", "-c", "512", "-fa", "on",
        "--split-mode", "none", "--main-gpu", "0",
        "-p", "The capital of France is", "-n", "10",
        "--no-display-prompt", "--log-disable",
    ]),
    ("vision-generate", [
        BINARY, "-m", MODEL, "--mmproj", MMPROJ,
        "-ngl", "99", "-c", "512", "-fa", "on",
        "--split-mode", "none", "--main-gpu", "0",
        "-p", "Describe this image.", "-n", "15",
        "--no-display-prompt", "--log-disable",
    ]),
]

for name, cmd in tests:
    print(f"\n=== {name} ===")
    sys.stdout.flush()
    t = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                           timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
        elapsed = time.time() - t
        print(f"[{elapsed:.1f}s] CODE={r.returncode}")
        print(f"STDOUT ({len(r.stdout)} chars):")
        print(repr(r.stdout[:500]))
        print(f"STDERR ({len(r.stderr)} chars):")
        print(repr(r.stderr[:300]))
    except subprocess.TimeoutExpired:
        print(f"[{time.time()-t:.1f}s] TIMEOUT")
    sys.stdout.flush()

print("\n=== DONE ===")

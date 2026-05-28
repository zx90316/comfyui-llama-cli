import subprocess
import hashlib
import json
import atexit
import threading
import time
import os
import logging
from typing import Optional

logger = logging.getLogger("comfyui-llama-cli")


class InteractiveProcess:
    """Wraps a llama-cli --interactive process for keep-alive VRAM reuse."""

    READY_MARKER = "\n> "

    def __init__(self, cmd: list, model_name: str = ""):
        self._cmd = cmd
        self._process: Optional[subprocess.Popen] = None
        self._model_name = model_name
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()

    def start(self):
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._process = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        self._start_time = time.time()
        self._wait_for_ready()

    def _wait_for_ready(self, timeout: int = 300):
        """Block until llama-cli prints its interactive prompt ('>') on stderr/stdout."""
        if not self._process:
            raise RuntimeError("Process not started")

        deadline = time.time() + timeout
        buf = []
        while time.time() < deadline:
            if self._process.poll() is not None:
                stderr_out = self._process.stderr.read() if self._process.stderr else ""
                raise RuntimeError(
                    f"llama-cli exited during startup (code {self._process.returncode}): {stderr_out[:500]}"
                )
            char = self._process.stdout.read(1)
            if not char:
                time.sleep(0.05)
                continue
            buf.append(char)
            if len(buf) >= 2 and buf[-2] == "\n" and buf[-1] == ">":
                space = self._process.stdout.read(1)
                return
        raise TimeoutError("llama-cli interactive mode did not become ready in time")

    def send_prompt(self, prompt: str, timeout: int = 600) -> tuple:
        """Write prompt to stdin, read stdout until the next interactive prompt marker."""
        with self._lock:
            if not self.is_alive():
                raise RuntimeError("Interactive process is not running")

            self._process.stdin.write(prompt + "\n")
            self._process.stdin.flush()

            deadline = time.time() + timeout
            buf = []
            while time.time() < deadline:
                if self._process.poll() is not None:
                    break
                char = self._process.stdout.read(1)
                if not char:
                    time.sleep(0.01)
                    continue
                buf.append(char)
                if len(buf) >= 2 and buf[-2] == "\n" and buf[-1] == ">":
                    peek = self._process.stdout.read(1)
                    if peek == " ":
                        text = "".join(buf[:-2])
                        return text.strip(), "", 0
                    else:
                        buf.append(peek)

            text = "".join(buf).strip()
            if self._process.poll() is not None:
                return text, f"Process exited with code {self._process.returncode}", self._process.returncode
            return text, "Timeout waiting for response", -1

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def terminate(self):
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=10)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
        self._process = None

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def uptime_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time


class LlamaProcessManager:
    """Singleton that manages llama-cli subprocess lifecycles."""

    _instance: Optional["LlamaProcessManager"] = None
    _init_lock = threading.Lock()

    def __init__(self):
        self._processes: dict[str, InteractiveProcess] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "LlamaProcessManager":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    atexit.register(cls._instance.shutdown_all)
        return cls._instance

    @staticmethod
    def hash_config(config: dict) -> str:
        stable = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(stable.encode()).hexdigest()[:16]

    def run_oneshot(self, cmd: list, timeout: int = 600) -> tuple:
        """Run a one-shot subprocess. Returns (stdout, stderr, returncode)."""
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                creationflags=creationflags,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", f"Process timed out after {timeout}s", -1
        except FileNotFoundError as e:
            return "", f"Binary not found: {e}", -1
        except Exception as e:
            return "", f"Failed to run process: {e}", -1

    def get_interactive(self, cmd: list, config: dict) -> InteractiveProcess:
        """Get or create a keep-alive interactive process for the given config."""
        config_hash = self.hash_config(config)
        with self._lock:
            existing = self._processes.get(config_hash)
            if existing and existing.is_alive():
                return existing

            if existing:
                existing.terminate()

            proc = InteractiveProcess(cmd, model_name=config.get("model_name", ""))
            proc.start()
            self._processes[config_hash] = proc
            return proc

    def release(self, config_hash: str) -> bool:
        """Terminate and remove a specific interactive process."""
        with self._lock:
            proc = self._processes.pop(config_hash, None)
            if proc:
                proc.terminate()
                return True
            return False

    def release_by_model(self, model_name: str) -> int:
        """Terminate all interactive processes using a specific model. Returns count released."""
        released = 0
        with self._lock:
            to_remove = []
            for h, proc in self._processes.items():
                if proc.model_name == model_name:
                    proc.terminate()
                    to_remove.append(h)
                    released += 1
            for h in to_remove:
                del self._processes[h]
        return released

    def shutdown_all(self):
        """Terminate all interactive processes."""
        with self._lock:
            for proc in self._processes.values():
                try:
                    proc.terminate()
                except Exception as e:
                    logger.warning(f"Failed to terminate process: {e}")
            self._processes.clear()
        logger.info("All llama-cli processes shut down")

    def get_status(self) -> list:
        """Return status info for all active interactive processes."""
        with self._lock:
            result = []
            for config_hash, proc in self._processes.items():
                result.append({
                    "config_hash": config_hash,
                    "pid": proc.pid,
                    "model_name": proc.model_name,
                    "alive": proc.is_alive(),
                    "uptime_seconds": round(proc.uptime_seconds, 1),
                })
            return result

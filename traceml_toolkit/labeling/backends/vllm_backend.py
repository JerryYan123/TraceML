"""vLLM backend: one worker process per generate() call, judged by its output file.

vLLM runs its engine core in a child process. When the worker exits (or hangs
in teardown and is killed), that child can survive as an orphan and keep its
GPU memory, so the next engine on the same GPU finds no room for its KV cache.
Each worker therefore runs in its own process group, and the whole group is
terminated after every call.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WORKER = Path(__file__).with_name("vllm_worker.py")
# Engine start-up failures caused by other processes on a shared GPU.
TRANSIENT_ERRORS = ("No available memory for the cache blocks", "CUDA out of memory",
                    "Free memory on device")


def _kill_group(proc: subprocess.Popen) -> None:
    for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 0)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.time() + wait
        while time.time() < deadline:
            try:
                os.killpg(proc.pid, 0)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.5)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


class VllmBackend:
    name = "vllm"

    def __init__(self, model_dir: Path, *, gpu: int | None = None, gpu_mem_util: float = 0.5,
                 tensor_parallel: int = 1, python: str | None = None, log_dir: Path | None = None,
                 timeout_s: int = 3600, start_retries: int = 2):
        self.model_dir = Path(model_dir)
        self.gpu = gpu
        self.gpu_mem_util = gpu_mem_util
        self.tensor_parallel = tensor_parallel
        self.python = python or os.environ.get("TRACEML_VLLM_PYTHON") or sys.executable
        self.log_dir = Path(log_dir) if log_dir else None
        self.timeout_s = timeout_s
        self.start_retries = start_retries
        self._calls = 0

    def _run_worker(self, inp: Path, out: Path, log_path: Path, max_new_tokens: int,
                    max_model_len: int) -> int | None:
        cmd = [self.python, str(WORKER), "--model", str(self.model_dir),
               "--input", str(inp), "--output", str(out),
               "--max-model-len", str(max_model_len), "--max-new-tokens", str(max_new_tokens),
               "--gpu-mem-util", str(self.gpu_mem_util), "--tensor-parallel", str(self.tensor_parallel)]
        env = os.environ.copy()
        if self.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        with log_path.open("a") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            deadline = time.time() + self.timeout_s
            try:
                while time.time() < deadline and not out.exists():
                    if proc.poll() is not None:
                        time.sleep(1)  # output is renamed into place just before exit
                        break
                    time.sleep(2)
                # give the worker a moment to exit on its own, then reap the whole group
                end = time.time() + 10
                while proc.poll() is None and time.time() < end:
                    time.sleep(0.5)
            finally:
                _kill_group(proc)
        return proc.returncode

    def generate(self, prompts: list[str], *, max_new_tokens: int, max_model_len: int) -> list[str]:
        if not prompts:
            return []
        self._calls += 1
        work = Path(tempfile.mkdtemp(prefix="traceml_vllm_"))
        inp, out = work / "prompts.jsonl", work / "generations.jsonl"
        with inp.open("w") as fh:
            for i, p in enumerate(prompts):
                fh.write(json.dumps({"idx": i, "prompt": p}) + "\n")
        log_path = (self.log_dir / f"vllm_{self.model_dir.parent.name}_{self._calls:02d}.log"
                    if self.log_dir else work / "vllm.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")
        rc = None
        for attempt in range(self.start_retries + 1):
            rc = self._run_worker(inp, out, log_path, max_new_tokens, max_model_len)
            if out.exists():
                break
            log_text = log_path.read_text(errors="replace")
            if attempt < self.start_retries and any(e in log_text for e in TRANSIENT_ERRORS):
                print(f"[traceml] vLLM could not reserve GPU memory (shared GPU?); retrying in 20 s "
                      f"({attempt + 1}/{self.start_retries})", flush=True)
                time.sleep(20)
                continue
            break
        if not out.exists():
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-25:])
            raise RuntimeError(f"vLLM worker produced no output (exit code {rc}); log: {log_path}\n{tail}\n"
                               "If the GPU is shared, pick a freer one with --gpu or lower --gpu-mem-util.")
        texts = [""] * len(prompts)
        for ln in out.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                texts[int(r["idx"])] = r.get("text", "")
        if self.log_dir:  # the log lives outside the scratch dir
            shutil.rmtree(work, ignore_errors=True)
        return texts

    def close(self) -> None:
        pass

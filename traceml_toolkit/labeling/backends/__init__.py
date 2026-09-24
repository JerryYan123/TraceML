"""Generation backends for the labelers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from traceml_toolkit.labeling.backends.base import LabelBackend


def vllm_available() -> bool:
    """vLLM importable in this interpreter (or an external one via TRACEML_VLLM_PYTHON) and a GPU present."""
    import os
    import shutil
    if os.environ.get("TRACEML_VLLM_PYTHON"):
        return True
    if importlib.util.find_spec("vllm") is None:
        return False
    return shutil.which("nvidia-smi") is not None


def transformers_available() -> bool:
    return (importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("transformers") is not None)


def resolve_backend_name(name: str) -> str:
    if name == "auto":
        if vllm_available():
            return "vllm"
        if transformers_available():
            return "transformers"
        raise RuntimeError(
            "no labeling backend available. On a CUDA machine: pip install 'traceml-toolkit[label]' "
            "(vLLM). Elsewhere: pip install 'traceml-toolkit[transformers]' (slow). "
            "Or skip labeling with --skip-label.")
    if name not in ("vllm", "transformers"):
        raise ValueError(f"unknown backend {name!r}")
    return name


def get_backend(name: str, model_dir: Path, *, gpu: int | None = None, gpu_mem_util: float = 0.5,
                log_dir: Path | None = None) -> LabelBackend:
    name = resolve_backend_name(name)
    if name == "vllm":
        from traceml_toolkit.labeling.backends.vllm_backend import VllmBackend
        return VllmBackend(model_dir, gpu=gpu, gpu_mem_util=gpu_mem_util, log_dir=log_dir)
    from traceml_toolkit.labeling.backends.hf_backend import TransformersBackend
    return TransformersBackend(model_dir, gpu=gpu)

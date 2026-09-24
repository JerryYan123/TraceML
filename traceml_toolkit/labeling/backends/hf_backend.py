"""Plain `transformers` backend: greedy decoding one prompt at a time.

Useful where vLLM is unavailable (Apple silicon, CPU, Windows). It is much
slower than vLLM and bf16 kernels differ, so labels can differ slightly from
the vLLM backend's; see the README for the measured agreement.
"""
from __future__ import annotations

from pathlib import Path


class TransformersBackend:
    name = "transformers"

    def __init__(self, model_dir: Path, *, gpu: int | None = None, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            if torch.cuda.is_available():
                device = f"cuda:{gpu or 0}"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=dtype).to(device).eval()
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos = [t for t in {im_end, self.tokenizer.eos_token_id} if isinstance(t, int) and t >= 0]
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos[0]
        self._eos, self._pad = eos, pad

    def generate(self, prompts: list[str], *, max_new_tokens: int, max_model_len: int) -> list[str]:
        # Decoding options go in as keyword arguments: a GenerationConfig object would
        # have its default-valued fields (do_sample=False) replaced by the checkpoint's
        # sampling defaults (temperature 0.6), silently turning greedy decoding off.
        greedy = dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                      top_k=None, eos_token_id=self._eos, pad_token_id=self._pad)
        out = []
        for p in prompts:
            enc = self.tokenizer(p, return_tensors="pt", add_special_tokens=False).to(self.device)
            with self.torch.no_grad():
                g = self.model.generate(**enc, **greedy)
            new = g[0, enc["input_ids"].shape[1]:]
            out.append(self.tokenizer.decode(new, skip_special_tokens=True))
        return out

    def close(self) -> None:
        del self.model
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
